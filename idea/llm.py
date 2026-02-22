from abc import ABC
import random
import secrets
import os
import re
import socket
from google import genai
from google.genai import types
import json
from datetime import datetime, timezone
from pydantic import BaseModel
from typing import Type, Optional, Dict, List, Callable, Tuple, Set, Any
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential_jitter, retry_if_exception
import numpy as np
from idea.models import Idea
from idea.prompts.loader import get_prompts, get_prompts_from_dict
import uuid
from collections import OrderedDict
from hashlib import sha256
import threading
from idea.ratings import parallel_evaluate_pairs
from idea.swiss_tournament import (
    elo_update,
    generate_swiss_round_pairs,
    pair_players_swiss,
    select_swiss_bye,
)

try:
    from google.api_core import exceptions as gapi_exceptions
except Exception:  # pragma: no cover - optional dependency guard
    gapi_exceptions = None

TRANSIENT_HTTP_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
NON_RETRYABLE_HTTP_STATUS_CODES = {400, 401, 403, 404, 405, 410, 411, 412, 413, 414, 415, 422}
NON_RETRYABLE_MARKERS = (
    "invalid argument",
    "permission denied",
    "forbidden",
    "unauthorized",
    "not found",
    "failed precondition",
)
TRANSIENT_MARKERS = (
    "rate limit",
    "too many requests",
    "resource exhausted",
    "temporarily unavailable",
    "deadline exceeded",
    "timed out",
    "timeout",
    "connection reset",
    "service unavailable",
    "internal error",
    "bad gateway",
    "gateway timeout",
)

THINKING_LEVEL_TO_BUDGET_FALLBACK = {
    "off": 0,
    "low": 1024,
    "medium": 8192,
    "high": -1,  # Dynamic/high-think fallback
}


def _extract_http_status(exc: BaseException) -> Optional[int]:
    """Best-effort extraction of HTTP status from provider exceptions."""
    status_candidates: List[Any] = []

    for attr_name in ("status_code", "status", "code", "http_status"):
        value = getattr(exc, attr_name, None)
        if value is not None:
            status_candidates.append(value)

    response = getattr(exc, "response", None)
    if response is not None:
        status_candidates.append(getattr(response, "status_code", None))
        status_obj = getattr(response, "status", None)
        if status_obj is not None:
            status_candidates.append(getattr(status_obj, "code", None))

    for candidate in status_candidates:
        if candidate is None:
            continue
        try:
            parsed = int(candidate)
        except Exception:
            continue
        if 100 <= parsed <= 599:
            return parsed

    return None


def _is_transient_generation_error(exc: BaseException) -> bool:
    """Only retry provider/network failures that are likely to succeed on retry."""
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError)):
        return True

    if gapi_exceptions is not None:
        transient_types = (
            gapi_exceptions.TooManyRequests,
            gapi_exceptions.ResourceExhausted,
            gapi_exceptions.InternalServerError,
            gapi_exceptions.ServiceUnavailable,
            gapi_exceptions.GatewayTimeout,
            gapi_exceptions.DeadlineExceeded,
            gapi_exceptions.Aborted,
        )
        non_retryable_types = (
            gapi_exceptions.InvalidArgument,
            gapi_exceptions.PermissionDenied,
            gapi_exceptions.Unauthenticated,
            gapi_exceptions.NotFound,
            gapi_exceptions.FailedPrecondition,
        )
        if isinstance(exc, transient_types):
            return True
        if isinstance(exc, non_retryable_types):
            return False

    status_code = _extract_http_status(exc)
    if status_code in TRANSIENT_HTTP_STATUS_CODES:
        return True
    if status_code in NON_RETRYABLE_HTTP_STATUS_CODES:
        return False

    message = str(exc).lower()
    if any(marker in message for marker in NON_RETRYABLE_MARKERS):
        return False
    if any(marker in message for marker in TRANSIENT_MARKERS):
        return True

    return False

class LLMWrapper(ABC):
    """Base class for LLM interactions"""
    MAX_TOKENS = 8192
    MAX_DIAGNOSTIC_EVENTS = 200

    def __init__(self,
                 provider: str = "google_generative_ai",
                 model_name: str = "gemini-2.0-flash",
                 prompt_template: str = "",
                 temperature: float = 1.0,
                 top_p: float = 0.95,
                 agent_name: str = "",
                 thinking_budget: Optional[int] = None,
                 thinking_level: Optional[str] = None,
                 api_key: Optional[str] = None,
                 random_seed: Optional[int] = None):
        self.provider = provider
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_budget = thinking_budget
        self.thinking_level = (
            str(thinking_level).strip().lower() if thinking_level is not None else None
        )
        self.api_key = api_key
        self.total_token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0
        self.agent_name = agent_name
        print(
            f"Initializing {agent_name or 'LLM'} with temperature: {temperature}, top_p: {top_p}, "
            f"thinking_budget: {thinking_budget}, thinking_level: {self.thinking_level}"
        )
        self._custom_templates: Dict[str, Dict[str, Any]] = {}
        self._custom_prompt_wrappers: Dict[str, Any] = {}
        self._custom_template_lock = threading.Lock()
        self._token_lock = threading.Lock()
        self._diagnostics_lock = threading.Lock()
        self._rng_lock = threading.Lock()
        self._diagnostics: Dict[str, int] = {}
        self._diagnostic_events: List[Dict[str, Any]] = []

        self.random_seed = int(random_seed) if random_seed is not None else int(secrets.randbits(64))
        self._rng = random.Random(self.random_seed)

        self.client = None
        self._resolved_api_key: Optional[str] = None
        self._thread_local_client = threading.local()
        self._client_lock = threading.Lock()
        timeout_raw = os.environ.get("GENAI_HTTP_TIMEOUT_MS", "60000")
        try:
            self._http_timeout_ms = max(1000, int(timeout_raw))
        except ValueError:
            self._http_timeout_ms = 60000
            self._increment_diagnostic(
                "invalid_http_timeout_ms",
                detail=f"GENAI_HTTP_TIMEOUT_MS={timeout_raw!r}",
            )
        self._http_options = types.HttpOptions(timeout=self._http_timeout_ms)
        self._setup_provider()

    def register_custom_template(self, template_id: str, template_data: Dict[str, Any]) -> None:
        """Register template data scoped to this LLM instance."""
        with self._custom_template_lock:
            self._custom_templates[template_id] = template_data
            self._custom_prompt_wrappers.pop(template_id, None)

    def _iter_custom_templates(self) -> List[Tuple[str, Dict[str, Any]]]:
        with self._custom_template_lock:
            return list(self._custom_templates.items())

    def _resolve_prompts(self, idea_type: str):
        with self._custom_template_lock:
            template_data = self._custom_templates.get(idea_type)
            cached_wrapper = self._custom_prompt_wrappers.get(idea_type)

        if template_data is None:
            return get_prompts(idea_type)

        if cached_wrapper is not None:
            return cached_wrapper

        wrapper = get_prompts_from_dict(template_data)
        with self._custom_template_lock:
            self._custom_prompt_wrappers[idea_type] = wrapper
        return wrapper

    def _increment_diagnostic(
        self,
        key: str,
        delta: int = 1,
        *,
        detail: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        with self._diagnostics_lock:
            self._diagnostics[key] = self._diagnostics.get(key, 0) + int(delta)
            if detail is not None or metadata is not None:
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "key": key,
                    "detail": str(detail) if detail is not None else "",
                    "agent": self.agent_name or self.__class__.__name__,
                }
                if metadata:
                    event["metadata"] = dict(metadata)
                self._diagnostic_events.append(event)
                if len(self._diagnostic_events) > self.MAX_DIAGNOSTIC_EVENTS:
                    self._diagnostic_events = self._diagnostic_events[-self.MAX_DIAGNOSTIC_EVENTS :]

    def get_diagnostics(self) -> Dict[str, int]:
        with self._diagnostics_lock:
            return dict(self._diagnostics)

    def get_diagnostic_events(self) -> List[Dict[str, Any]]:
        with self._diagnostics_lock:
            return [dict(event) for event in self._diagnostic_events]

    def _random_sample(self, population: List[Any], k: int) -> List[Any]:
        with self._rng_lock:
            return self._rng.sample(population, k)

    def _random_shuffle(self, seq: List[Any]) -> None:
        with self._rng_lock:
            self._rng.shuffle(seq)

    def _random_uniform(self, a: float, b: float) -> float:
        with self._rng_lock:
            return self._rng.uniform(a, b)

    def _random_randbits(self, bits: int) -> int:
        with self._rng_lock:
            return self._rng.getrandbits(bits)

    def _get_client(self):
        if self.provider != "google_generative_ai":
            return self.client

        thread_client = getattr(self._thread_local_client, "client", None)
        if thread_client is not None:
            return thread_client

        api_key = self._resolved_api_key
        if not api_key:
            return self.client

        thread_client = genai.Client(api_key=api_key, http_options=self._http_options)
        self._thread_local_client.client = thread_client
        return thread_client

    def _setup_provider(self):
        if self.provider == "google_generative_ai":
            api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            self._resolved_api_key = api_key
            if not api_key:
                print("Warning: GEMINI_API_KEY not set and no api_key provided")

            # Initialize client once
            with self._client_lock:
                if self.client is None and api_key:
                    self.client = genai.Client(api_key=api_key, http_options=self._http_options)
        # Add other providers here

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential_jitter(initial=1, max=20, jitter=1),
        retry=retry_if_exception(_is_transient_generation_error),
        reraise=True
    )
    def generate_text(self,
                     prompt: str,
                     temperature: Optional[float] = None,
                     top_p: Optional[float] = None,
                     response_schema: Type[BaseModel] = None) -> str:
        """Base method for generating text with retry logic built in"""
        if self.provider == "google_generative_ai":
            return self._generate_content(prompt, temperature, top_p, response_schema)
        return "Not implemented"

    def _generate_content(self, prompt: str, temperature: Optional[float], top_p: Optional[float], response_schema: Type[BaseModel]) -> str:
        """Generate using google.genai client"""
        try:
            client = self._get_client()
            if client is None:
                self._increment_diagnostic(
                    "provider_client_unavailable",
                    detail="Gemini client was not initialized when generate_content was called.",
                )
                raise RuntimeError("Gemini client is not initialized")

            # Prepare generation config
            actual_temp = temperature if temperature is not None else self.temperature
            actual_top_p = top_p if top_p is not None else self.top_p

            config_dict = {
                "temperature": actual_temp,
                "top_p": actual_top_p,
                "max_output_tokens": self.MAX_TOKENS,
            }

            # Add thinking config when requested.
            thinking_fields = getattr(types.ThinkingConfig, "model_fields", {}) or {}
            supports_thinking_level = "thinking_level" in thinking_fields
            supports_thinking_budget = "thinking_budget" in thinking_fields
            thinking_kwargs: Dict[str, Any] = {}

            normalized_level = self.thinking_level
            if normalized_level is not None and normalized_level not in THINKING_LEVEL_TO_BUDGET_FALLBACK:
                self._increment_diagnostic(
                    "invalid_thinking_level",
                    detail=f"thinking_level={self.thinking_level!r}",
                )
                normalized_level = None

            if normalized_level is not None:
                if supports_thinking_level:
                    thinking_kwargs["thinking_level"] = normalized_level
                elif supports_thinking_budget:
                    mapped_budget = THINKING_LEVEL_TO_BUDGET_FALLBACK[normalized_level]
                    thinking_kwargs["thinking_budget"] = mapped_budget
                    self._increment_diagnostic(
                        "thinking_level_budget_fallbacks",
                        detail=f"SDK lacks thinking_level; mapped {normalized_level} -> {mapped_budget}",
                    )
                else:
                    self._increment_diagnostic(
                        "thinking_config_unsupported",
                        detail="SDK exposes neither thinking_level nor thinking_budget.",
                    )

            if self.thinking_budget is not None and "thinking_budget" not in thinking_kwargs:
                if supports_thinking_budget:
                    try:
                        thinking_kwargs["thinking_budget"] = int(self.thinking_budget)
                    except (TypeError, ValueError):
                        self._increment_diagnostic(
                            "invalid_thinking_budget",
                            detail=f"thinking_budget={self.thinking_budget!r}",
                        )
                else:
                    self._increment_diagnostic(
                        "thinking_budget_unsupported",
                        detail="SDK does not expose thinking_budget.",
                    )

            if thinking_kwargs:
                config_dict["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)

            if response_schema:
                config_dict["response_schema"] = response_schema
                config_dict["response_mime_type"] = "application/json"

            config = types.GenerateContentConfig(**config_dict)

            print(f"{self.agent_name} using client with temperature: {actual_temp}, top_p: {actual_top_p}")

            # Generate content
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            # Track tokens
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                with self._token_lock:
                    self.total_token_count += response.usage_metadata.total_token_count
                    if hasattr(response.usage_metadata, 'prompt_token_count'):
                        self.input_token_count += response.usage_metadata.prompt_token_count
                    if hasattr(response.usage_metadata, 'candidates_token_count'):
                        self.output_token_count += response.usage_metadata.candidates_token_count

            print(
                "Total tokens",
                self.agent_name,
                ":",
                self.total_token_count,
                "(Input:",
                self.input_token_count,
                ", Output:",
                self.output_token_count,
                ")",
            )

            try:
                text = response.text
            except ValueError:
                self._increment_diagnostic(
                    "blocked_or_empty_responses",
                    detail="Provider returned response object without usable text payload.",
                )
                print("Warning: Client response blocked or empty.")
                return "No response."
            if text is None or str(text).strip() == "":
                self._increment_diagnostic(
                    "blocked_or_empty_responses",
                    detail="Provider returned an empty text payload.",
                )
                print("Warning: Client response text was empty.")
                return "No response."
            return str(text)

        except Exception as e:
            self._increment_diagnostic("generation_errors", detail=str(e))
            print(f"ERROR: Client generation failed: {e}")
            raise e

class Ideator(LLMWrapper):
    """Generates and manages ideas"""
    agent_name = "Ideator"
    MAX_PARENT_CONTEXT_CONCEPTS = 40
    AUTO_MAX_SEED_CONTEXT_CALLS = 3
    MAX_SEED_CONTEXT_CALLS = 12

    def __init__(self, seed_context_pool_size: Optional[int] = None, **kwargs):
        super().__init__(agent_name=self.agent_name, **kwargs)
        self.seed_context_pool_size: Optional[int] = None
        if seed_context_pool_size is not None:
            try:
                parsed = int(seed_context_pool_size)
                if parsed > 0:
                    self.seed_context_pool_size = parsed
            except (TypeError, ValueError):
                self._increment_diagnostic(
                    "invalid_seed_context_pool_size",
                    detail=f"seed_context_pool_size={seed_context_pool_size!r}",
                )

    def _extract_context_text(self, text: str) -> str:
        """Extract raw concept payload, preferring text after CONCEPTS: marker."""
        if "CONCEPTS:" in text:
            return text.split("CONCEPTS:", 1)[1].strip()
        self._increment_diagnostic(
            "context_missing_concepts_marker",
            detail=(text or "")[:400],
        )
        return text.strip()

    def _parse_context_items(self, context_text: str) -> List[str]:
        """Parse concept payload supporting bullet lists, || separators, and comma fallback."""
        cleaned_text = (context_text or "").strip()
        if not cleaned_text:
            return []

        lines = [line.strip() for line in cleaned_text.splitlines() if line.strip()]
        if len(lines) >= 2:
            parsed_lines: List[str] = []
            for line in lines:
                item = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().rstrip(",")
                if item:
                    parsed_lines.append(item)
            parsed_lines = list(dict.fromkeys(parsed_lines))
            if len(parsed_lines) >= 3:
                return parsed_lines

        if "||" in cleaned_text:
            pipe_items = [item.strip() for item in cleaned_text.split("||") if item.strip()]
            pipe_items = list(dict.fromkeys(pipe_items))
            if pipe_items:
                return pipe_items

        comma_items = [item.strip() for item in cleaned_text.split(",") if item.strip()]
        return list(dict.fromkeys(comma_items))

    def _build_seed_context_bank(self, n: int, idea_type: str) -> List[str]:
        """Build a reusable concept bank with 1-3 context calls for efficiency."""
        env_override = os.getenv("IDEA_CONTEXT_POOL_CALLS")
        if self.seed_context_pool_size is not None:
            context_calls = self.seed_context_pool_size
        elif env_override is not None:
            try:
                context_calls = int(env_override)
            except ValueError:
                self._increment_diagnostic(
                    "invalid_context_pool_call_override",
                    detail=f"IDEA_CONTEXT_POOL_CALLS={env_override!r}",
                )
                context_calls = 1
        else:
            # Scale up context calls for larger populations while staying within 1-3 calls.
            # 1 call for <=10 ideas, 2 for 11-20, 3 for 21+.
            context_calls = ((max(n, 1) - 1) // 10) + 1
            context_calls = min(context_calls, self.AUTO_MAX_SEED_CONTEXT_CALLS)

        context_calls = max(1, min(context_calls, self.MAX_SEED_CONTEXT_CALLS))
        self._increment_diagnostic("seed_context_calls", context_calls)

        concept_bank: List[str] = []
        for _ in range(context_calls):
            sampled_context = self.generate_context(idea_type)
            concept_bank.extend(self._parse_context_items(sampled_context))

        concept_bank = list(dict.fromkeys(concept_bank))
        if not concept_bank:
            self._increment_diagnostic(
                "seed_context_bank_empty",
                detail=f"n={n}, context_calls={context_calls}",
            )
        return concept_bank

    def _sample_context_pool(self, concept_bank: List[str]) -> str:
        """Sample a context pool string from a concept bank."""
        if not concept_bank:
            return "general concepts"

        target_size = max(3, min(int(len(concept_bank) * 0.4), 15))
        sample_size = min(len(concept_bank), target_size)
        sampled = self._random_sample(concept_bank, sample_size)
        return ", ".join(sampled)

    def generate_context(self, idea_type: str) -> str:
        """Generate initial context for ideation"""
        prompts = self._resolve_prompts(idea_type)

        # Use the configured temperature unless explicitly overridden
        text = self.generate_text(prompts.CONTEXT_PROMPT)
        print(f"Text: {text}")

        context_text = self._extract_context_text(text)
        concepts = self._parse_context_items(context_text)

        if not concepts:
            self._increment_diagnostic(
                "context_empty_after_parse",
                detail=(text or "")[:400],
            )
            print(f"Warning: No valid concepts found in text: '{text}'")
            return text.strip() or "general concepts"

        # Use between 30-50% of the words to maintain diversity while avoiding overwhelming context
        target_size = max(3, min(int(len(concepts) * 0.4), 15))  # 40% but cap at 15 words
        sample_size = min(len(concepts), target_size)

        subset = self._random_sample(concepts, sample_size)
        return ", ".join(subset)

    def generate_specific_prompt(self, context_pool: str, idea_type: str) -> str:
        """Generate a specific idea prompt from the context pool using the translation layer"""
        prompts = self._resolve_prompts(idea_type)

        # Create the translation prompt with shuffled subset
        translation_prompt = prompts.SPECIFIC_PROMPT.format(context_pool=context_pool)

        specific_prompt = self.generate_text(translation_prompt)
        print(f"Generated specific prompt: {specific_prompt}")

        return specific_prompt

    def generate_context_from_parents(self, parent_genotypes: List[str], mutation_rate: float = 0.0, mutation_context_pool: str = "") -> str:
        """Generate context pool from parent genotypes by combining and sampling concepts, with optional mutation.

        Args:
            parent_genotypes: List of genotypes from parent ideas
            mutation_rate: Probability of selecting a concept from the mutation pool (0.0 to 1.0)
            mutation_context_pool: A string of comma-separated concepts to use for mutation

        Returns:
            A sampled context pool string
        """
        # Parse concepts for each parent genotype independently.
        parent_concept_lists: List[List[str]] = []
        for genotype in parent_genotypes:
            concepts = [concept.strip() for concept in genotype.split(';') if concept.strip()]
            concepts = list(dict.fromkeys(concepts))
            if concepts:
                parent_concept_lists.append(concepts)

        # Process mutation concepts if available
        mutation_concepts = []
        if mutation_rate > 0 and mutation_context_pool:
            mutation_concepts = self._parse_context_items(mutation_context_pool)
            # Remove duplicates
            mutation_concepts = list(dict.fromkeys(mutation_concepts))
            print(f"Mutation enabled: rate={mutation_rate}, pool size={len(mutation_concepts)}")

        if not parent_concept_lists:
            self._increment_diagnostic(
                "empty_parent_genotypes",
                detail=f"parent_genotypes={parent_genotypes!r}",
            )
            if mutation_concepts:
                fallback_count = min(len(mutation_concepts), max(1, min(5, len(mutation_concepts))))
                return ", ".join(self._random_sample(mutation_concepts, fallback_count))
            return "general concepts"

        # Target child context size is the average parent genotype length.
        parent_lengths = [len(concepts) for concepts in parent_concept_lists]
        target_child_size = int(round(sum(parent_lengths) / len(parent_lengths)))
        target_child_size = max(3, min(target_child_size, self.MAX_PARENT_CONTEXT_CONCEPTS))

        primary_parent_idx = 0
        primary_parent = parent_concept_lists[primary_parent_idx]
        primary_ratio = self._random_uniform(0.4, 0.6)
        desired_primary_count = max(1, int(round(target_child_size * primary_ratio)))
        desired_primary_count = min(desired_primary_count, len(primary_parent))

        sampled_concepts: List[str] = []
        sampled_concepts_lower = set()

        primary_selection = self._random_sample(primary_parent, desired_primary_count)
        for concept in primary_selection:
            concept_key = concept.lower()
            if concept_key not in sampled_concepts_lower:
                sampled_concepts.append(concept)
                sampled_concepts_lower.add(concept_key)

        desired_secondary_count = max(0, target_child_size - len(sampled_concepts))
        secondary_pool: List[str] = []
        for idx, concepts in enumerate(parent_concept_lists):
            if idx == primary_parent_idx:
                continue
            secondary_pool.extend(concepts)
        secondary_pool = list(dict.fromkeys(secondary_pool))

        secondary_added = 0
        if secondary_pool and desired_secondary_count > 0:
            secondary_selection = self._random_sample(
                secondary_pool,
                min(desired_secondary_count, len(secondary_pool)),
            )
            for concept in secondary_selection:
                concept_key = concept.lower()
                if concept_key in sampled_concepts_lower:
                    continue
                sampled_concepts.append(concept)
                sampled_concepts_lower.add(concept_key)
                secondary_added += 1

        # Do not backfill overlaps from the secondary parent pool.
        if secondary_added < desired_secondary_count:
            self._increment_diagnostic(
                "parent_overlap_reduced_child_context",
                detail=(
                    f"desired_secondary={desired_secondary_count}, "
                    f"actual_secondary={secondary_added}, "
                    f"target_child_size={target_child_size}"
                ),
            )

        # Mutation pool is restricted to concepts not already present in selected parent concepts.
        mutation_candidates = [
            c for c in mutation_concepts if c.lower() not in sampled_concepts_lower
        ]
        if mutation_concepts and not mutation_candidates:
            self._increment_diagnostic(
                "mutation_pool_fully_overlapping",
                detail=f"mutation_pool_size={len(mutation_concepts)}",
            )

        mutation_count = 0
        if mutation_candidates and mutation_rate > 0:
            desired_mutation_count = max(1, int(round(target_child_size * mutation_rate)))
            available_slots = max(0, self.MAX_PARENT_CONTEXT_CONCEPTS - len(sampled_concepts))
            actual_mutation_count = min(
                len(mutation_candidates),
                desired_mutation_count,
                available_slots,
            )
            if actual_mutation_count > 0:
                mutated_selection = self._random_sample(mutation_candidates, actual_mutation_count)
                sampled_concepts.extend(mutated_selection)
                mutation_count = actual_mutation_count
                print(f"Injected {len(mutated_selection)} mutation concepts: {mutated_selection}")
            else:
                self._increment_diagnostic(
                    "mutation_injection_capped",
                    detail=(
                        f"desired_mutation={desired_mutation_count}, "
                        f"available_slots={available_slots}, "
                        f"mutation_candidates={len(mutation_candidates)}"
                    ),
                )

        # Safety guard for extremely verbose genotype outputs.
        if len(sampled_concepts) > self.MAX_PARENT_CONTEXT_CONCEPTS:
            self._increment_diagnostic(
                "parent_context_truncated",
                detail=f"len={len(sampled_concepts)}, cap={self.MAX_PARENT_CONTEXT_CONCEPTS}",
            )
            sampled_concepts = sampled_concepts[: self.MAX_PARENT_CONTEXT_CONCEPTS]

        if not sampled_concepts:
            self._increment_diagnostic(
                "context_from_parents_empty",
                detail="No concepts remained after parent/mutation filtering.",
            )
            if mutation_concepts:
                fallback_count = min(len(mutation_concepts), max(1, min(5, len(mutation_concepts))))
                sampled_concepts = self._random_sample(mutation_concepts, fallback_count)
            else:
                return "general concepts"

        # Shuffle the final mix.
        self._random_shuffle(sampled_concepts)

        context_pool = ", ".join(sampled_concepts)
        print(
            "Generated context from parents "
            f"(target={target_child_size}, selected={len(sampled_concepts)}, mutation={mutation_count}): {context_pool}"
        )
        return context_pool

    def generate_idea_from_context(self, context_pool: str, idea_type: str) -> tuple[str, str]:
        """Helper function to generate an idea from a context pool

        This function encapsulates steps 3 and 4 from the user's request:
        3. Using the sample to ask LLM to create specific prompt
        4. Generate an idea from specific prompt

        Args:
            context_pool: The context pool to generate from
            idea_type: Type of idea to generate

        Returns:
            tuple: (generated_idea, specific_prompt_used)
        """
        # Generate specific prompt from context pool
        specific_prompt = self.generate_specific_prompt(context_pool, idea_type)

        # Keep generation compact: specific direction + consolidated template constraints.
        prompts = self._resolve_prompts(idea_type)
        extended_prompt = specific_prompt
        if hasattr(prompts, 'template') and prompts.template.special_requirements:
            extended_prompt = f"{specific_prompt}\n\nConstraints:\n{prompts.template.special_requirements}"

        response = self.generate_text(extended_prompt)

        return response, specific_prompt

    def get_idea_prompt(self, idea_type: str) -> str:
        """Get prompt template for specific idea type"""
        prompts = self._resolve_prompts(idea_type)
        return prompts.IDEA_PROMPT

    def seed_ideas(self, n: int, idea_type: str) -> tuple[List[str], List[str]]:
        """Generate n initial ideas

        Returns:
            tuple: (ideas, specific_prompts)
        """
        ideas = []
        specific_prompts = []
        context_bank = self._build_seed_context_bank(n, idea_type)

        for _ in tqdm(range(n), desc="Generating ideas"):
            # Sample a context pool from a shared context bank.
            context_pool = self._sample_context_pool(context_bank)
            if context_pool == "general concepts":
                context_pool = self.generate_context(idea_type)

            response, specific_prompt = self.generate_idea_from_context(context_pool, idea_type)
            specific_prompts.append(specific_prompt)

            ideas.append({"id": uuid.uuid4(), "idea": response, "parent_ids": []})

        return ideas, specific_prompts


class Formatter(LLMWrapper):
    """Reformats unstructured ideas into a cleaner format"""
    agent_name = "Formatter"

    def __init__(self, **kwargs):
        # Use a default temperature only if not provided in kwargs
        temp = kwargs.pop('temperature', 1.0)
        top_p = kwargs.pop('top_p', 0.95)
        super().__init__(agent_name=self.agent_name, temperature=temp, top_p=top_p, **kwargs)

    def format_idea(self, raw_idea: str, idea_type: str) -> str:
        """Format a raw idea into a structured format"""
        prompts = self._resolve_prompts(idea_type)

        # Check if this is an Oracle-generated idea
        is_oracle_idea = isinstance(raw_idea, dict) and raw_idea.get("oracle_generated", False)
        oracle_analysis = raw_idea.get("oracle_analysis", "") if is_oracle_idea else ""

        # If raw_idea is a dictionary with 'id' and 'idea' keys, extract just the idea text
        idea_text = raw_idea["idea"] if isinstance(raw_idea, dict) and "idea" in raw_idea else raw_idea
        print(f"Formatting idea:\n {idea_text}")

        if is_oracle_idea:
            # For Oracle ideas, use a different prompt that preserves the analysis context
            prompt = f"""Format the following idea into a structured format with title and content.

                    ORACLE CONTEXT: This idea was generated by the Oracle diversity agent to address patterns and gaps in the population.

                    IDEA TO FORMAT:
                    {idea_text}

                    Please format this into a structured format with a compelling title and well-organized content. The Oracle analysis will be preserved separately.
                    """
        else:
            prompt = prompts.FORMAT_PROMPT.format(input_text=idea_text)

        print(f"Prompt:\n {prompt}")

        # Force JSON response schema for structured output
        # Use response_schema for structured output with the new client
        response = self.generate_text(
            prompt,
            response_schema=Idea,
            # Ensure we're using a temperature that favors structured output
            temperature=0.7
        )

        fallback_idea_text = ""
        if isinstance(idea_text, str):
            fallback_idea_text = idea_text.strip()
        elif hasattr(idea_text, "content"):
            fallback_idea_text = str(getattr(idea_text, "content", "") or "").strip()
        elif isinstance(idea_text, dict):
            fallback_idea_text = str(
                idea_text.get("content")
                or idea_text.get("idea")
                or ""
            ).strip()
        elif idea_text is not None:
            fallback_idea_text = str(idea_text).strip()

        # Clean up markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:]

        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]

        clean_response = clean_response.strip()

        used_fallback = False
        try:
            formatted_idea = Idea(**json.loads(clean_response))

            # Double check that content is not empty if it was required
            if not formatted_idea.content or formatted_idea.content.strip() == "":
                 raise ValueError("Content field is empty in formatted idea")

        except (json.JSONDecodeError, ValueError, Exception) as e:
            used_fallback = True
            self._increment_diagnostic(
                "formatter_schema_parse_failures",
                detail=f"{e}: {(response or '')[:400]}",
            )
            print(f"FORMATTER: JSON parsing failed: {e}")
            print(f"FORMATTER: Raw response: {response}")

            # Fallback: Try to extract title and content manually
            title = "Untitled"
            content = response.strip()

            # If raw response is just JSON-like but failed parsing, try to clean it
            if response.strip().startswith('{') and response.strip().endswith('}'):
                try:
                     # Try to be lenient with newlines in strings
                     import re
                     # Basic fix for unescaped newlines in JSON values
                     cleaned = re.sub(r'(?<=: ")(.*?)(?=")', lambda m: m.group(1).replace('\n', '\\n'), response, flags=re.DOTALL)
                     data = json.loads(cleaned)
                     if 'title' in data:
                         title = data['title']
                     if 'content' in data:
                         content = data['content']
                     # If successful, create idea and return
                     if content and content.strip():
                         formatted_idea = Idea(title=title, content=content)
                         self._increment_diagnostic("formatter_json_clean_recovery")
                         # Skip to return block...
                         # But since we can't easily jump, we'll just fall through to manual extraction if this fails
                except Exception as e:
                    print(f"FORMATTER: JSON fallback cleaning failed: {e}")
                    self._increment_diagnostic(
                        "formatter_json_clean_recovery_failures",
                        detail=str(e),
                    )
                    pass

            # Try to parse "Title: X\nContent: Y" format
            if "Title:" in response and "Content:" in response:
                lines = response.split('\n')
                title_line = None
                content_lines = []
                found_content = False

                for line in lines:
                    if line.strip().startswith("Title:"):
                        title_line = line.replace("Title:", "").strip()
                    elif line.strip().startswith("Content:"):
                        content_lines.append(line.replace("Content:", "").strip())
                        found_content = True
                    elif found_content:
                        content_lines.append(line)

                if title_line:
                    title = title_line
                if content_lines:
                    content = "\n".join(content_lines).strip()
                self._increment_diagnostic("formatter_title_content_recovery")

            normalized_content = content.strip() if isinstance(content, str) else ""
            if normalized_content.lower() in {"", "no response", "no response."}:
                normalized_fallback = fallback_idea_text.strip()
                if normalized_fallback.lower() not in {"", "no response", "no response."}:
                    content = normalized_fallback
                    self._increment_diagnostic("formatter_empty_content_recovery")
                else:
                    content = "Model response was empty."
                    self._increment_diagnostic("formatter_empty_content_recovery")

            print(f"FORMATTER: Fallback extracted - Title: '{title}', Content length: {len(content)}")
            formatted_idea = Idea(title=title, content=content)

        if used_fallback:
            self._increment_diagnostic("formatter_fallback_used")

        # If the input was a dictionary with an ID, preserve that ID and all metadata
        if isinstance(raw_idea, dict) and "id" in raw_idea:
            result = {"id": raw_idea["id"], "idea": formatted_idea}
            # Preserve parent_ids if they exist
            if "parent_ids" in raw_idea:
                result["parent_ids"] = raw_idea["parent_ids"]
            else:
                result["parent_ids"] = []

            # Preserve Oracle metadata
            if is_oracle_idea:
                result["oracle_generated"] = True
                result["oracle_analysis"] = oracle_analysis

            # Preserve Elite metadata
            if raw_idea.get("elite_selected", False):
                result["elite_selected"] = raw_idea["elite_selected"]
                result["elite_source_id"] = raw_idea.get("elite_source_id")
                result["elite_source_generation"] = raw_idea.get("elite_source_generation")

            return result
        return formatted_idea

class Critic(LLMWrapper):
    """Analyzes and refines ideas"""
    agent_name = "Critic"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, **kwargs)
        # Small result cache to avoid duplicate pair evaluations
        self._compare_cache: OrderedDict[str, str] = OrderedDict()
        self._cache_max_size = 1024
        # Thread pool for parallel comparisons (configurable via env)
        self._max_workers = int(os.environ.get("COMPARISON_CONCURRENCY", "8"))
        # Lock for thread-safe cache access
        self._cache_lock = threading.Lock()

    def critique(self, idea: str, idea_type: str) -> str:
        """Provide critique for an idea"""
        prompts = self._resolve_prompts(idea_type)
        # Extract idea text if it's a dictionary
        idea_text = idea["idea"] if isinstance(idea, dict) and "idea" in idea else idea
        prompt = prompts.CRITIQUE_PROMPT.format(idea=idea_text)
        return self.generate_text(prompt)

    def refine(self, idea: str, idea_type: str) -> str:
        """Refine an idea based on critique"""
        # Extract idea text if it's a dictionary
        idea_text = idea["idea"] if isinstance(idea, dict) and "idea" in idea else idea
        critique = self.critique(idea_text, idea_type)
        prompts = self._resolve_prompts(idea_type)
        prompt = prompts.REFINE_PROMPT.format(
            idea=idea_text,
            critique=critique
        )
        refined_idea = self.generate_text(prompt)

        # If the input was a dictionary with an ID, preserve that ID and all metadata
        if isinstance(idea, dict) and "id" in idea:
            result = {"id": idea["id"], "idea": refined_idea}
            # Preserve parent_ids if they exist
            if "parent_ids" in idea:
                result["parent_ids"] = idea["parent_ids"]
            else:
                result["parent_ids"] = []

            # Preserve Oracle metadata
            if idea.get("oracle_generated", False):
                result["oracle_generated"] = idea["oracle_generated"]
                result["oracle_analysis"] = idea.get("oracle_analysis", "")

            # Preserve Elite metadata
            if idea.get("elite_selected", False):
                result["elite_selected"] = idea["elite_selected"]
                result["elite_source_id"] = idea.get("elite_source_id")
                result["elite_source_generation"] = idea.get("elite_source_generation")

            return result
        return refined_idea

    def _elo_update(self, elo_a, elo_b, winner):
        """Update the Elo rating of an idea"""
        return elo_update(elo_a, elo_b, winner)

    def _select_swiss_bye(self, ordered_indices: List[int], bye_counts: Dict[int, int]) -> Optional[int]:
        """
        Select a bye candidate for an odd-sized Swiss round.

        Preference order:
        1) Fewest byes so far
        2) Lowest-ranked (last in ordered list)
        """
        return select_swiss_bye(ordered_indices, bye_counts)

    def _pair_players_swiss(
        self,
        ordered_indices: List[int],
        match_history: Set[Tuple[int, int]],
        backtrack_limit: int = 20000,
    ) -> Optional[List[Tuple[int, int]]]:
        """
        Pair players in Swiss style while minimizing repeat matchups.

        Returns a list of pairs (idx_a, idx_b), or None if no pairing found
        within the backtracking limit.
        """
        return pair_players_swiss(
            ordered_indices,
            match_history,
            backtrack_limit=backtrack_limit,
        )

    def _generate_swiss_round_pairs(
        self,
        ranks: Dict[int, float],
        match_history: Set[Tuple[int, int]],
        bye_counts: Dict[int, int],
    ) -> Tuple[List[Tuple[int, int]], Optional[int]]:
        """
        Generate Swiss pairings for a single round with minimal repeat matchups.
        """
        return generate_swiss_round_pairs(ranks, match_history, bye_counts)

    def _extract_idea_meta(self, idea) -> Dict[str, Optional[str]]:
        """Extract lightweight metadata for tournament display."""
        idea_id = None
        title = "Untitled"

        if isinstance(idea, dict):
            raw_id = idea.get("id")
            if raw_id:
                idea_id = str(raw_id)
            idea_obj = idea.get("idea", idea)
            if hasattr(idea_obj, "title"):
                title = idea_obj.title or title
            elif isinstance(idea_obj, dict):
                title = idea_obj.get("title", title)
            else:
                title = str(idea_obj)
        else:
            title = str(idea)

        if len(title) > 120:
            title = title[:117] + "..."

        return {"id": idea_id, "title": title}

    def get_tournament_ranks(
        self,
        ideas: List[str],
        idea_type: str,
        rounds: int = 1,
        progress_callback: Callable[[int, int], None] = None,
        details: Optional[List[Dict[str, Any]]] = None,
        full_tournament_rounds: Optional[int] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> dict:
        """Get tournament ranks using a global Swiss-system tournament.

        Falls back to sequential evaluation for tiny populations to keep behavior deterministic.
        """

        # Handle edge case: if there's only one idea or no ideas, return appropriate ranking
        if len(ideas) <= 1:
            return {0: 1500} if len(ideas) == 1 else {}

        if rounds <= 0:
            rounds = 1

        ranks = {i: 1500 for i in range(len(ideas))}

        # Decide whether to enable parallel evaluation
        enable_parallel = (
            os.environ.get("ENABLE_PARALLEL_TOURNAMENT", "1") != "0"
            and len(ideas) >= 4
            and rounds >= 2
        )

        match_history: Set[Tuple[int, int]] = set()
        bye_counts: Dict[int, int] = {}
        segment_rounds = None
        if full_tournament_rounds and full_tournament_rounds > 0:
            segment_rounds = int(full_tournament_rounds)

        # Precompute total comparisons for progress reporting
        total_pairs = 0
        expected_pairs_per_round = len(ideas) // 2
        total_pairs = expected_pairs_per_round * rounds
        completed_pairs = 0

        def report_progress():
            if progress_callback:
                try:
                    progress_callback(completed_pairs, total_pairs)
                except Exception as e:
                    print(f"Error in progress callback: {e}")

        for _round in range(rounds):
            if callable(should_stop) and should_stop():
                print("Tournament stop requested. Ending tournament rounds early.")
                break

            # For multi-tournament runs we reset Swiss bracket state at each segment,
            # while keeping accumulated Elo scores.
            if segment_rounds and _round > 0 and (_round % segment_rounds) == 0:
                match_history = set()
                bye_counts = {}

            pairs, bye_idx = self._generate_swiss_round_pairs(ranks, match_history, bye_counts)

            round_pairs = []
            for idx_a, idx_b in pairs:
                a_meta = self._extract_idea_meta(ideas[idx_a])
                b_meta = self._extract_idea_meta(ideas[idx_b])
                round_pairs.append({
                    "a_idx": idx_a,
                    "b_idx": idx_b,
                    "a_id": a_meta["id"],
                    "b_id": b_meta["id"],
                    "a_title": a_meta["title"],
                    "b_title": b_meta["title"],
                })

            round_record = {
                "round": _round + 1,
                "pairs": round_pairs,
            }
            if segment_rounds:
                round_record["tournament_index"] = (_round // segment_rounds) + 1
                round_record["round_in_tournament"] = (_round % segment_rounds) + 1

            if bye_idx is not None:
                bye_meta = self._extract_idea_meta(ideas[bye_idx])
                round_record["bye"] = {
                    "idx": bye_idx,
                    "id": bye_meta["id"],
                    "title": bye_meta["title"],
                }

            if details is not None:
                details.append(round_record)

            if not pairs:
                continue

            if enable_parallel:
                def on_pair_complete(_c, _t):
                    nonlocal completed_pairs
                    completed_pairs += 1
                    report_progress()

                results = parallel_evaluate_pairs(
                    pairs=pairs,
                    items=ideas,
                    compare_fn=self.compare_ideas,
                    idea_type=idea_type,
                    concurrency=self._max_workers,
                    randomize_presentation=True,
                    progress_callback=on_pair_complete,
                    should_stop=should_stop,
                )

                # Update ranks in the deterministic order of generated pairs
                results_map = {(a, b): w for a, b, w in results}
                for pair_index, (idx_a, idx_b) in enumerate(pairs):
                    winner = results_map.get((idx_a, idx_b))
                    if details is not None and pair_index < len(round_pairs):
                        round_pairs[pair_index]["winner"] = winner
                    elo_a, elo_b = self._elo_update(ranks[idx_a], ranks[idx_b], winner)
                    ranks[idx_a] = elo_a
                    ranks[idx_b] = elo_b
            else:
                for pair_index, (idx_a, idx_b) in enumerate(pairs):
                    idea_a = ideas[idx_a]
                    idea_b = ideas[idx_b]
                    idea_a_obj = idea_a["idea"] if isinstance(idea_a, dict) and "idea" in idea_a else idea_a
                    idea_b_obj = idea_b["idea"] if isinstance(idea_b, dict) and "idea" in idea_b else idea_b
                    idea_a_dict = idea_a_obj.dict() if hasattr(idea_a_obj, 'dict') else idea_a_obj
                    idea_b_dict = idea_b_obj.dict() if hasattr(idea_b_obj, 'dict') else idea_b_obj

                    # Deterministic orientation in sequential mode to keep tests stable
                    winner = self.compare_ideas(idea_a_dict, idea_b_dict, idea_type)
                    if details is not None and pair_index < len(round_pairs):
                        round_pairs[pair_index]["winner"] = winner
                    elo_a, elo_b = self._elo_update(ranks[idx_a], ranks[idx_b], winner)
                    ranks[idx_a] = elo_a
                    ranks[idx_b] = elo_b
                    completed_pairs += 1
                    report_progress()

        return ranks


    def compare_ideas(self, idea_a, idea_b, idea_type: str):
        """
        Compare two ideas using the LLM and determine which is better.
        Returns: "A", "B", "tie", or None if there was an error
        """
        prompts = self._resolve_prompts(idea_type)

        # Get the item type from the prompt configuration
        item_type = getattr(prompts, "ITEM_TYPE", "ideas")
        criteria = prompts.COMPARISON_CRITERIA

        # The comparison prompt is now provided directly by the template
        # loader (YAMLTemplateWrapper ensures a default is available).
        prompt_template = prompts.COMPARISON_PROMPT

        prompt = prompt_template.format(
            item_type=item_type,
            criteria=", ".join(criteria),
            idea_a_title=idea_a.get('title', 'Untitled'),
            idea_a_content=idea_a.get('content', ''),
            idea_b_title=idea_b.get('title', 'Untitled'),
            idea_b_content=idea_b.get('content', '')
        )

        # Cache by model, temperature, and prompt hash to avoid duplicate calls
        cache_key = f"{self.model_name}|{self.temperature}|{sha256(prompt.encode('utf-8')).hexdigest()}"
        with self._cache_lock:
            if cache_key in self._compare_cache:
                result_cached = self._compare_cache.pop(cache_key)
                self._compare_cache[cache_key] = result_cached  # Move to end (LRU)
                return result_cached

        try:
            # Use the configured temperature for comparisons
            response = self.generate_text(prompt)
            result = response.strip().upper()

            print(f"LLM comparison response: {result}")

            # More robust parsing
            if "RESULT: A" in result or "Result: A" in result:
                return "A"
            elif "RESULT: B" in result or "Result: B" in result:
                return "B"
            elif "RESULT: TIE" in result or "Result: tie" in result:
                return "tie"
            elif "A" in result and "B" not in result:
                return "A"
            elif "B" in result and "A" not in result:
                return "B"
            else:
                winner = "tie"

            # Update cache (LRU eviction)
            with self._cache_lock:
                self._compare_cache[cache_key] = winner
                if len(self._compare_cache) > self._cache_max_size:
                    # pop oldest
                    self._compare_cache.popitem(last=False)
            return winner
        except Exception as e:
            self._increment_diagnostic("compare_ideas_errors", detail=str(e))
            print(f"Error in compare_ideas: {e}")
            return None  # Return None instead of "tie" on error


class Breeder(LLMWrapper):
    """Breeds ideas and handles genotype encoding/decoding"""
    agent_name = "Breeder"
    parent_count = 2

    def __init__(
        self,
        mutation_rate: float = 0.0,
        seed_context_pool_size: Optional[int] = None,
        **kwargs,
    ):
        # Don't set temperature directly here, let it come from kwargs
        super().__init__(agent_name=self.agent_name, **kwargs)
        self.mutation_rate = mutation_rate
        self.seed_context_pool_size = seed_context_pool_size
        self._thread_local = threading.local()
        print(f"Breeder initialized with mutation_rate={mutation_rate}")

    def _get_thread_ideator(self) -> Ideator:
        ideator = getattr(self._thread_local, "ideator", None)
        if ideator is None:
            ideator = Ideator(
                provider=self.provider,
                model_name=self.model_name,
                temperature=self.temperature,
                top_p=self.top_p,
                api_key=self.api_key,
                random_seed=self._random_randbits(64),
                seed_context_pool_size=self.seed_context_pool_size,
            )
            for template_id, template_data in self._iter_custom_templates():
                ideator.register_custom_template(template_id, template_data)
            self._thread_local.ideator = ideator
        return ideator

    def register_custom_template(self, template_id: str, template_data: Dict[str, Any]) -> None:
        super().register_custom_template(template_id, template_data)
        ideator = getattr(self._thread_local, "ideator", None)
        if ideator is not None:
            ideator.register_custom_template(template_id, template_data)

    def encode_to_genotype(self, idea: str, idea_type: str) -> str:
        """
        Convert a full idea (phenotype) to its basic elements (genotype)

        Args:
            idea: The full idea to encode (can be dict with 'idea' key or string)
            idea_type: Type of idea for customization

        Returns:
            A genotype string representing the basic elements
        """
        # Extract idea text if it's a dictionary
        idea_text = idea["idea"] if isinstance(idea, dict) and "idea" in idea else idea

        # Get the idea object if it's a formatted idea
        if hasattr(idea_text, 'title') and hasattr(idea_text, 'content'):
            idea_content = f"Title: {idea_text.title}\nContent: {idea_text.content}"
        else:
            idea_content = str(idea_text)

        # Create encoding prompt based on idea type
        prompts = self._resolve_prompts(idea_type)

        # Get genotype encoding prompt from template
        prompt_template = prompts.GENOTYPE_ENCODE_PROMPT

        prompt = prompt_template.format(idea_content=idea_content)

        response = self.generate_text(prompt)

        # Clean up the response
        genotype = response.strip()
        if genotype.startswith("Genotype:"):
            genotype = genotype[9:].strip()

        print(f"Encoded genotype: {genotype}")
        return genotype

    def breed(self, ideas: List[str], idea_type: str) -> str:
        """Breed ideas to create a new idea using the new approach that mirrors initial population generation

        This follows the same pattern as initial idea generation:
        1. Get concepts from the parents (their genotypes) and combine them
        2. Sample 50% at random (in python code)
        3. Using the sample to ask LLM to create specific prompt
        4. Generate an idea from specific prompt

        Args:
            ideas: List of parent ideas that have already been selected in the main evolution loop
            idea_type: Type of idea to breed

        Returns:
            A new idea with a unique ID and parent IDs
        """
        # Extract parent IDs
        parent_ids = []
        for parent in ideas:
            if isinstance(parent, dict) and "id" in parent:
                parent_ids.append(str(parent["id"]))

        # Step 1: Get concepts from the parents (their genotypes) and combine them
        parent_genotypes = []
        for parent in ideas:
            genotype = self.encode_to_genotype(parent, idea_type)
            parent_genotypes.append(genotype)

        # Reuse one helper ideator per worker thread to avoid repeated client setup.
        ideator = self._get_thread_ideator()

        # Step 2: Sample 50% at random (handled in generate_context_from_parents)
        # Step 3 & 4: Using the sample to create specific prompt and generate idea

        # Generate fresh context for mutation if enabled
        mutation_context_pool = ""
        if self.mutation_rate > 0:
            print("Generating fresh context for mutation...")
            mutation_context_pool = ideator.generate_context(idea_type)

        context_pool = ideator.generate_context_from_parents(
            parent_genotypes,
            mutation_rate=self.mutation_rate,
            mutation_context_pool=mutation_context_pool
        )
        new_idea, specific_prompt = ideator.generate_idea_from_context(context_pool, idea_type)

        print(f"Generated new idea from parent concepts: {new_idea[:100]}...")
        print(f"Using specific prompt: {specific_prompt[:100]}...")

        # Create a new idea with a unique ID and parent IDs
        # Also include the specific prompt used for breeding
        return {
            "id": uuid.uuid4(),
            "idea": new_idea,
            "parent_ids": parent_ids,
            "specific_prompt": specific_prompt  # Store the specific prompt for later display
        }


class Oracle(LLMWrapper):
    """Analyzes entire population history and introduces diversity by identifying overused patterns"""
    agent_name = "Oracle"

    def __init__(self, **kwargs):
        # Use a higher temperature to encourage creativity and diversity
        temp = kwargs.pop('temperature', 1.0)
        top_p = kwargs.pop('top_p', 0.95)
        super().__init__(agent_name=self.agent_name, temperature=temp, top_p=top_p, **kwargs)

    def analyze_and_diversify(self, history: List[List[str]], idea_type: str) -> dict:
        """
        Analyze the entire evolution history and generate a new diverse idea to replace an existing one.

        The replacement selection is handled by embedding-based centroid distance calculation.

        Args:
            history: Complete evolution history, list of generations (each generation is a list of ideas)
            idea_type: Type of ideas being evolved

        Returns:
            Dictionary with new_idea and action type
        """
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt(history, idea_type)

        print(f"Oracle analyzing {len(history)} generations with {sum(len(gen) for gen in history)} total ideas...")

        # This is the "expensive" single query that does everything
        response = self.generate_text(analysis_prompt)

        return self._parse_oracle_response(response)

    def _build_analysis_prompt(self, history: List[List[str]], idea_type: str) -> str:
        """Build the comprehensive analysis prompt for the Oracle"""

        # Format all historical ideas by generation
        history_text = ""
        for gen_idx, generation in enumerate(history):
            history_text += f"\n--- GENERATION {gen_idx} ---\n"
            for idea_idx, idea in enumerate(generation):
                idea_content = self._extract_idea_content(idea)
                history_text += f"Idea {gen_idx}.{idea_idx}: {idea_content}\n"

        # Get prompts for the idea type
        prompts = self._resolve_prompts(idea_type)

        # Get Oracle-specific prompts from the template
        mode_instruction = prompts.ORACLE_INSTRUCTION
        format_instructions = prompts.ORACLE_FORMAT_INSTRUCTIONS
        example_idea_prompts = getattr(prompts, 'EXAMPLE_IDEA_PROMPTS', '')

        # Build the Oracle's comprehensive prompt using the template
        oracle_constraints = getattr(prompts, 'ORACLE_CONSTRAINTS', '')
        prompt = prompts.ORACLE_MAIN_PROMPT.format(
            idea_type=idea_type,
            history_text=history_text,
            mode_instruction=mode_instruction,
            format_instructions=format_instructions,
            oracle_constraints=oracle_constraints,
            example_idea_prompts=example_idea_prompts
        )

        return prompt

    def _extract_idea_content(self, idea) -> str:
        """Extract readable content from an idea object for analysis"""
        # Handle different idea formats
        if isinstance(idea, dict) and "idea" in idea:
            idea_obj = idea["idea"]
            # If the idea object has title and content attributes
            if hasattr(idea_obj, 'title') and hasattr(idea_obj, 'content'):
                return f"Title: {idea_obj.title}\nContent: {idea_obj.content}"
            # If the idea object is a string
            elif isinstance(idea_obj, str):
                return idea_obj
            # If the idea object is already a dict
            elif isinstance(idea_obj, dict):
                title = idea_obj.get('title', 'Untitled')
                content = idea_obj.get('content', str(idea_obj))
                return f"Title: {title}\nContent: {content}"

        # Fallback: treat as string
        return str(idea)


    def _parse_oracle_response(self, response: str) -> dict:
        """Parse the Oracle's response and extract the analysis and new idea separately"""

        # Parse the structured response to separate analysis from new idea
        oracle_analysis = ""
        idea_prompt = ""

        try:
            # Split response into sections
            if "=== ORACLE ANALYSIS ===" in response and "=== IDEA PROMPT ===" in response:
                parts = response.split("=== ORACLE ANALYSIS ===")[1]
                analysis_part, idea_part = parts.split("=== IDEA PROMPT ===")
                oracle_analysis = analysis_part.strip()
                idea_prompt = idea_part.strip()
                print(
                    "ORACLE: Successfully parsed structured response - analysis:",
                    len(oracle_analysis),
                    "chars, idea prompt:",
                    len(idea_prompt),
                    "chars",
                )
            else:
                # Fallback: treat entire response as new idea content but preserve Oracle metadata
                self._increment_diagnostic(
                    "oracle_parse_fallbacks",
                    detail=(response or "")[:500],
                )
                idea_prompt = response.strip()
                oracle_analysis = (
                    "Oracle response was not properly formatted. Expected sections '=== ORACLE ANALYSIS ===' "
                    "and '=== IDEA PROMPT ===' but got unstructured response. This indicates the LLM did not follow the required format."
                )
                print("ORACLE: Fallback parsing - treating entire response as idea content. Response length:", len(response), "chars")
        except Exception as e:
            # Fallback parsing failed
            self._increment_diagnostic("oracle_parse_errors", detail=str(e))
            idea_prompt = response.strip()
            oracle_analysis = f"Oracle parsing error: {e}. Response was treated as idea content."
            print("ORACLE: Parsing exception -", e)

        # Oracle only supports replace mode
        # Replacement selection is handled externally via embedding-based centroid distance
        print("ORACLE: Generated replacement idea. Selection of which idea to replace will be handled externally via embeddings.")

        return {
            "action": "replace",
            "idea_prompt": idea_prompt,
            "oracle_analysis": oracle_analysis
        }
