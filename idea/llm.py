from abc import ABC
import random
import os
from google import genai
from google.genai import types
import json
from pydantic import BaseModel
from typing import Type, Optional, Dict, List, Callable, Tuple, Set, Any
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from idea.models import Idea
from idea.prompts.loader import get_prompts, get_prompts_from_dict
import uuid
from collections import OrderedDict
from hashlib import sha256
import threading
from idea.ratings import parallel_evaluate_pairs

class LLMWrapper(ABC):
    """Base class for LLM interactions"""
    MAX_TOKENS = 8192

    def __init__(self,
                 provider: str = "google_generative_ai",
                 model_name: str = "gemini-2.0-flash",
                 prompt_template: str = "",
                 temperature: float = 1.0,
                 top_p: float = 0.95,
                 agent_name: str = "",
                 thinking_budget: Optional[int] = None,
                 api_key: Optional[str] = None):
        self.provider = provider
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.top_p = top_p
        self.thinking_budget = thinking_budget
        self.api_key = api_key
        self.total_token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0
        self.agent_name = agent_name
        print(f"Initializing {agent_name or 'LLM'} with temperature: {temperature}, top_p: {top_p}, thinking_budget: {thinking_budget}")
        self._custom_templates: Dict[str, Dict[str, Any]] = {}
        self._custom_prompt_wrappers: Dict[str, Any] = {}
        self._custom_template_lock = threading.Lock()

        self.client = None
        self._client_lock = threading.Lock()
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

    def _setup_provider(self):
        if self.provider == "google_generative_ai":
            api_key = self.api_key or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                print("Warning: GEMINI_API_KEY not set and no api_key provided")

            # Initialize client once
            with self._client_lock:
                if self.client is None and api_key:
                    self.client = genai.Client(api_key=api_key)
        # Add other providers here

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=30, max=300),
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
            # Prepare generation config
            actual_temp = temperature if temperature is not None else self.temperature
            actual_top_p = top_p if top_p is not None else self.top_p

            config_dict = {
                "temperature": actual_temp,
                "top_p": actual_top_p,
                "max_output_tokens": self.MAX_TOKENS,
            }

            # Add thinking config if applicable
            if self.thinking_budget is not None:
                if self.thinking_budget == -1:
                    config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=-1) # Dynamic
                    # print(f"{self.agent_name} using dynamic thinking budget")
                elif self.thinking_budget == 0:
                    config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=0) # Disabled
                    # print(f"{self.agent_name} thinking disabled")
                else:
                    config_dict["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
                    # print(f"{self.agent_name} using thinking budget: {self.thinking_budget}")

            if response_schema:
                config_dict["response_schema"] = response_schema
                config_dict["response_mime_type"] = "application/json"

            config = types.GenerateContentConfig(**config_dict)

            print(f"{self.agent_name} using client with temperature: {actual_temp}, top_p: {actual_top_p}")

            # Generate content
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            # Track tokens
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
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
                return response.text
            except ValueError:
                print("Warning: Client response blocked or empty.")
                return "No response."

        except Exception as e:
            print(f"ERROR: Client generation failed: {e}")
            raise e

class Ideator(LLMWrapper):
    """Generates and manages ideas"""
    agent_name = "Ideator"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, **kwargs)

    def generate_context(self, idea_type: str) -> str:
        """Generate initial context for ideation"""
        prompts = self._resolve_prompts(idea_type)

        # Use the configured temperature unless explicitly overridden
        text = self.generate_text(prompts.CONTEXT_PROMPT)
        print(f"Text: {text}")

        # Extract concepts from the response
        if "CONCEPTS:" in text:
            context_text = text.split("CONCEPTS:")[1].strip()
        else:
            # Fallback if the expected format is not found
            context_text = text.strip()

        # Use more of the context - sample 30-50% instead of 10%
        words = [word.strip() for word in context_text.split(',') if word.strip()]

        if not words:
            print(f"Warning: No valid concepts found in text: '{text}'")
            return text.strip() or "general concepts"

        # Use between 30-50% of the words to maintain diversity while avoiding overwhelming context
        target_size = max(3, min(int(len(words) * 0.4), 15))  # 40% but cap at 15 words
        sample_size = min(len(words), target_size)

        subset = random.sample(words, sample_size)
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
        # Combine all concepts from parent genotypes
        all_concepts = []
        for genotype in parent_genotypes:
            # Split genotype by semicolons and clean up
            concepts = [concept.strip() for concept in genotype.split(';') if concept.strip()]
            all_concepts.extend(concepts)

        # Remove duplicates while preserving order
        unique_parent_concepts = list(dict.fromkeys(all_concepts))

        # Process mutation concepts if available
        mutation_concepts = []
        if mutation_rate > 0 and mutation_context_pool:
            mutation_concepts = [c.strip() for c in mutation_context_pool.split(',') if c.strip()]
            # Remove duplicates
            mutation_concepts = list(dict.fromkeys(mutation_concepts))
            print(f"Mutation enabled: rate={mutation_rate}, pool size={len(mutation_concepts)}")

        # Determine total sample size (aim for around 10-15 concepts total)
        total_sample_size = max(3, min(len(unique_parent_concepts) + len(mutation_concepts), 15))

        # Calculate how many to draw from each pool
        if mutation_concepts and mutation_rate > 0:
            mutation_count = max(1, int(total_sample_size * mutation_rate))
            parent_count = total_sample_size - mutation_count
        else:
            mutation_count = 0
            parent_count = total_sample_size

        # Sample from parents
        sampled_concepts = []
        if unique_parent_concepts:
            # Ensure we don't try to sample more than available
            actual_parent_count = min(len(unique_parent_concepts), parent_count)
            sampled_concepts.extend(random.sample(unique_parent_concepts, actual_parent_count))

        # Sample from mutation pool
        if mutation_concepts and mutation_count > 0:
            # Ensure we don't try to sample more than available
            actual_mutation_count = min(len(mutation_concepts), mutation_count)
            mutated_selection = random.sample(mutation_concepts, actual_mutation_count)
            sampled_concepts.extend(mutated_selection)
            print(f"Injected {len(mutated_selection)} mutation concepts: {mutated_selection}")

        # Shuffle the final mix
        random.shuffle(sampled_concepts)

        context_pool = ", ".join(sampled_concepts)
        print(f"Generated context from parents (with mutation): {context_pool}")
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

        # Get the special requirements and extend the specific prompt with them
        prompts = self._resolve_prompts(idea_type)
        extended_prompt = specific_prompt

        # If there are special requirements, append them to the specific prompt
        if hasattr(prompts, 'template') and prompts.template.special_requirements:
            extended_prompt = f"{specific_prompt}\n\nConstraints:\n{prompts.template.special_requirements}"

        # Generate idea using the extended prompt (specific prompt + requirements)
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

        for _ in tqdm(range(n), desc="Generating ideas"):
            # Generate context pool
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

        # Clean up markdown code blocks if present
        clean_response = response.strip()
        if clean_response.startswith("```json"):
            clean_response = clean_response[7:]
        elif clean_response.startswith("```"):
            clean_response = clean_response[3:]

        if clean_response.endswith("```"):
            clean_response = clean_response[:-3]

        clean_response = clean_response.strip()

        try:
            formatted_idea = Idea(**json.loads(clean_response))

            # Double check that content is not empty if it was required
            if not formatted_idea.content or formatted_idea.content.strip() == "":
                 raise ValueError("Content field is empty in formatted idea")

        except (json.JSONDecodeError, ValueError, Exception) as e:
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
                         # Skip to return block...
                         # But since we can't easily jump, we'll just fall through to manual extraction if this fails
                except Exception as e:
                    print(f"FORMATTER: JSON fallback cleaning failed: {e}")
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

            print(f"FORMATTER: Fallback extracted - Title: '{title}', Content length: {len(content)}")
            formatted_idea = Idea(title=title, content=content)

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
        k = 32
        expected_a = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
        expected_b = 1 / (1 + 10 ** ((elo_a - elo_b) / 400))

        if winner == "A":
            elo_a = elo_a + k * (1 - expected_a)
            elo_b = elo_b + k * (0 - expected_b)
        elif winner == "B":
            elo_a = elo_a + k * (0 - expected_a)
            elo_b = elo_b + k * (1 - expected_b)
        else:
            elo_a = elo_a + k * (0.5 - expected_a)
            elo_b = elo_b + k * (0.5 - expected_b)

        return elo_a, elo_b

    def _select_swiss_bye(self, ordered_indices: List[int], bye_counts: Dict[int, int]) -> Optional[int]:
        """
        Select a bye candidate for an odd-sized Swiss round.

        Preference order:
        1) Fewest byes so far
        2) Lowest-ranked (last in ordered list)
        """
        if not ordered_indices:
            return None

        min_byes = min(bye_counts.get(idx, 0) for idx in ordered_indices)
        # Iterate from lowest-ranked to highest-ranked for deterministic selection
        for idx in reversed(ordered_indices):
            if bye_counts.get(idx, 0) == min_byes:
                return idx
        return ordered_indices[-1]

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
        steps = 0

        def backtrack(remaining: List[int]) -> Optional[Tuple[List[Tuple[int, int]], int]]:
            nonlocal steps
            steps += 1
            if steps > backtrack_limit:
                return None
            if not remaining:
                return [], 0

            first = remaining[0]
            # Prefer non-repeat pairs, then stable by index order.
            candidates = []
            for i in range(1, len(remaining)):
                second = remaining[i]
                pair_key = (min(first, second), max(first, second))
                repeat = pair_key in match_history
                candidates.append((repeat, second, i))

            candidates.sort(key=lambda x: (x[0], x[1]))

            best_pairs = None
            best_repeats = None

            for repeat, second, idx in candidates:
                next_remaining = remaining[1:idx] + remaining[idx + 1 :]
                result = backtrack(next_remaining)
                if result is None:
                    continue
                sub_pairs, sub_repeats = result
                total_repeats = (1 if repeat else 0) + sub_repeats
                if best_repeats is None or total_repeats < best_repeats:
                    best_pairs = [(first, second)] + sub_pairs
                    best_repeats = total_repeats
                    if best_repeats == 0:
                        break

            if best_pairs is None:
                return None
            return best_pairs, best_repeats

        result = backtrack(ordered_indices)
        if result is None:
            return None
        return result[0]

    def _generate_swiss_round_pairs(
        self,
        ranks: Dict[int, float],
        match_history: Set[Tuple[int, int]],
        bye_counts: Dict[int, int],
    ) -> Tuple[List[Tuple[int, int]], Optional[int]]:
        """
        Generate Swiss pairings for a single round with minimal repeat matchups.
        """
        ordered_indices = sorted(ranks.keys(), key=lambda i: (-ranks[i], i))

        bye_idx = None
        if len(ordered_indices) % 2 == 1:
            bye_idx = self._select_swiss_bye(ordered_indices, bye_counts)
            if bye_idx is not None:
                ordered_indices.remove(bye_idx)

        pairs = self._pair_players_swiss(ordered_indices, match_history)
        if pairs is None:
            # Fallback greedy pairing if backtracking exceeded limit
            pairs = []
            remaining = ordered_indices[:]
            while len(remaining) >= 2:
                a = remaining.pop(0)
                # Prefer non-repeat partners if available
                partner_idx = None
                for i, b in enumerate(remaining):
                    pair_key = (min(a, b), max(a, b))
                    if pair_key not in match_history:
                        partner_idx = i
                        break
                if partner_idx is None:
                    partner_idx = 0
                b = remaining.pop(partner_idx)
                pairs.append((a, b))

        # Update match history for this round
        for a, b in pairs:
            match_history.add((min(a, b), max(a, b)))

        if bye_idx is not None:
            bye_counts[bye_idx] = bye_counts.get(bye_idx, 0) + 1

        return pairs, bye_idx

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
            print(f"Error in compare_ideas: {e}")
            return None  # Return None instead of "tie" on error


class Breeder(LLMWrapper):
    """Breeds ideas and handles genotype encoding/decoding"""
    agent_name = "Breeder"
    parent_count = 2

    def __init__(self, mutation_rate: float = 0.0, **kwargs):
        # Don't set temperature directly here, let it come from kwargs
        super().__init__(agent_name=self.agent_name, **kwargs)
        self.mutation_rate = mutation_rate
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
                api_key=self.api_key
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
                idea_prompt = response.strip()
                oracle_analysis = (
                    "Oracle response was not properly formatted. Expected sections '=== ORACLE ANALYSIS ===' "
                    "and '=== IDEA PROMPT ===' but got unstructured response. This indicates the LLM did not follow the required format."
                )
                print("ORACLE: Fallback parsing - treating entire response as idea content. Response length:", len(response), "chars")
        except Exception as e:
            # Fallback parsing failed
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
