import os
import hashlib
import math
import threading
import httpx
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
import asyncio
from idea.provider_execution import await_provider_call
import logging
from idea.config import DEFAULT_EMBEDDING_MODEL, embedding_prices_per_million_tokens

OPENAI_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Set up logging
logger = logging.getLogger(__name__)

def mean_pairwise_euclidean_distance(embeddings: List[np.ndarray]) -> float:
    """Exact mean distance without an n-by-n-by-dimensions temporary tensor."""
    if len(embeddings) < 2:
        return 0.0

    # Vectorized pairwise Euclidean distance computation.
    matrix = np.array(embeddings)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        return 0.0
    squared_lengths = np.sum(matrix * matrix, axis=1)
    total = 0.0
    count = 0
    for offset in range(0, len(matrix), 256):
        rows = matrix[offset:offset + 256]
        squared = squared_lengths[offset:offset + len(rows), None] + squared_lengths[None, :] - 2 * rows @ matrix.T
        distances = np.sqrt(np.maximum(squared, 0.0))
        for row_index, row in enumerate(distances):
            peers = row[offset + row_index + 1:]
            total += float(peers.sum())
            count += peers.size
    return total / count if count else 0.0


class DiversityCalculator:
    """
    Calculates population diversity using provider embeddings and mean square distance.

    This class generates embeddings for ideas and computes diversity metrics
    to track how the evolutionary algorithm's population changes over time.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        provider: str = "google_generative_ai",
    ):
        """Initialize the diversity calculator with the selected provider."""
        self.provider = provider
        if provider == "openai":
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
            self.embedding_model = os.environ.get(
                "OPENAI_EMBEDDING_MODEL", OPENAI_DEFAULT_EMBEDDING_MODEL
            )
            api_key_name = "OPENAI_API_KEY"
        else:
            self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
            self.embedding_model = os.environ.get(
                "GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL
            )
            api_key_name = "GEMINI_API_KEY"

        if not self.api_key:
            logger.warning("%s not found. Diversity calculation will be disabled.", api_key_name)
            self.client = None
        else:
            timeout_raw = os.environ.get("GENAI_HTTP_TIMEOUT_MS", "60000")
            try:
                http_timeout_ms = max(1000, int(timeout_raw))
            except ValueError:
                http_timeout_ms = 60000
                logger.warning(
                    "Invalid GENAI_HTTP_TIMEOUT_MS=%r. Falling back to %d ms.",
                    timeout_raw,
                    http_timeout_ms,
                )
            try:
                if provider == "openai":
                    base_url = os.environ.get(
                        "OPENAI_BASE_URL", "https://api.openai.com"
                    ).rstrip("/")
                    self.client = httpx.Client(
                        base_url=base_url,
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        timeout=http_timeout_ms / 1000,
                    )
                else:
                    self.client = genai.Client(
                        api_key=self.api_key,
                        http_options=types.HttpOptions(timeout=http_timeout_ms),
                    )
                logger.debug("DiversityCalculator initialized successfully")
            except Exception as e:
                logger.error("Failed to initialize %s embedding client: %s", provider, e)
                self.client = None

        # Cache for embeddings to avoid re-computing for same content
        self._embedding_cache = {}
        self.model_name = self.embedding_model
        self.input_token_count = 0
        self.output_token_count = 0
        self.total_token_count = 0
        self.estimated_input_token_count = 0
        self.embedding_request_count = 0
        self.embedding_token_counts_by_model: Dict[str, Dict[str, float]] = {}
        self._usage_lock = threading.Lock()
        # Embedding spaces cannot be mixed per text. Fail visibly on an
        # unavailable model; a model change requires re-embedding the set.
        self._embedding_models = [self.embedding_model]
        self._embedding_dimensions = None
        self._batch_size = 32

    def is_enabled(self) -> bool:
        """Check if diversity calculation is available."""
        return self.client is not None

    @staticmethod
    def _normalize_embedding_model_name(model_name: str) -> str:
        return (model_name or DEFAULT_EMBEDDING_MODEL).removeprefix("models/")

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        # Some embedding responses do not expose usage metadata. This
        # conservative estimate keeps embedding spend visible instead of free.
        return max(1, math.ceil(len(text or "") / 4))

    @staticmethod
    def _get_usage_value(usage: Any, *keys: str) -> Optional[int]:
        if usage is None:
            return None
        for key in keys:
            if isinstance(usage, dict):
                value = usage.get(key)
            else:
                value = getattr(usage, key, None)
            if value is None:
                continue
            try:
                numeric = int(value)
            except (TypeError, ValueError, OverflowError):
                continue
            if numeric >= 0 and not isinstance(value, bool):
                return numeric
        return None

    def _embedding_input_price(self, model_name: str) -> float:
        normalized = self._normalize_embedding_model_name(model_name)
        pricing = embedding_prices_per_million_tokens.get(normalized, {})
        return float(
            pricing.get("text_input")
            or pricing.get("input")
            or pricing.get("batch_input")
            or 0.0
        )

    def _record_embedding_usage(self, model_name: str, text: str, result: Any) -> None:
        with self._usage_lock:
            self._record_embedding_usage_unlocked(model_name, text, result)

    def _record_embedding_usage_unlocked(self, model_name: str, text: str, result: Any) -> None:
        usage = result.get("usage") if isinstance(result, dict) else (
            getattr(result, "usage_metadata", None)
            or getattr(result, "usageMetadata", None)
            or getattr(result, "metadata", None)
        )
        input_tokens = self._get_usage_value(
            usage,
            "prompt_token_count",
            "prompt_tokens",
            "input_tokens",
            "input_token_count",
            "text_token_count",
            "total_token_count",
        )
        used_estimate = input_tokens is None
        if input_tokens is None:
            input_tokens = self._estimate_text_tokens(text)

        normalized = self._normalize_embedding_model_name(model_name)
        input_cost = self._embedding_input_price(normalized) * input_tokens / 1_000_000

        self.input_token_count += input_tokens
        self.total_token_count += input_tokens
        self.embedding_request_count += 1
        if used_estimate:
            self.estimated_input_token_count += input_tokens

        model_counts = self.embedding_token_counts_by_model.setdefault(
            normalized,
            {
                "input": 0,
                "total": 0,
                "cost": 0.0,
                "requests": 0,
                "estimated_input": 0,
            },
        )
        model_counts["input"] += input_tokens
        model_counts["total"] += input_tokens
        model_counts["cost"] += input_cost
        model_counts["requests"] += 1
        if used_estimate:
            model_counts["estimated_input"] += input_tokens
        from idea.billing import record_price_usage
        from idea.config import MODEL_CATALOG_VERIFIED_AT
        snapshots = model_counts.setdefault("_usage_by_price", {})
        record_price_usage(snapshots, {
            "total_cost": input_cost, "input_rate": self._embedding_input_price(normalized),
            "pricing_known": normalized in embedding_prices_per_million_tokens,
            "price_version": MODEL_CATALOG_VERIFIED_AT,
            "price_effective_date": MODEL_CATALOG_VERIFIED_AT,
            "price_source": "https://ai.google.dev/gemini-api/docs/pricing" if self.provider == "google_generative_ai" else "https://developers.openai.com/api/docs/pricing/",
            "pricing_regime": "standard",
        }, {"input": input_tokens, "estimated_input": input_tokens if used_estimate else 0})

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Return embedding token usage in the same shape as LLM components."""
        with self._usage_lock:
            return self._token_usage_summary_unlocked()

    def _token_usage_summary_unlocked(self) -> Dict[str, Any]:
        total_cost = sum(
            float(model_counts.get("cost", 0.0))
            for model_counts in self.embedding_token_counts_by_model.values()
        )
        model_names = sorted(self.embedding_token_counts_by_model.keys())
        model_name = model_names[0] if len(model_names) == 1 else (
            "mixed-embeddings" if model_names else self._normalize_embedding_model_name(self.embedding_model)
        )
        return {
            "total": int(self.total_token_count),
            "input": int(self.input_token_count),
            "output": int(self.output_token_count),
            "model": model_name,
            "provider": self.provider,
            "cost": total_cost,
            "input_cost": total_cost,
            "output_cost": 0.0,
            "estimated_input": int(self.estimated_input_token_count),
            "requests": int(self.embedding_request_count),
            "missing_usage_requests": sum(int(counts.get("missing_usage_requests", 0)) for counts in self.embedding_token_counts_by_model.values()),
            "per_model": {
                model: {
                    "input": int(counts.get("input", 0)),
                    "total": int(counts.get("total", 0)),
                    "cost": float(counts.get("cost", 0.0)),
                    "requests": int(counts.get("requests", 0)),
                    "estimated_input": int(counts.get("estimated_input", 0)),
                    "missing_usage_requests": int(counts.get("missing_usage_requests", 0)),
                    "usage_by_price": [dict(row) for row in counts.get("_usage_by_price", {}).values()],
                    "usage_basis": "includes_local_estimates" if counts.get("estimated_input") else "provider_reported",
                }
                for model, counts in self.embedding_token_counts_by_model.items()
            },
        }

    def _get_idea_text(self, idea_dict: Dict[str, Any]) -> str:
        """
        Extract meaningful text from an idea dictionary for embedding.

        Args:
            idea_dict: Dictionary containing idea information

        Returns:
            Combined text representation of the idea
        """
        try:
            # Handle different idea dictionary formats
            if "idea" in idea_dict:
                idea_obj = idea_dict["idea"]
                if isinstance(idea_obj, str):
                    return idea_obj.strip()
                if isinstance(idea_obj, dict):
                    return f"{idea_obj.get('title') or ''}\n\n{idea_obj.get('content') or ''}".strip()
                if hasattr(idea_obj, 'title') and hasattr(idea_obj, 'content'):
                    title = idea_obj.title or ""
                    content = idea_obj.content or ""
                    return f"{title}\n\n{content}".strip()
                elif hasattr(idea_obj, 'content'):
                    return idea_obj.content or ""

            # Fallback: try to extract from dictionary keys
            title = idea_dict.get('title', '')
            content = idea_dict.get('content', '')
            if title or content:
                return f"{title}\n\n{content}".strip()

            # Last resort: convert entire dict to string
            return str(idea_dict)

        except Exception as e:
            logger.warning(f"Error extracting text from idea: {e}")
            return str(idea_dict)

    def _cache_key(self, text: str) -> str:
        identity = f"v2|{self.provider}|{self.embedding_model}|{text}"
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _embed_batch_sync(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """One pinned-space provider request, executed outside the event loop."""
        cancel_event = getattr(self, "cancel_event", None)
        cancel_check = getattr(self, "cancel_check", None)
        if ((cancel_event is not None and cancel_event.is_set()) or
                (callable(cancel_check) and cancel_check())):
            return [None] * len(texts)
        request_started = usage_recorded = False
        try:
            request_input = texts[0] if len(texts) == 1 else texts
            if self.provider == "openai":
                request_started = True
                response = self.client.post("/v1/embeddings", json={
                    "model": self.embedding_model, "input": request_input})
                response.raise_for_status()
                result = response.json()
                data = result.get("data", [])
                if any("index" in item for item in data):
                    data = sorted(data, key=lambda item: item.get("index", 0))
                values = [item.get("embedding") for item in data]
            else:
                kwargs = {"model": self.embedding_model, "contents": request_input}
                if self.embedding_model.removeprefix("models/").startswith("gemini-embedding-2"):
                    # Embedding 2 aggregates list[str] into one multimodal item.
                    # Explicit Content objects preserve one vector per candidate.
                    kwargs["contents"] = [types.Content(parts=[types.Part.from_text(text=text)]) for text in texts]
                else:
                    kwargs["config"] = types.EmbedContentConfig(task_type="CLUSTERING")
                request_started = True
                result = self.client.models.embed_content(**kwargs)
                values = getattr(result, "embeddings", None)
                if not values:
                    value = getattr(result, "embedding", None)
                    values = [value] if value is not None else []
                values = [value.get("values") if isinstance(value, dict)
                          else getattr(value, "values", value) for value in values]
            # A successful provider response may already have incurred spend,
            # even when its vectors subsequently fail validation.
            self._record_embedding_usage(self.embedding_model, "\n".join(texts), result)
            usage_recorded = True
            if len(values) != len(texts):
                raise ValueError("Embedding response count did not match request")
            embeddings = [np.asarray(value, dtype=float) for value in values]
            dimensions = {embedding.size for embedding in embeddings}
            if len(dimensions) != 1 or any(embedding.ndim != 1 or not embedding.size
                or not np.isfinite(embedding).all() for embedding in embeddings):
                raise ValueError("Invalid or inconsistent embedding vectors")
            dimension = dimensions.pop()
            if self._embedding_dimensions is not None and dimension != self._embedding_dimensions:
                raise ValueError("Embedding dimensions changed within the pinned model")
            self._embedding_dimensions = dimension
            return embeddings
        except Exception as exc:
            if request_started and not usage_recorded:
                with self._usage_lock:
                    model = self._normalize_embedding_model_name(self.embedding_model)
                    counts = self.embedding_token_counts_by_model.setdefault(model, {
                        "input": 0, "total": 0, "cost": 0.0, "requests": 0, "estimated_input": 0,
                    })
                    counts["missing_usage_requests"] = counts.get("missing_usage_requests", 0) + 1
            logger.error("Embedding request failed for %s: %s", self.embedding_model, exc)
            return [None] * len(texts)

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        return (await self._get_embeddings_batch([text]))[0]

    async def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        if not self.client:
            return [None] * len(texts)
        cleaned = [(text or "").strip() for text in texts]
        missing = list(dict.fromkeys(text for text in cleaned if text
            and self._cache_key(text) not in self._embedding_cache))
        # Bounded provider batches keep request size and worker count predictable.
        for offset in range(0, len(missing), self._batch_size):
            batch = missing[offset:offset + self._batch_size]
            embeddings = await await_provider_call(self._embed_batch_sync, batch)
            for text, embedding in zip(batch, embeddings):
                if embedding is not None:
                    self._embedding_cache[self._cache_key(text)] = embedding
            while len(self._embedding_cache) > 10000:
                self._embedding_cache.pop(next(iter(self._embedding_cache)))
        return [self._embedding_cache.get(self._cache_key(text)) if text else None
                for text in cleaned]

    def _calculate_mean_square_distance(self, embeddings: List[np.ndarray]) -> float:
        """Compatibility name for the exact mean Euclidean distance."""
        return mean_pairwise_euclidean_distance(embeddings)

    def _calculate_inter_generation_diversity(self, history_embeddings: List[List[np.ndarray]]) -> Optional[float]:
        """
        Calculate inter-generation diversity by computing centroids of each generation
        and measuring mean euclidean distances between all pairs of centroids.

        Args:
            history_embeddings: List of generations, each containing a list of embedding vectors

        Returns:
            Mean euclidean distance between generation centroids, or None if insufficient generations
        """
        if len(history_embeddings) < 2:
            return None

        # Calculate centroid for each generation
        centroids = []
        for gen_embeddings in history_embeddings:
            if len(gen_embeddings) == 0:
                continue
            # Calculate centroid as mean of all embeddings in the generation
            centroid = np.mean(gen_embeddings, axis=0)
            centroids.append(centroid)

        # Calculate mean euclidean distance between all pairs of centroids
        if len(centroids) < 2:
            return None

        return self._calculate_mean_square_distance(centroids)

    async def calculate_diversity(
        self,
        history: List[List[Dict[str, Any]]],
        precomputed_embeddings: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Dict[str, Any]:
        """
        Calculate diversity metrics for the entire population history.

        Args:
            history: List of generations, each containing a list of idea dictionaries

        Returns:
            Dictionary containing diversity metrics and metadata
        """
        if not self.is_enabled():
            return {
                "enabled": False,
                "reason": f"{self.provider} API key not available",
                "diversity_score": 0.0,
                "total_ideas": 0,
                "generation_diversities": []
            }

        try:
            logger.info(f"Calculating diversity for {len(history)} generations")

            # Flatten all ideas from all generations
            all_ideas = []
            generation_idea_counts = []

            for generation in history:
                all_ideas.extend(generation)
                generation_idea_counts.append(len(generation))

            if not all_ideas:
                return {
                    "enabled": True,
                    "diversity_score": 0.0,
                    "total_ideas": 0,
                    "generation_diversities": [],
                    "embedding_success_rate": 0.0
                }

            # Extract text from all ideas
            all_texts = [self._get_idea_text(idea) for idea in all_ideas]
            logger.info(f"Extracted text from {len(all_texts)} ideas")

            # Get embeddings for all ideas (or reuse precomputed values from the engine).
            if precomputed_embeddings is not None and len(precomputed_embeddings) == len(all_ideas):
                all_embeddings = precomputed_embeddings
            else:
                all_embeddings = await self._get_embeddings_batch(all_texts)

            # Filter out failed embeddings
            valid_embeddings = [emb for emb in all_embeddings if emb is not None]
            embedding_success_rate = len(valid_embeddings) / len(all_embeddings) if all_embeddings else 0.0

            logger.info(f"Generated {len(valid_embeddings)} valid embeddings out of {len(all_embeddings)} attempts")

            if len(valid_embeddings) < 2:
                return {
                    "enabled": True,
                    "diversity_score": 0.0,
                    "total_ideas": len(all_ideas),
                    "generation_diversities": [],
                    "embedding_success_rate": embedding_success_rate,
                    "warning": "Insufficient valid embeddings for diversity calculation"
                }

            # Calculate overall diversity
            overall_diversity = await await_provider_call(self._calculate_mean_square_distance, valid_embeddings)

            # Calculate per-generation diversity and organize embeddings by generation
            generation_diversities = []
            history_embeddings = []  # For inter-generation diversity calculation
            embedding_idx = 0

            for gen_idx, gen_count in enumerate(generation_idea_counts):
                gen_embeddings = []
                for _ in range(gen_count):
                    if embedding_idx < len(all_embeddings) and all_embeddings[embedding_idx] is not None:
                        gen_embeddings.append(all_embeddings[embedding_idx])
                    embedding_idx += 1

                # Add to history embeddings for inter-generation calculation
                history_embeddings.append(gen_embeddings)

                if len(gen_embeddings) >= 2:
                    gen_diversity = (overall_diversity if len(history) == 1 else
                        await await_provider_call(self._calculate_mean_square_distance, gen_embeddings))
                else:
                    gen_diversity = 0.0

                generation_diversities.append({
                    "generation": gen_idx,
                    "diversity_score": gen_diversity,
                    "idea_count": gen_count,
                    "valid_embeddings": len(gen_embeddings)
                })

            # Calculate inter-generation diversity
            inter_generation_diversity = await await_provider_call(self._calculate_inter_generation_diversity, history_embeddings)

            diversity_result = {
                "enabled": True,
                "diversity_score": overall_diversity,
                "inter_generation_diversity": inter_generation_diversity,
                "total_ideas": len(all_ideas),
                "valid_embeddings": len(valid_embeddings),
                "embedding_success_rate": embedding_success_rate,
                "generation_diversities": generation_diversities,
                "embedding_dimensions": len(valid_embeddings[0]) if valid_embeddings else 0,
                "calculation_method": "mean_square_distance"
            }

            logger.info(f"Diversity calculation complete. Overall score: {overall_diversity:.4f}")
            return diversity_result

        except Exception as e:
            logger.error(f"Error calculating diversity: {e}")
            return {
                "enabled": True,
                "error": str(e),
                "diversity_score": 0.0,
                "total_ideas": len(sum(history, [])),
                "generation_diversities": []
            }

    def print_diversity_summary(self, diversity_data: Dict[str, Any]) -> None:
        """
        Print a formatted summary of diversity metrics to the console.

        Args:
            diversity_data: Diversity calculation results
        """
        if not diversity_data.get("enabled", False):
            reason = diversity_data.get("reason", "Unknown reason")
            print(f"🔍 Diversity Tracking: Disabled ({reason})")
            return

        if "error" in diversity_data:
            print(f"🔍 Diversity Tracking: Error - {diversity_data['error']}")
            return

        print("🔍 DIVERSITY METRICS 🔍")
        print("=" * 50)
        print(f"Overall Diversity Score: {diversity_data['diversity_score']:.4f}")
        inter_gen_diversity = diversity_data.get('inter_generation_diversity', None)
        if inter_gen_diversity is not None:
            print(f"Inter-Generation Diversity: {inter_gen_diversity:.4f}")
        else:
            print("Inter-Generation Diversity: N/A (requires 2+ generations)")
        print(f"Total Ideas Analyzed: {diversity_data['total_ideas']}")
        sampling = diversity_data.get("history_sampling")
        if sampling:
            print(f"Historical vector sample: {sampling['sample_size']}/{sampling['total_size']} unique revisions")
        print(f"Valid Embeddings: {diversity_data['valid_embeddings']}")
        print(f"Embedding Success Rate: {diversity_data['embedding_success_rate']:.1%}")

        if diversity_data.get("generation_diversities"):
            print("\nPer-Generation Diversity:")
            print("-" * 40)
            for gen_data in diversity_data["generation_diversities"]:
                gen_num = gen_data["generation"]
                score = gen_data["diversity_score"]
                count = gen_data["idea_count"]
                valid = gen_data["valid_embeddings"]

                gen_label = "Initial" if gen_num == 0 else f"Gen {gen_num}"
                print(f"  {gen_label:>8}: {score:>8.4f} ({valid}/{count} valid embeddings)")

        if diversity_data.get("warning"):
            print(f"\n⚠️  Warning: {diversity_data['warning']}")

        print("=" * 50)
