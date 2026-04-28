import os
import hashlib
import math
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
import asyncio
import logging
from idea.config import DEFAULT_EMBEDDING_MODEL, embedding_prices_per_million_tokens

# Set up logging
logger = logging.getLogger(__name__)

class DiversityCalculator:
    """
    Calculates population diversity using Gemini embeddings and mean square distance.

    This class generates embeddings for ideas and computes diversity metrics
    to track how the evolutionary algorithm's population changes over time.
    """

    def __init__(self, api_key: Optional[str] = None):
        """Initialize the diversity calculator with Gemini client."""
        # Use provided key or fallback to environment variable
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.embedding_model = os.environ.get("GEMINI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)

        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Diversity calculation will be disabled.")
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
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(timeout=http_timeout_ms),
                )
                logger.debug("DiversityCalculator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
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
        # Fallback models to try if the primary model fails or returns no embeddings
        fallback_models = [
            "models/gemini-embedding-2",
            "gemini-embedding-2",
            "models/gemini-embedding-001",
            "gemini-embedding-001",
            "models/text-embedding-004",
            "text-embedding-004",
            "gemini-embedding-exp-03-07",
        ]
        self._embedding_models = [self.embedding_model] + [
            model for model in fallback_models if model != self.embedding_model
        ]

    def is_enabled(self) -> bool:
        """Check if diversity calculation is available."""
        return self.client is not None

    @staticmethod
    def _normalize_embedding_model_name(model_name: str) -> str:
        return (model_name or DEFAULT_EMBEDDING_MODEL).removeprefix("models/")

    @staticmethod
    def _estimate_text_tokens(text: str) -> int:
        # Gemini embedding responses do not always expose usage metadata. This
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
            except (TypeError, ValueError):
                continue
            if numeric >= 0:
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
        usage = (
            getattr(result, "usage_metadata", None)
            or getattr(result, "usageMetadata", None)
            or getattr(result, "metadata", None)
        )
        input_tokens = self._get_usage_value(
            usage,
            "prompt_token_count",
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

    def get_token_usage_summary(self) -> Dict[str, Any]:
        """Return embedding token usage in the same shape as LLM components."""
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
            "cost": total_cost,
            "input_cost": total_cost,
            "output_cost": 0.0,
            "estimated_input": int(self.estimated_input_token_count),
            "requests": int(self.embedding_request_count),
            "per_model": {
                model: {
                    "input": int(counts.get("input", 0)),
                    "total": int(counts.get("total", 0)),
                    "cost": float(counts.get("cost", 0.0)),
                    "requests": int(counts.get("requests", 0)),
                    "estimated_input": int(counts.get("estimated_input", 0)),
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

    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Get embedding for a text using Gemini API with caching.

        Args:
            text: Text to embed

        Returns:
            Numpy array of embedding values, or None if failed
        """
        if not self.client:
            return None

        cleaned_text = (text or "").strip()
        if not cleaned_text:
            return None

        # Use stable text hash as cache key
        cache_key = hashlib.sha256(cleaned_text.encode("utf-8")).hexdigest()
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        def _extract_embedding(result) -> Optional[np.ndarray]:
            if result is None:
                return None
            # Newer responses may include embeddings list
            embeddings = getattr(result, "embeddings", None)
            if embeddings:
                candidate = embeddings[0]
                values = getattr(candidate, "values", None)
                if values is None and isinstance(candidate, dict):
                    values = candidate.get("values")
                if values is None:
                    values = candidate
                return np.array(values)
            # Some responses may include a single embedding field
            embedding = getattr(result, "embedding", None)
            if embedding is not None:
                values = getattr(embedding, "values", None)
                if values is None and isinstance(embedding, dict):
                    values = embedding.get("values")
                if values is None:
                    values = embedding
                return np.array(values)
            return None

        def _embedding_config_for_model(model_name: str):
            bare_model_name = model_name.removeprefix("models/")
            if bare_model_name == "gemini-embedding-2":
                return None
            return types.EmbedContentConfig(task_type="CLUSTERING")

        try:
            last_error = None
            for model_name in self._embedding_models:
                try:
                    request_config = _embedding_config_for_model(model_name)
                    request_kwargs = {
                        "model": model_name,
                        "contents": cleaned_text,
                    }
                    if request_config is not None:
                        request_kwargs["config"] = request_config

                    # Try embedding call with single string
                    result = self.client.models.embed_content(**request_kwargs)
                    embedding = _extract_embedding(result)
                    if embedding is None:
                        # Some SDK versions expect a list of contents
                        request_kwargs["contents"] = [cleaned_text]
                        result = self.client.models.embed_content(**request_kwargs)
                        embedding = _extract_embedding(result)
                    if embedding is not None:
                        self._record_embedding_usage(model_name, cleaned_text, result)
                        self._embedding_cache[cache_key] = embedding
                        return embedding
                except Exception as e:
                    last_error = e
                    continue
            if last_error:
                logger.error(f"Error getting embedding from Gemini: {last_error}")
            else:
                logger.warning("No embeddings returned from Gemini API")
            return None

        except Exception as e:
            logger.error(f"Unexpected error getting embedding from Gemini: {e}")
            return None

    async def _get_embeddings_batch(self, texts: List[str]) -> List[Optional[np.ndarray]]:
        """
        Get embeddings for multiple texts concurrently.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding arrays (or None for failed embeddings)
        """
        if not self.client:
            return [None] * len(texts)

        # Process embeddings concurrently
        tasks = [self._get_embedding(text) for text in texts]
        embeddings = await asyncio.gather(*tasks, return_exceptions=True)

        # Handle exceptions in results
        result_embeddings = []
        for i, embedding in enumerate(embeddings):
            if isinstance(embedding, Exception):
                logger.warning(f"Failed to get embedding for text {i}: {embedding}")
                result_embeddings.append(None)
            else:
                result_embeddings.append(embedding)

        return result_embeddings

    def _calculate_mean_square_distance(self, embeddings: List[np.ndarray]) -> float:
        """
        Calculate Euclidean distance between all pairs of embeddings and return the mean.

        Args:
            embeddings: List of embedding vectors

        Returns:
            Mean Euclidean distance value
        """
        if len(embeddings) < 2:
            return 0.0

        # Vectorized pairwise Euclidean distance computation.
        matrix = np.array(embeddings)
        if matrix.ndim != 2 or matrix.shape[0] < 2:
            return 0.0
        deltas = matrix[:, None, :] - matrix[None, :, :]
        distances = np.linalg.norm(deltas, axis=2)
        upper = np.triu_indices(matrix.shape[0], k=1)
        if upper[0].size == 0:
            return 0.0
        return float(np.mean(distances[upper]))

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
                "reason": "Gemini API key not available",
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
            overall_diversity = self._calculate_mean_square_distance(valid_embeddings)

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
                    gen_diversity = self._calculate_mean_square_distance(gen_embeddings)
                else:
                    gen_diversity = 0.0

                generation_diversities.append({
                    "generation": gen_idx,
                    "diversity_score": gen_diversity,
                    "idea_count": gen_count,
                    "valid_embeddings": len(gen_embeddings)
                })

            # Calculate inter-generation diversity
            inter_generation_diversity = self._calculate_inter_generation_diversity(history_embeddings)

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
