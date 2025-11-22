import os
import numpy as np
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
import asyncio
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DiversityCalculator:
    """
    Calculates population diversity using Gemini embeddings and mean square distance.

    This class generates embeddings for ideas and computes diversity metrics
    to track how the evolutionary algorithm's population changes over time.
    """

    def __init__(self):
        """Initialize the diversity calculator with Gemini client."""
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("GEMINI_API_KEY not found. Diversity calculation will be disabled.")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=self.api_key)
                logger.info("DiversityCalculator initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini client: {e}")
                self.client = None

        # Cache for embeddings to avoid re-computing for same content
        self._embedding_cache = {}

    def is_enabled(self) -> bool:
        """Check if diversity calculation is available."""
        return self.client is not None

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

        # Use text hash as cache key
        cache_key = hash(text)
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]

        try:
            # Generate embedding using Gemini
            result = self.client.models.embed_content(
                model="gemini-embedding-exp-03-07",
                contents=text,
                config=types.EmbedContentConfig(task_type="CLUSTERING")
            )

            if result.embeddings and len(result.embeddings) > 0:
                # Convert to numpy array
                embedding = np.array(result.embeddings[0].values)
                self._embedding_cache[cache_key] = embedding
                return embedding
            else:
                logger.warning("No embeddings returned from Gemini API")
                return None

        except Exception as e:
            logger.error(f"Error getting embedding from Gemini: {e}")
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

        n = len(embeddings)
        total_distance = 0.0
        pair_count = 0

        # Calculate pairwise distances
        for i in range(n):
            for j in range(i + 1, n):
                # Euclidean distance
                distance = np.sqrt(np.sum((embeddings[i] - embeddings[j]) ** 2))
                total_distance += distance
                pair_count += 1

        # Return mean square distance
        return total_distance / pair_count if pair_count > 0 else 0.0

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

    async def calculate_diversity(self, history: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
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

            # Get embeddings for all ideas
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