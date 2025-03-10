from typing import List, Dict, Any, Callable, Awaitable
import random
import numpy as np
import asyncio
from idea.models import Idea
from idea.llm import Ideator, Formatter, Critic, Breeder
from tqdm import tqdm


class EvolutionEngine:
    def __init__(
        self,
        idea_type="airesearch",
        pop_size: int = 5,
        generations: int = 3,
        model_type: str = "gemini-1.5-flash",
        ideator_temp: float = 2.0,
        critic_temp: float = 1.5,
        breeder_temp: float = 2.0,
        tournament_size: int = 5,
        tournament_comparisons: int = 20
    ):
        self.idea_type = idea_type
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.tournament_comparisons = tournament_comparisons
        self.population: List[Idea] = []
        # TODO: make this configurable with a dropdown list for each LLM type using the following models:
        # gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-01-21

        # Initialize LLM components with appropriate temperatures
        print(f"Initializing agents with temperatures: Ideator={ideator_temp}, Critic={critic_temp}, Breeder={breeder_temp}")
        self.ideator = Ideator(provider="google_generative_ai", model_name=model_type, temperature=ideator_temp)
        self.formatter = Formatter(provider="google_generative_ai", model_name="gemini-1.5-flash")
        self.critic = Critic(provider="google_generative_ai", model_name=model_type, temperature=critic_temp)
        self.breeder = Breeder(provider="google_generative_ai", model_name=model_type, temperature=breeder_temp)

        self.history = []  # List[List[Idea]]
        self.contexts = []  # List of contexts for the initial population

    def generate_contexts(self):
        """Generate contexts for the initial population"""
        self.contexts = []
        for _ in range(self.pop_size):
            context = self.ideator.generate_context(self.idea_type)
            self.contexts.append(context)
        return self.contexts

    async def run_evolution_with_updates(self, progress_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Runs the evolution process with progress updates

        Args:
            progress_callback: Async function that will be called with progress updates
        """
        try:
            # Seed the initial population
            print("Generating initial population (Generation 0)...")
            self.population = self.ideator.seed_ideas(self.pop_size, self.idea_type)

            # Process and update each idea as it's completed
            print("Refining initial population...")
            for i, idea in enumerate(self.population):
                refined_idea = self.critic.refine(idea, self.idea_type)
                formatted_idea = self.formatter.format_idea(refined_idea, self.idea_type)
                self.population[i] = formatted_idea

                # Calculate progress
                progress_percent = (i + 1) / (self.pop_size * (self.generations + 1)) * 100

                # Create a partial history for the progress update
                current_history = [self.population[:i+1]]

                # Send progress update
                await progress_callback({
                    "current_generation": 0,
                    "total_generations": self.generations,
                    "is_running": True,
                    "history": current_history,
                    "contexts": self.contexts,
                    "progress": progress_percent
                })

                # Small delay to allow frontend to process updates
                await asyncio.sleep(0.1)

            self.history = [self.population.copy()]

            # Run evolution for specified number of generations
            for gen in range(self.generations):
                print(f"Starting generation {gen + 1}...")

                # Adjust tournament size if population is too small
                actual_tournament_size = self.tournament_size
                if len(self.population) < 2 * actual_tournament_size:
                    actual_tournament_size = max(2, len(self.population) // 2)
                    print(f"Adjusting tournament size to {actual_tournament_size} due to small population")

                new_population = []
                random.shuffle(self.population)

                # Process population in chunks
                for i in range(0, len(self.population), actual_tournament_size):
                    group = self.population[i : i + actual_tournament_size]
                    ranks = self.critic.get_tournament_ranks(group, self.idea_type, self.tournament_comparisons)

                    for idea_idx, rank in sorted(ranks.items(), key=lambda x: x[1]):
                        # Extract title from the idea object within the dictionary
                        idea_obj = group[idea_idx]["idea"]
                        title = idea_obj.title if hasattr(idea_obj, 'title') else "Untitled"
                        print(f"{title} - {rank}")

                    for k in range(len(group)):
                        # using tournament ranks, weighted sample from group
                        weights = [ranks[i] for i in range(len(group))]
                        # normalize weights to be between 0 and 1
                        # Note: this shift will cause the lowest ranked idea to have a weight of 0, eliminating it from selection
                        weights = [(w - min(weights)) / (max(weights) - min(weights)) for w in weights]
                        weights = [w / sum(weights) for w in weights]

                        # Select parents and breed
                        parent_indices = np.random.choice(list(ranks.keys()), size=self.breeder.parent_count, p=weights, replace=False)
                        parent_ideas = [group[idx] for idx in parent_indices]
                        new_idea = self.breeder.breed(parent_ideas, self.idea_type)

                        # Format the idea and add to new population
                        formatted_idea = self.formatter.format_idea(new_idea, self.idea_type)
                        new_population.append(formatted_idea)

                        # Calculate overall progress
                        total_ideas = self.pop_size * (self.generations + 1)
                        completed_ideas = self.pop_size + (gen * self.pop_size) + len(new_population)
                        progress_percent = (completed_ideas / total_ideas) * 100

                        # Create a copy of the history with the current generation's progress
                        history_copy = self.history.copy()
                        history_copy.append(new_population.copy())

                        # Send progress update
                        await progress_callback({
                            "current_generation": gen + 1,
                            "total_generations": self.generations,
                            "is_running": True,
                            "history": history_copy,
                            "contexts": self.contexts,
                            "progress": progress_percent
                        })

                        # Small delay to allow frontend to process updates
                        await asyncio.sleep(0.1)

                # Update population with new ideas
                self.population = new_population
                self.history.append(self.population)
                print(f"Generation {gen + 1} complete. Population size: {len(self.population)}")

            # Mark evolution as complete
            await progress_callback({
                "current_generation": self.generations,
                "total_generations": self.generations,
                "is_running": False,
                "history": self.history,
                "contexts": self.contexts,
                "progress": 100
            })
            print("Evolution complete!")

        except Exception as e:
            import traceback
            print(f"Error in evolution: {e}")
            print(traceback.format_exc())
            await progress_callback({
                "is_running": False,
                "error": str(e)
            })

    def get_proposals_by_generation(self, generation_index: int) -> List[Dict]:
        """
        Get all proposals from a specific generation

        Args:
            generation_index: The index of the generation to retrieve

        Returns:
            List of idea dictionaries with 'id' and 'idea' keys
        """
        if generation_index < 0 or generation_index >= len(self.history):
            return []
        return self.history[generation_index]
