from typing import List
import random
import asyncio

from idea.models import Idea
from idea.llm import Ideator, Formatter, Critic
from tqdm import tqdm


class EvolutionEngine:
    def __init__(
        self,
        idea_type="airesearch",
        context_type="random_words",
        pop_size: int = 5,
        generations: int = 3,
        model_type: str = "gemini-1.5-flash"
    ):
        self.idea_type = idea_type
        self.context_type = context_type
        self.pop_size = pop_size
        self.generations = generations
        self.population: List[Idea] = []
        # TODO: make this configurable with a dropdown list for each LLM type using the following models:
        # gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-01-21
        self.ideator = Ideator(provider="google_generative_ai", model_name=model_type)
        self.formatter = Formatter(provider="google_generative_ai", model_name="gemini-1.5-flash")
        self.critic = Critic(provider="google_generative_ai", model_name=model_type)
        self.history = []  # List[List[Idea]]

    def run_evolution(self):
        """Runs the complete evolution process"""
        print("Starting evolution...")

        # Seed the initial population
        print("Generating initial population...")
        self.population = self.ideator.seed_ideas(self.pop_size, self.context_type, self.idea_type)
        print("Refining initial population...")
        # self.population = [self.critic.refine(idea) for idea in self.population]
        print("Formatting initial population...")
        self.population = [self.formatter.format_idea(idea, self.idea_type) for idea in self.population]
        self.history.append(self.population)
        print(f"Initial population size: {len(self.population)}")

        # Run evolution for specified number of generations
        for gen in range(self.generations):
            print(f"Starting generation {gen + 1}...")
            chunk_size = 5
            new_population = []
            random.shuffle(self.population)

            # Process population in chunks
            for i in range(0, len(self.population), chunk_size):
                group = self.population[i : i + chunk_size]
                group = self.critic.remove_worst_idea(group)
                new_idea = self.ideator.generate_new_idea(group)
                group.append(new_idea)
                new_population.extend(group)

            # Update population with refined ideas
            print(f"Refining generation {gen + 1}...")
            self.population = new_population
            # self.population = [self.critic.refine(idea) for idea in self.population]
            self.population = [self.formatter.format_idea(idea, self.idea_type) for idea in self.population]
            self.history.append(self.population)
            print(f"Generation {gen + 1} complete. Population size: {len(self.population)}")

        print("Evolution complete!")

    def get_proposals_by_generation(self, generation_index: int) -> List[Idea]:
        if generation_index < 0 or generation_index >= len(self.history):
            return []
        return self.history[generation_index]
