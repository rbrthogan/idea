from typing import List
import random

from idea.models import Idea
from idea.llm import LLMWrapper
from idea.operators import refine_idea, seed_ideas, remove_worst_idea, generate_new_idea
from tqdm import tqdm


class EvolutionEngine:
    def __init__(
        self,
        llm: LLMWrapper,
        seed_fn=seed_ideas,
        idea_type="airesearch",
        context_type="random_words",
        pop_size: int = 5,
        generations: int = 3,
    ):
        self.llm = llm
        self.seed_fn = seed_fn
        self.idea_type = idea_type
        self.context_type = context_type
        self.pop_size = pop_size
        self.generations = generations
        self.population: List[Idea] = []

        # Keep track of population changes across generations
        self.history = []  # List[List[Idea]]

    def run_evolution(self):
        """
        Main procedure:
        1. Seed
        2. For each generation:
            a) Refine each idea
            b) Breed subsets
        """
        # 1. Seed the population
        self.population = self.seed_fn(self.pop_size, idea_type=self.idea_type, context_type=self.context_type, llm=self.llm)
        self.population = [refine_idea(idea, self.llm) for idea in tqdm(self.population, desc="Refining ideas")]
        self.history.append(self.population)

        for gen in tqdm(range(self.generations), desc="Evolving"):
            # remove the worst idea
            # chunk the population into groups of 5
            chunk_size = 5
            new_population = []
            random.shuffle(self.population)
            for i in range(0, len(self.population), chunk_size):
                group = self.population[i : i + chunk_size]
                group = remove_worst_idea(group, self.llm)
                new_idea = generate_new_idea(group, self.llm)
                group.append(new_idea)
                new_population.extend(group)
            self.population = new_population


            # new_offspring is our new population for next generation
            self.history.append(self.population)

    def get_proposals_by_generation(self, generation_index: int) -> List[Idea]:
        if generation_index < 0 or generation_index >= len(self.history):
            return []
        return self.history[generation_index]
