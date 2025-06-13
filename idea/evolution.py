from typing import List, Dict, Any, Callable, Awaitable
import random
import numpy as np
import asyncio
from idea.models import Idea
from idea.llm import Ideator, Formatter, Critic, Breeder, GenotypeEncoder
from idea.prompts.loader import list_available_templates
from tqdm import tqdm


def get_default_template_id():
    """Get the first available template ID as default"""
    try:
        templates = list_available_templates()
        # Find the first template without errors
        for template_id, template_info in templates.items():
            if 'error' not in template_info:
                return template_id
        # Fallback to airesearch if nothing else works
        return 'airesearch'
    except:
        return 'airesearch'


class EvolutionEngine:
    def __init__(
        self,
        idea_type=None,
        pop_size: int = 5,
        generations: int = 3,
        model_type: str = "gemini-2.0-flash",
        ideator_temp: float = 2.0,
        critic_temp: float = 1.5,
        breeder_temp: float = 2.0,
        tournament_size: int = 5,
        tournament_comparisons: int = 20,
        use_genotype_breeding: bool = False,
        genotype_encoder_temp: float = 1.2
    ):
        self.idea_type = idea_type or get_default_template_id()
        self.pop_size = pop_size
        self.generations = generations
        self.tournament_size = tournament_size
        self.tournament_comparisons = tournament_comparisons
        self.use_genotype_breeding = use_genotype_breeding
        self.population: List[Idea] = []
        # TODO: make this configurable with a dropdown list for each LLM type using the following models:
        # gemini-1.5-flash, gemini-2.0-flash-exp, gemini-2.0-flash-thinking-exp-01-21

        # Initialize LLM components with appropriate temperatures
        print(f"Initializing agents with temperatures: Ideator={ideator_temp}, Critic={critic_temp}, Breeder={breeder_temp}")
        if use_genotype_breeding:
            print(f"Genotype breeding enabled with GenotypeEncoder temperature: {genotype_encoder_temp}")

        self.ideator = Ideator(provider="google_generative_ai", model_name=model_type, temperature=ideator_temp)
        self.formatter = Formatter(provider="google_generative_ai", model_name="gemini-1.5-flash")
        self.critic = Critic(provider="google_generative_ai", model_name=model_type, temperature=critic_temp)
        self.breeder = Breeder(provider="google_generative_ai", model_name=model_type, temperature=breeder_temp)

        # Initialize genotype encoder if genotype breeding is enabled
        if use_genotype_breeding:
            self.genotype_encoder = GenotypeEncoder(provider="google_generative_ai", model_name=model_type, temperature=genotype_encoder_temp)
        else:
            self.genotype_encoder = None

        self.history = []  # List[List[Idea]]
        self.contexts = []  # List of contexts for the initial population

        # Add stop flag for graceful interruption
        self.stop_requested = False
        self.is_stopped = False

    def generate_contexts(self):
        """Generate contexts for the initial population"""
        self.contexts = []
        for _ in range(self.pop_size):
            context = self.ideator.generate_context(self.idea_type)
            self.contexts.append(context)
        return self.contexts

    def stop_evolution(self):
        """Request the evolution to stop gracefully"""
        print("Stop requested - evolution will halt at the next safe point")
        self.stop_requested = True

    def reset_stop_state(self):
        """Reset the stop state for a new evolution"""
        self.stop_requested = False
        self.is_stopped = False

    async def run_evolution_with_updates(self, progress_callback: Callable[[Dict[str, Any]], Awaitable[None]]):
        """
        Runs the evolution process with progress updates

        Args:
            progress_callback: Async function that will be called with progress updates
        """
        try:
            # Reset stop state at the beginning
            self.reset_stop_state()

            # Seed the initial population
            print("Generating initial population (Generation 0)...")
            self.population = self.ideator.seed_ideas(self.pop_size, self.idea_type)

            # Process and update each idea as it's completed
            print("Refining initial population...")
            for i, idea in enumerate(self.population):
                # Check for stop request
                if self.stop_requested:
                    print("Stop requested during initial population generation")
                    self.is_stopped = True
                    await progress_callback({
                        "current_generation": 0,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "history": [self.population[:i]] if i > 0 else [],
                        "contexts": self.contexts,
                        "progress": (i / (self.pop_size * (self.generations + 1))) * 100,
                        "stop_message": f"Evolution stopped during initial generation (completed {i}/{self.pop_size} ideas)"
                    })
                    return

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

                # Small delay to allow frontend to process updates and check for stop
                await asyncio.sleep(0.1)

            self.history = [self.population.copy()]

            # Run evolution for specified number of generations
            for gen in range(self.generations):
                # Check for stop request at the beginning of each generation
                if self.stop_requested:
                    self.is_stopped = True
                    print(f"Stop requested - evolution halted after generation {gen}")
                    await progress_callback({
                        "current_generation": gen,
                        "total_generations": self.generations,
                        "is_running": False,
                        "is_stopped": True,
                        "history": self.history,
                        "contexts": self.contexts,
                        "progress": ((self.pop_size + gen * self.pop_size) / (self.pop_size * (self.generations + 1))) * 100,
                        "stop_message": f"Evolution stopped after completing generation {gen}",
                        "token_counts": self.get_total_token_count()
                    })
                    return

                print(f"Starting generation {gen + 1}...")

                # Adjust tournament size if population is too small
                actual_tournament_size = self.tournament_size
                if len(self.population) < 2 * actual_tournament_size:
                    actual_tournament_size = max(3, len(self.population) // 2)
                    print(f"Adjusting tournament size to {actual_tournament_size} due to small population")

                new_population = []
                random.shuffle(self.population)

                # Process population in chunks
                for i in range(0, len(self.population), actual_tournament_size):
                    # Check for stop request during generation processing
                    if self.stop_requested:
                        self.is_stopped = True
                        print(f"Stop requested - evolution halted during generation {gen + 1}")

                        # If we have some new population, add it to history
                        if new_population:
                            self.history.append(new_population)

                        await progress_callback({
                            "current_generation": gen + 1,
                            "total_generations": self.generations,
                            "is_running": False,
                            "is_stopped": True,
                            "history": self.history,
                            "contexts": self.contexts,
                            "progress": ((self.pop_size + gen * self.pop_size + len(new_population)) / (self.pop_size * (self.generations + 1))) * 100,
                            "stop_message": f"Evolution stopped during generation {gen + 1} (completed {len(new_population)}/{self.pop_size} ideas)",
                            "token_counts": self.get_total_token_count()
                        })
                        return

                    group = self.population[i : i + actual_tournament_size]
                    ranks = self.critic.get_tournament_ranks(group, self.idea_type, self.tournament_comparisons)

                    for idea_idx, rank in sorted(ranks.items(), key=lambda x: x[1]):
                        # Extract title from the idea object within the dictionary
                        idea_obj = group[idea_idx]["idea"]
                        title = idea_obj.title if hasattr(idea_obj, 'title') else "Untitled"
                        print(f"{title} - {rank}")

                    for k in range(len(group)):
                        # Check for stop request during breeding
                        if self.stop_requested:
                            self.is_stopped = True
                            print(f"Stop requested - evolution halted during breeding in generation {gen + 1}")

                            # If we have some new population, add it to history
                            if new_population:
                                self.history.append(new_population)

                            await progress_callback({
                                "current_generation": gen + 1,
                                "total_generations": self.generations,
                                "is_running": False,
                                "is_stopped": True,
                                "history": self.history,
                                "contexts": self.contexts,
                                "progress": ((self.pop_size + gen * self.pop_size + len(new_population)) / (self.pop_size * (self.generations + 1))) * 100,
                                "stop_message": f"Evolution stopped during generation {gen + 1} (completed {len(new_population)}/{self.pop_size} ideas)",
                                "token_counts": self.get_total_token_count()
                            })
                            return

                        # using tournament ranks, weighted sample from group
                        weights = [ranks[i] for i in range(len(group))]
                        # normalize weights to be between 0 and 1
                        # Note: this shift will cause the lowest ranked idea to have a weight of 0, eliminating it from selection
                        weights = [(w - min(weights)) / (max(weights) - min(weights)) + 1e-6 for w in weights]
                        weights = [(w / sum(weights)) for w in weights]


                        # Select parents and breed
                        parent_indices = np.random.choice(list(ranks.keys()), size=self.breeder.parent_count, p=weights, replace=False)
                        parent_ideas = [group[idx] for idx in parent_indices]
                        new_idea = self.breeder.breed(parent_ideas, self.idea_type, self.use_genotype_breeding, self.genotype_encoder)

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

                        # Small delay to allow frontend to process updates and check for stop
                        await asyncio.sleep(0.1)

                # Update population with new ideas
                self.population = new_population
                self.history.append(self.population)
                print(f"Generation {gen + 1} complete. Population size: {len(self.population)}")

            # Mark evolution as complete (only if not stopped)
            if not self.stop_requested:
                await progress_callback({
                    "current_generation": self.generations,
                    "total_generations": self.generations,
                    "is_running": False,
                    "history": self.history,
                    "contexts": self.contexts,
                    "progress": 100,
                    "token_counts": self.get_total_token_count()
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

    def get_ideas_by_generation(self, generation_index: int) -> List[Dict]:
        """
        Get all ideas from a specific generation

        Args:
            generation_index: The index of the generation to retrieve

        Returns:
            List of idea dictionaries with 'id' and 'idea' keys
        """
        if generation_index < 0 or generation_index >= len(self.history):
            return []
        return self.history[generation_index]

    def get_total_token_count(self):
        """Get the total token count from all LLM components with cost calculation"""
        # Get input and output tokens from each component
        ideator_input = getattr(self.ideator, 'input_token_count', 0)
        ideator_output = getattr(self.ideator, 'output_token_count', 0)
        formatter_input = getattr(self.formatter, 'input_token_count', 0)
        formatter_output = getattr(self.formatter, 'output_token_count', 0)
        critic_input = getattr(self.critic, 'input_token_count', 0)
        critic_output = getattr(self.critic, 'output_token_count', 0)
        breeder_input = getattr(self.breeder, 'input_token_count', 0)
        breeder_output = getattr(self.breeder, 'output_token_count', 0)
        genotype_encoder_input = getattr(self.genotype_encoder, 'input_token_count', 0) if self.genotype_encoder else 0
        genotype_encoder_output = getattr(self.genotype_encoder, 'output_token_count', 0) if self.genotype_encoder else 0

        # Calculate totals
        total_input = ideator_input + formatter_input + critic_input + breeder_input + genotype_encoder_input
        total_output = ideator_output + formatter_output + critic_output + breeder_output + genotype_encoder_output
        total = total_input + total_output

        # Get pricing information from config
        from idea.config import model_prices_per_million_tokens

        # Get model names for each agent
        ideator_model = getattr(self.ideator, 'model_name', 'gemini-2.0-flash')
        formatter_model = getattr(self.formatter, 'model_name', 'gemini-2.0-flash')
        critic_model = getattr(self.critic, 'model_name', 'gemini-2.0-flash')
        breeder_model = getattr(self.breeder, 'model_name', 'gemini-2.0-flash')
        genotype_encoder_model = getattr(self.genotype_encoder, 'model_name', 'gemini-2.0-flash') if self.genotype_encoder else None

        # Default pricing if model not found in config
        default_price = {"input": 0.1, "output": 0.4}

        # Get pricing for each model
        ideator_pricing = model_prices_per_million_tokens.get(ideator_model, default_price)
        formatter_pricing = model_prices_per_million_tokens.get(formatter_model, default_price)
        critic_pricing = model_prices_per_million_tokens.get(critic_model, default_price)
        breeder_pricing = model_prices_per_million_tokens.get(breeder_model, default_price)
        genotype_encoder_pricing = model_prices_per_million_tokens.get(genotype_encoder_model, default_price) if genotype_encoder_model else default_price

        # Calculate cost for each component
        ideator_input_cost = (ideator_pricing["input"] * ideator_input) / 1_000_000
        ideator_output_cost = (ideator_pricing["output"] * ideator_output) / 1_000_000
        formatter_input_cost = (formatter_pricing["input"] * formatter_input) / 1_000_000
        formatter_output_cost = (formatter_pricing["output"] * formatter_output) / 1_000_000
        critic_input_cost = (critic_pricing["input"] * critic_input) / 1_000_000
        critic_output_cost = (critic_pricing["output"] * critic_output) / 1_000_000
        breeder_input_cost = (breeder_pricing["input"] * breeder_input) / 1_000_000
        breeder_output_cost = (breeder_pricing["output"] * breeder_output) / 1_000_000
        genotype_encoder_input_cost = (genotype_encoder_pricing["input"] * genotype_encoder_input) / 1_000_000 if self.genotype_encoder else 0
        genotype_encoder_output_cost = (genotype_encoder_pricing["output"] * genotype_encoder_output) / 1_000_000 if self.genotype_encoder else 0

        # Calculate total costs
        total_input_cost = ideator_input_cost + formatter_input_cost + critic_input_cost + breeder_input_cost + genotype_encoder_input_cost
        total_output_cost = ideator_output_cost + formatter_output_cost + critic_output_cost + breeder_output_cost + genotype_encoder_output_cost
        total_cost = total_input_cost + total_output_cost

        token_data = {
            'ideator': {
                'total': self.ideator.total_token_count,
                'input': ideator_input,
                'output': ideator_output,
                'model': ideator_model,
                'cost': ideator_input_cost + ideator_output_cost
            },
            'formatter': {
                'total': self.formatter.total_token_count,
                'input': formatter_input,
                'output': formatter_output,
                'model': formatter_model,
                'cost': formatter_input_cost + formatter_output_cost
            },
            'critic': {
                'total': self.critic.total_token_count,
                'input': critic_input,
                'output': critic_output,
                'model': critic_model,
                'cost': critic_input_cost + critic_output_cost
            },
            'breeder': {
                'total': self.breeder.total_token_count,
                'input': breeder_input,
                'output': breeder_output,
                'model': breeder_model,
                'cost': breeder_input_cost + breeder_output_cost
            },
            'total': total,
            'total_input': total_input,
            'total_output': total_output,
            'cost': {
                'input_cost': total_input_cost,
                'output_cost': total_output_cost,
                'total_cost': total_cost,
                'currency': 'USD'
            },
            'models': {
                'ideator': ideator_model,
                'formatter': formatter_model,
                'critic': critic_model,
                'breeder': breeder_model
            }
        }

        # Add genotype encoder data if enabled
        if self.genotype_encoder:
            token_data['genotype_encoder'] = {
                'total': self.genotype_encoder.total_token_count,
                'input': genotype_encoder_input,
                'output': genotype_encoder_output,
                'model': genotype_encoder_model,
                'cost': genotype_encoder_input_cost + genotype_encoder_output_cost
            }
            token_data['models']['genotype_encoder'] = genotype_encoder_model

        # Calculate estimated total cost for each available model using the
        # overall token counts. This gives users a rough idea of what the
        # evolution would have cost if a different model had been selected.
        from idea.config import LLM_MODELS

        estimates = {}
        for model in LLM_MODELS:
            model_id = model['id']
            model_name = model.get('name', model_id)
            pricing = model_prices_per_million_tokens.get(model_id, default_price)
            est_cost = (
                pricing['input'] * total_input / 1_000_000
                + pricing['output'] * total_output / 1_000_000
            )
            estimates[model_id] = {'name': model_name, 'cost': est_cost}

        token_data['estimates'] = estimates
        return token_data
