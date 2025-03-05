from abc import ABC
import random
import google.generativeai as genai
import json
from pydantic import BaseModel
from typing import Type, Optional, Dict, Any, List
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from idea.models import Idea
from idea.prompts.loader import get_prompts, get_field_name

class LLMWrapper(ABC):
    """Base class for LLM interactions"""
    MAX_TOKENS = 8192

    def __init__(self,
                 provider: str = "google_generative_ai",
                 model_name: str = "gemini-1.5-flash",
                 prompt_template: str = "",
                 temperature: float = 0.7,
                 agent_name: str = ""):
        self.provider = provider
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.total_token_count = 0
        self.agent_name = agent_name
        self._setup_provider()

    def _setup_provider(self):
        if self.provider == "google_generative_ai":
            if genai is None:
                raise ImportError("google.generativeai is not installed")
            genai.configure(api_key=None)
        # Add other providers here

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=30, max=300),
        reraise=True
    )
    def generate_text(self,
                     prompt: str,
                     temperature: Optional[float] = None,
                     response_schema: Type[BaseModel] = None) -> str:
        """Base method for generating text with retry logic built in"""
        if self.provider == "google_generative_ai":
            config = self._get_generation_config(temperature, response_schema)
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=config
            )
            response = model.generate_content(prompt, generation_config=config)
            self.total_token_count += response.usage_metadata.total_token_count
            print(f"Total tokens {self.agent_name}: {self.total_token_count}")
            return response.text if response.text else "No response."
        return "Not implemented"

    def _get_generation_config(self,
                             temperature: Optional[float],
                             response_schema: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        config = {
            "temperature": temperature or self.temperature,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": self.MAX_TOKENS,
        }
        if response_schema:
            config["response_schema"] = response_schema
            config["response_mime_type"] = "application/json"
        return config

class Ideator(LLMWrapper):
    """Generates and manages ideas"""
    agent_name = "Ideator"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, **kwargs)

    def generate_context(self, method: str, idea_type: str) -> str:
        """Generate initial context for ideation"""
        prompts = get_prompts(idea_type)
        field = get_field_name(idea_type)

        if method == "random_words":
            context_text = self.generate_text(prompts.RANDOM_WORDS_PROMPT, temperature=2.0)
        elif method == "key_ideas_elements":
            text = self.generate_text(prompts.KEY_IDEAS_ELEMENTS_PROMPT.format(field=field), temperature=2.0)
            print(f"Text: {text}")
            context_text = text.split("CONCEPTS:")[1].strip()
        else:
            raise ValueError(f"Invalid method: {method}")
        #sample 10% of the words
        words = [word.strip() for word in context_text.split(',')]
        return ", ".join(random.sample(words, max(1, int(len(words) * 0.1))))

    def get_idea_prompt(self, idea_type: str) -> str:
        """Get prompt template for specific idea type"""
        prompts = get_prompts(idea_type)
        return prompts.IDEA_PROMPT

    def get_new_idea_prompt(self, idea_type: str) -> str:
        """Get prompt template for generating a new idea"""
        prompts = get_prompts(idea_type)
        return prompts.NEW_IDEA_PROMPT

    def seed_ideas(self, n: int, context_type: str, idea_type: str) -> List[str]:
        """Generate n initial ideas"""
        idea_prompt = self.get_idea_prompt(idea_type)

        ideas = []

        for _ in tqdm(range(n), desc="Generating ideas"):
            context = self.generate_context(context_type, idea_type)
            print(f"Context: {context}")
            prompt = f"{context}\nInstruction: {idea_prompt}"
            response = self.generate_text(prompt, temperature=1.5)
            ideas.append(response)
        return ideas

    def generate_new_idea(self, ideas: List[str], idea_type: str) -> str:
        """Generate a new idea based on existing ones"""
        current_ideas = "\n\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
        prompt = self.get_new_idea_prompt(idea_type).format(current_ideas=current_ideas)
        prompt = f"current ideas:\n{current_ideas}\n\n{prompt}"
        return self.generate_text(prompt, temperature=1.0)

class Formatter(LLMWrapper):
    """Reformats unstructured ideas into a cleaner format"""
    agent_name = "Formatter"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, temperature=0.3, **kwargs)

    def format_idea(self, raw_idea: str, idea_type: str) -> str:
        prompts = get_prompts(idea_type)
        prompt = prompts.FORMAT_PROMPT.format(input_text=raw_idea)
        response = self.generate_text(prompt, response_schema=Idea)
        idea = Idea(**json.loads(response))
        return idea

class Critic(LLMWrapper):
    """Analyzes and refines ideas"""
    agent_name = "Critic"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, temperature=0.7, **kwargs)

    def critique(self, idea: str, idea_type: str) -> str:
        """Provide critique for an idea"""
        prompts = get_prompts(idea_type)
        prompt = prompts.CRITIQUE_PROMPT.format(idea=idea)
        return self.generate_text(prompt)

    def refine(self, idea: str, idea_type: str) -> str:
        """Refine an idea based on critique"""
        critique = self.critique(idea, idea_type)
        prompts = get_prompts(idea_type)
        prompt = prompts.REFINE_PROMPT.format(
            idea=idea,
            critique=critique
        )
        return self.generate_text(prompt)

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

    def get_tournament_ranks(self, ideas: List[str], idea_type: str, comparisons: int) -> dict:
        """Get the tournament rank of an idea"""

        ranks = {i: 1500 for i in range(len(ideas))}
        for k in range(comparisons):
            if k < len(ideas):
                # assert that each idea is one side of the comparison at least once
                idea_idx_a = k
                # Create a list of valid indices excluding idea_idx_a
                valid_indices = [idx for idx in range(len(ideas)) if idx != idea_idx_a]
                idea_idx_b = np.random.choice(valid_indices, size=1)[0]
            else:
                idea_idx_a, idea_idx_b = np.random.choice(len(ideas), size=2, replace=False)
            winner = self.compare_ideas(ideas[idea_idx_a].dict(), ideas[idea_idx_b].dict(), idea_type)
            elo_a, elo_b = self._elo_update(ranks[idea_idx_a], ranks[idea_idx_b], winner)
            ranks[idea_idx_a] = elo_a
            ranks[idea_idx_b] = elo_b

        return ranks

    def remove_worst_idea(self, ideas: List[str], idea_type: str) -> List[str]:
        """Identify and remove the worst idea from a list"""
        idea_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
        prompts = get_prompts(idea_type)
        prompt = prompts.REMOVE_WORST_IDEA_PROMPT.format(
            ideas=idea_str,
            criteria=", ".join(prompts.COMPARISON_CRITERIA)
        )
        result = self.generate_text(prompt)
        parsed_result = result.split("Worst Entry:")[1].strip()
        try:
            idea_index = int(parsed_result) - 1
        except ValueError:
            print(f"Invalid proposal index: {parsed_result}")
            idea_index = np.random.randint(0, len(ideas))
        return [ideas[i] for i in range(len(ideas)) if i != idea_index]

    def compare_ideas(self, idea_a, idea_b, idea_type: str):
        """
        Compare two ideas using the LLM and determine which is better.
        Returns: "A", "B", "tie", or None if there was an error
        """
        prompts = get_prompts(idea_type)

        # Get the item type from the prompt configuration
        item_type = getattr(prompts, "ITEM_TYPE", "ideas")
        criteria = prompts.COMPARISON_CRITERIA

        prompt = f"""You are an expert evaluator of {item_type}. You will be presented with two {item_type}, and your task is to determine which one is better.

        Idea A:
        Title: {idea_a.get('title', 'Untitled')}
        {idea_a.get('proposal', '')}

        Idea B:
        Title: {idea_b.get('title', 'Untitled')}
        {idea_b.get('proposal', '')}

        Evaluate both ideas based on the following criteria:
        {", ".join(criteria)}

        Criterion 1 is the most important.

        After your evaluation, respond with exactly one of these three options:
        - "Result: A" if Idea A is better
        - "Result: B" if Idea B is better
        - "Result: tie" if both ideas are approximately equal in quality

        Your response must contain exactly one of these three phrases and nothing else.
        """

        try:
            response = self.generate_text(prompt, temperature=0.3)
            result = response.strip().upper()

            print(f"LLM comparison response: {result}")

            # More robust parsing
            if "RESULT: A" in result:
                return "A"
            elif "RESULT: B" in result:
                return "B"
            elif "RESULT: TIE" in result:
                return "tie"
            elif "A" in result and "B" not in result:
                return "A"
            elif "B" in result and "A" not in result:
                return "B"
            else:
                return "tie"
        except Exception as e:
            print(f"Error in compare_ideas: {e}")
            return None  # Return None instead of "tie" on error

if __name__ == "__main__":
    import os
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    # Create the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
    }

    model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest",
    generation_config=generation_config,
    )

    response = model.generate_content(
        "write a proposal for a new AI research project",
        generation_config=generation_config,
    )

    print(response.json())


class Breeder(LLMWrapper):
    """Breeds ideas"""
    agent_name = "Breeder"
    parent_count = 2

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, temperature=1.0, **kwargs)

    def breed(self, ideas: List[str], idea_type: str) -> str:
        """Breed two ideas"""
        prompt = self.get_breed_prompt(idea_type)
        return self.generate_text(prompt)

    def get_breed_prompt(self, idea_type: str) -> str:
        """Get prompt template for breeding ideas"""
        prompts = get_prompts(idea_type)
        return prompts.BREED_PROMPT

    # TODO: and a method to convert idea Phenotype to Genotype (basic components used in breeding)
