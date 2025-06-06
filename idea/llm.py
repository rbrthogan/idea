from abc import ABC
import random
import os
import google.generativeai as genai
import json
from pydantic import BaseModel
from typing import Type, Optional, Dict, Any, List
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from idea.models import Idea
from idea.prompts.loader import get_prompts
import uuid
class LLMWrapper(ABC):
    """Base class for LLM interactions"""
    MAX_TOKENS = 8192

    def __init__(self,
                 provider: str = "google_generative_ai",
                 model_name: str = "gemini-2.0-flash",
                 prompt_template: str = "",
                 temperature: float = 0.7,
                 agent_name: str = ""):
        self.provider = provider
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
        self.total_token_count = 0
        self.input_token_count = 0
        self.output_token_count = 0
        self.agent_name = agent_name
        print(f"Initializing {agent_name or 'LLM'} with temperature: {temperature}")
        self._setup_provider()

    def _setup_provider(self):
        if self.provider == "google_generative_ai":
            if genai is None:
                raise ImportError("google.generativeai is not installed")
            api_key = os.environ.get("GEMINI_API_KEY")
            genai.configure(api_key=api_key)
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

            # Track total tokens
            self.total_token_count += response.usage_metadata.total_token_count

            # Try to get input and output tokens if available
            try:
                if hasattr(response.usage_metadata, 'prompt_token_count'):
                    self.input_token_count += response.usage_metadata.prompt_token_count
                if hasattr(response.usage_metadata, 'candidates_token_count'):
                    self.output_token_count += response.usage_metadata.candidates_token_count
            except AttributeError:
                # If detailed token counts aren't available, estimate based on total
                # Assuming a typical 1:4 ratio of input:output tokens
                self.input_token_count += int(response.usage_metadata.total_token_count * 0.2)
                self.output_token_count += int(response.usage_metadata.total_token_count * 0.8)

            print(f"Total tokens {self.agent_name}: {self.total_token_count} (Input: {self.input_token_count}, Output: {self.output_token_count})")
            return response.text if response.text else "No response."
        return "Not implemented"

    def _get_generation_config(self,
                             temperature: Optional[float],
                             response_schema: Optional[Type[BaseModel]] = None) -> Dict[str, Any]:
        actual_temp = temperature if temperature is not None else self.temperature
        print(f"{self.agent_name} using temperature: {actual_temp} (override: {temperature}, default: {self.temperature})")

        config = {
            "temperature": actual_temp,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": self.MAX_TOKENS,
        }
        if response_schema:
            config["response_schema"] = response_schema
            config["response_mime_type"] = "application/json"
        print(f"Generation config: {config}")
        return config

class Ideator(LLMWrapper):
    """Generates and manages ideas"""
    agent_name = "Ideator"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, **kwargs)

    def generate_context(self, idea_type: str) -> str:
        """Generate initial context for ideation"""
        prompts = get_prompts(idea_type)

        # Always use the CONTEXT_PROMPT regardless of method
        text = self.generate_text(prompts.CONTEXT_PROMPT, temperature=2.0)
        print(f"Text: {text}")

        # Extract concepts from the response
        if "CONCEPTS:" in text:
            context_text = text.split("CONCEPTS:")[1].strip()
        else:
            # Fallback if the expected format is not found
            context_text = text.strip()

        # Sample 10% of the words
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

    def seed_ideas(self, n: int, idea_type: str) -> List[str]:
        """Generate n initial ideas"""
        idea_prompt = self.get_idea_prompt(idea_type)

        ideas = []

        for _ in tqdm(range(n), desc="Generating ideas"):
            # Generate context using the context_prompt method regardless of context_type parameter
            context = self.generate_context(idea_type)
            print(f"Context: {context}")
            prompt = f"{context}\nInstruction: {idea_prompt}"
            response = self.generate_text(prompt, temperature=1.5)
            ideas.append({"id": uuid.uuid4(), "idea": response, "parent_ids": []})
        return ideas


class Formatter(LLMWrapper):
    """Reformats unstructured ideas into a cleaner format"""
    agent_name = "Formatter"

    def __init__(self, **kwargs):
        # Use a default temperature only if not provided in kwargs
        temp = kwargs.pop('temperature', 1.0)
        super().__init__(agent_name=self.agent_name, temperature=temp, **kwargs)

    def format_idea(self, raw_idea: str, idea_type: str) -> str:
        """Format a raw idea into a structured format"""
        prompts = get_prompts(idea_type)
        # If raw_idea is a dictionary with 'id' and 'idea' keys, extract just the idea text
        idea_text = raw_idea["idea"] if isinstance(raw_idea, dict) and "idea" in raw_idea else raw_idea
        print(f"Formatting idea:\n {idea_text}")
        prompt = prompts.FORMAT_PROMPT.format(input_text=idea_text)
        print(f"Prompt:\n {prompt}")
        response = self.generate_text(prompt, response_schema=Idea)
        formatted_idea = Idea(**json.loads(response))

        # If the input was a dictionary with an ID, preserve that ID and parent_ids
        if isinstance(raw_idea, dict) and "id" in raw_idea:
            result = {"id": raw_idea["id"], "idea": formatted_idea}
            # Preserve parent_ids if they exist
            if "parent_ids" in raw_idea:
                result["parent_ids"] = raw_idea["parent_ids"]
            else:
                result["parent_ids"] = []
            return result
        return formatted_idea

class Critic(LLMWrapper):
    """Analyzes and refines ideas"""
    agent_name = "Critic"

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, **kwargs)

    def critique(self, idea: str, idea_type: str) -> str:
        """Provide critique for an idea"""
        prompts = get_prompts(idea_type)
        # Extract idea text if it's a dictionary
        idea_text = idea["idea"] if isinstance(idea, dict) and "idea" in idea else idea
        prompt = prompts.CRITIQUE_PROMPT.format(idea=idea_text)
        return self.generate_text(prompt)

    def refine(self, idea: str, idea_type: str) -> str:
        """Refine an idea based on critique"""
        # Extract idea text if it's a dictionary
        idea_text = idea["idea"] if isinstance(idea, dict) and "idea" in idea else idea
        critique = self.critique(idea_text, idea_type)
        prompts = get_prompts(idea_type)
        prompt = prompts.REFINE_PROMPT.format(
            idea=idea_text,
            critique=critique
        )
        refined_idea = self.generate_text(prompt)

        # If the input was a dictionary with an ID, preserve that ID and parent_ids
        if isinstance(idea, dict) and "id" in idea:
            result = {"id": idea["id"], "idea": refined_idea}
            # Preserve parent_ids if they exist
            if "parent_ids" in idea:
                result["parent_ids"] = idea["parent_ids"]
            else:
                result["parent_ids"] = []
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

    def get_tournament_ranks(self, ideas: List[str], idea_type: str, comparisons: int) -> dict:
        """Get the tournament rank of an idea"""

        max_elo_diff = 100
        ranks = {i: 1500 for i in range(len(ideas))}
        for k in range(comparisons):
            if k < len(ideas):
                # assert that each idea is one side of the comparison at least once
                idea_idx_a = k
                # Create a list of valid indices excluding idea_idx_a and ensure the difference is within max_elo_diff
                other_indices = [idx for idx in range(len(ideas)) if idx != idea_idx_a]
                valid_indices = [idx for idx in other_indices if abs(ranks[idx] - ranks[idea_idx_a]) <= max_elo_diff]
                if len(valid_indices) == 0:
                    idea_idx_b = np.random.choice(other_indices, size=1)[0]
                else:
                    idea_idx_b = np.random.choice(valid_indices, size=1)[0]
            else:
                idea_idx_a, idea_idx_b = np.random.choice(len(ideas), size=2, replace=False)

            # Extract idea objects for comparison
            idea_a = ideas[idea_idx_a]
            idea_b = ideas[idea_idx_b]

            # If ideas are dictionaries with 'idea' key, extract the idea objects
            idea_a_obj = idea_a["idea"] if isinstance(idea_a, dict) and "idea" in idea_a else idea_a
            idea_b_obj = idea_b["idea"] if isinstance(idea_b, dict) and "idea" in idea_b else idea_b

            # Convert to dict if not already
            idea_a_dict = idea_a_obj.dict() if hasattr(idea_a_obj, 'dict') else idea_a_obj
            idea_b_dict = idea_b_obj.dict() if hasattr(idea_b_obj, 'dict') else idea_b_obj

            winner = self.compare_ideas(idea_a_dict, idea_b_dict, idea_type)
            elo_a, elo_b = self._elo_update(ranks[idea_idx_a], ranks[idea_idx_b], winner)
            ranks[idea_idx_a] = elo_a
            ranks[idea_idx_b] = elo_b

        return ranks


    def compare_ideas(self, idea_a, idea_b, idea_type: str):
        """
        Compare two ideas using the LLM and determine which is better.
        Returns: "A", "B", "tie", or None if there was an error
        """
        prompts = get_prompts(idea_type)

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

# def Oracle(LLMWrapper):
#     """Oracle for the evolution"""
#     agent_name = "Oracle"

#     def __init__(self, **kwargs):
#         super().__init__(agent_name=self.agent_name, temperature=0.3, **kwargs)


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
        # Don't set temperature directly here, let it come from kwargs
        super().__init__(agent_name=self.agent_name, **kwargs)

    def breed(self, ideas: List[str], idea_type: str) -> str:
        """Breed ideas to create a new idea

        Args:
            ideas: List of parent ideas that have already been selected in the main evolution loop
            idea_type: Type of idea to breed

        Returns:
            A new idea with a unique ID and parent IDs
        """
        prompts = get_prompts(idea_type)

        # Extract idea texts from parent ideas and collect parent IDs
        parent_texts = []
        parent_ids = []

        for parent in ideas:
            # Extract parent ID
            if isinstance(parent, dict) and "id" in parent:
                parent_ids.append(str(parent["id"]))

            # Extract parent text
            if isinstance(parent, dict) and "idea" in parent:
                parent_obj = parent["idea"]
                if hasattr(parent_obj, 'title') and hasattr(parent_obj, 'content'):
                    parent_texts.append(f"{parent_obj.title}: {parent_obj.content}")
                else:
                    parent_texts.append(str(parent_obj))
            else:
                parent_texts.append(str(parent))

        # Format the parent ideas for the prompt
        parent_str = "\n\n".join([f"Parent {i+1}:\n{text}" for i, text in enumerate(parent_texts)])

        # Create the breeding prompt
        prompt = prompts.BREED_PROMPT.format(ideas=parent_str)

        # Generate the new idea
        response = self.generate_text(prompt)

        # Create a new idea with a unique ID and parent IDs
        return {"id": uuid.uuid4(), "idea": response, "parent_ids": parent_ids}

    # TODO: and a method to convert idea Phenotype to Genotype (basic components used in breeding)
