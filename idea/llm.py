from abc import ABC
import google.generativeai as genai
import json
from pydantic import BaseModel
from typing import Type, Optional, Dict, Any, List
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import numpy as np
from idea.models import Idea

class LLMWrapper(ABC):
    """Base class for LLM interactions"""
    MAX_TOKENS = 8192

    def __init__(self,
                 provider: str = "google_generative_ai",
                 model_name: str = "gemini-1.5-flash",
                 prompt_template: str = "",
                 temperature: float = 0.7):
        self.provider = provider
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.temperature = temperature
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
    RANDOM_WORDS_PROMPT = """generate a list of 50 randomly choses english words.
    When generating each new reword, review what has come before
    and create a word that is as different as possible from the preceding set.
    Return the list of words as a single string, separated by spaces with no other text."""

    KEY_IDEAS_ELEMENTS = """ You are a historian of innovation.
    Please give 5 ideas that demonstrated crucial innovation in the field of {field}.
    Write a very short description of each idea, no more than 20 words.
    Finally, extract some of the key elements or concepts that are important to the idea without
    giving away the idea itself. Keep them more general and abstract.
    When finished, collect all the concepts/elements and return them as a comma separated list in the format:
    CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>"""

    AI_RESEARCH_PROMPT = """The above is some context to get you inspired.
    Using some or none of it for inspiration,
    suggest a promising area for novel research in AI (it can be a big idea, or a small idea, in any subdomain).
    keep it concise and to the point but with enough detail for a researcher to critique it."""

    GAME_DESIGN_PROMPT = """You are a uniquely creative game designer.
    The above is some context to get you inspired.
    Using some or none of it for inspiration,
    suggest a new game design that is both interesting and fun to play.
    With key game mechanics and controls.
    The game should be simple enough that it can be implemented as a browser game without relying on lots of external assets.
    You should format your idea with enough specificity that a developer can implement it."""

    def __init__(self, **kwargs):
        self.idea_field_map = {
            "airesearch": "AI Research",
            "game_design": "2D arcade game design"
        }
        super().__init__(**kwargs)

    def generate_context(self, method: str, field: str = None) -> str:
        """Generate initial context for ideation"""
        if method == "random_words":
            return self.generate_text(self.RANDOM_WORDS_PROMPT, temperature=1.0)
        elif method == "key_ideas_elements":
            text = self.generate_text(self.KEY_IDEAS_ELEMENTS.format(field=field), temperature=1.0)
            return text.split("CONCEPTS:")[1].strip()
        raise ValueError(f"Invalid method: {method}")

    def get_idea_prompt(self, idea_type: str) -> str:
        """Get prompt template for specific idea type"""
        if idea_type == "airesearch":
            return self.AI_RESEARCH_PROMPT
        elif idea_type == "game_design":
            return self.GAME_DESIGN_PROMPT
        raise ValueError(f"Invalid idea type: {idea_type}")

    def seed_ideas(self, n: int, context_type: str, idea_type: str) -> List[str]:
        """Generate n initial ideas"""
        idea_prompt = self.get_idea_prompt(idea_type)

        ideas = []

        for _ in tqdm(range(n), desc="Generating ideas"):
            context = self.generate_context(context_type, self.idea_field_map[idea_type])
            print(f"Context: {context}")
            prompt = f"{context}\nInstruction: {idea_prompt}"
            response = self.generate_text(prompt, temperature=1.0)
            ideas.append(response)
        return ideas

    def generate_new_idea(self, ideas: List[str]) -> str:
        """Generate a new idea based on existing ones"""
        idea_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
        prompt = f"""You are an experienced researcher and you are given a list of proposals.
        {idea_str}
        Considering the above ideas please propose a new idea, that could be completely new
        and different from the ideas above or could combine ideas to create a new idea.
        Please avoid minor refinements of the ideas above, and instead propose a new idea
        that is a significant departure."""
        return self.generate_text(prompt, temperature=1.0)

class Formatter(LLMWrapper):
    """Reformats unstructured ideas into a cleaner format"""
    DEFAULT_PROMPT = """Take the following idea and rewrite it in a clear,
    structured format: {input_text}"""

    def __init__(self, **kwargs):
        super().__init__(prompt_template=self.DEFAULT_PROMPT, temperature=0.3, **kwargs)

    def format_idea(self, raw_idea: str) -> str:
        prompt = self.prompt_template.format(input_text=raw_idea)
        response = self.generate_text(prompt, response_schema=Idea)
        idea = Idea(**json.loads(response))
        return idea

class Critic(LLMWrapper):
    """Analyzes and refines ideas"""
    CRITIQUE_PROMPT = """You are a helpful AI that refines ideas.
    Current Idea:
    {idea}

    Please consider the above proposal. Offer critical feedback.
    Pointing out potential pitfalls as well as strengths.
    No additional text, just the critique."""

    REFINE_PROMPT = """Current Idea:
    {idea}

    Critique: {critique}

    Please review both, consider your own opinion and create your own proposal.
    This could be a refinement of the original proposal or a fresh take on it.
    No additional text, just the refined idea on its own."""

    def __init__(self, **kwargs):
        super().__init__(temperature=0.4, **kwargs)

    def critique(self, idea: str) -> str:
        """Provide critique for an idea"""
        prompt = self.CRITIQUE_PROMPT.format(idea=idea)
        return self.generate_text(prompt)

    def refine(self, idea: str) -> str:
        """Refine an idea based on critique"""
        critique = self.critique(idea)
        prompt = self.REFINE_PROMPT.format(
            idea=idea,
            critique=critique
        )
        return self.generate_text(prompt)

    def remove_worst_idea(self, ideas: List[str]) -> List[str]:
        """Identify and remove the worst idea from a list"""
        idea_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
        prompt = f"""You are an experienced reviewer and you are given a list of proposals.
        Please review the ideas and give a once sentence pro and con for each.
        After this, please give the idea that you think is the worst considering value, novelty, and feasibility.
        The ideas are:
        {idea_str}
        Please return the idea that you think is the worst in the following format (with no other text following):
        Worst Idea: <idea number>"""

        result = self.generate_text(prompt)
        parsed_result = result.split("Worst Idea:")[1].strip()
        try:
            idea_index = int(parsed_result) - 1
        except ValueError:
            print(f"Invalid idea index: {parsed_result}")
            idea_index = np.random.randint(0, len(ideas))
        return [ideas[i] for i in range(len(ideas)) if i != idea_index]

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