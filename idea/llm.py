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
    # TODO: the context prompts often return very similar things. Create 1 large context prompt and split result among initial ideas
    RANDOM_WORDS_PROMPT = """generate a list of 50 randomly choses english words.
    When generating each new reword, review what has come before
    and create a word that is as different as possible from the preceding set.
    Return the list of words as a single string, separated by spaces with no other text."""

    KEY_IDEAS_ELEMENTS_RESEARCH = """ You are a historian of innovation.
    Please give 5 ideas that demonstrated crucial innovation in the field of {field}.
    Write a very short description of each idea, no more than 20 words.
    Finally, extract some of the key elements or concepts that are important to the idea without
    giving away the idea itself. Keep them more general and abstract.
    When finished, collect all the concepts/elements and return them as a comma separated list in the format:
    CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>"""

    KEY_IDEAS_ELEMENTS_GAME_DESIGN = """ You are a uniquely creative game designer.
    Please give 5 simple game designs that demonstrated crucial innovation in the field of {field}.
    Write a very short description of each game design, no more than 20 words.
    Finally, extract some of the key elements or concepts that are important to the game design without
    giving away the game design itself. Keep them more general and abstract.
    When finished, collect all the concepts/elements and return them as a comma separated list in the format:
    CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>"""

    KEY_IDEAS_ELEMENTS_DRABBLE = """ You are a uniquely creative writer.
    Please list 5 impactful short stories that had a lasting impact on the field of {field}.
    Write a very short description of each story, no more than 20 words.
    Finally, extract some of the key elements or concepts that are important to the story without
    giving away the story itself. Keep them more general and abstract.
    When finished, collect all the concepts/elements and return them as a comma separated list in the format:
    CONCEPTS:<concept1>, <concept2>, <concept3>, <concept4>, <concept5>"""

    AI_RESEARCH_PROMPT = """The above is some context to get you inspired.
    Using some or none of it for inspiration,
    suggest a promising area for novel research in AI (it can be a big idea, or a small idea, in any subdomain).
    keep it concise and to the point but with enough detail for a researcher to critique it.
    The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate"""

    GAME_DESIGN_PROMPT = """You are a uniquely creative game designer.
    The above is some context to get you inspired.
    Using some or none of it for inspiration,
    suggest a new game design that is both interesting and fun to play.
    With key game mechanics and controls.
    The game should be simple enough that it can be implemented as a browser game without relying on lots of external assets.
    The game complexity should similar in level to classic games like Breakout, Snake, Tetris, Pong, Pac-Man, Space Invaders, Frogger, Minesweeper, Tic Tac Toe, 2048, Wordle etc.
    Avoid being derivative of these but limit the ambition to the level of complexity of these games.
    You should format your idea with enough specificity that a developer can implement it.
     The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate"""

    DRABBLE_PROMPT = """Write a drabble - a complete story in exactly 100 words.

    A drabble is a short work of fiction exactly 100 words in length (not including the title).
    Despite its brevity, a drabble should tell a complete story with a beginning, middle, and end.

    Your story will be judge on creativity, originality, and the ability to tell a complete story in 100 words.

    Format your response as:
    Title: [Your creative title]
    Proposal: [Your 100-word story]
    """


    def __init__(self, **kwargs):
        self.idea_field_map = {
            "airesearch": "AI Research",
            "game_design": "2D arcade game design",
            "drabble": "Creative writing"
        }
        super().__init__(agent_name=self.agent_name, **kwargs)

    def generate_context(self, method: str, field: str = None) -> str:
        """Generate initial context for ideation"""
        if method == "random_words":
            return self.generate_text(self.RANDOM_WORDS_PROMPT, temperature=1.0)
        elif method == "key_ideas_elements":
            if field == "AI Research":
                text = self.generate_text(self.KEY_IDEAS_ELEMENTS_RESEARCH.format(field=field), temperature=1.0)
            elif field == "2D arcade game design":
                text = self.generate_text(self.KEY_IDEAS_ELEMENTS_GAME_DESIGN.format(field=field), temperature=1.0)
            elif field == "Creative writing":
                text = self.generate_text(self.KEY_IDEAS_ELEMENTS_DRABBLE.format(field=field), temperature=1.0)
            else:
                raise ValueError(f"Invalid field: {field}")
            return text.split("CONCEPTS:")[1].strip()
        raise ValueError(f"Invalid method: {method}")

    def get_idea_prompt(self, idea_type: str) -> str:
        """Get prompt template for specific idea type"""
        if idea_type == "airesearch":
            return self.AI_RESEARCH_PROMPT
        elif idea_type == "game_design":
            return self.GAME_DESIGN_PROMPT
        elif idea_type == "drabble":
            return self.DRABBLE_PROMPT
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
        # TODO: fix this for drabble
        """Generate a new idea based on existing ones"""
        idea_str = "\n".join([f"{i+1}. {idea}" for i, idea in enumerate(ideas)])
        prompt = f"""You are an experienced researcher and you are given a list of proposals.
        {idea_str}
        Considering the above ideas please propose a new idea, that could be completely new
        and different from the ideas above or could combine ideas to create a new idea.
        Please avoid minor refinements of the ideas above, and instead propose a new idea
        that is a significant departure.
        The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate"""
        return self.generate_text(prompt, temperature=1.0)

class Formatter(LLMWrapper):
    """Reformats unstructured ideas into a cleaner format"""
    agent_name = "Formatter"
    DEFAULT_PROMPT = """Take the following idea and rewrite it in a clear,
    structured format. The body of the proposal should be written in markdown syntax with headings, paragraphs, bullet lists as appropriate: {input_text}"""

    DRABBLE_PROMPT = """Take the following idea and rewrite it in a clear,
    structured format. With a title denoted by [Title:], and the story proposal denoted by [Proposal:]: {input_text}"""

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, prompt_template=self.DEFAULT_PROMPT, temperature=0.3, **kwargs)

    def format_idea(self, raw_idea: str, idea_type: str) -> str:
        if idea_type == "drabble":
            print(f"Formatting drabble: {raw_idea}")
            prompt = self.DRABBLE_PROMPT.format(input_text=raw_idea)
        else:
            prompt = self.prompt_template.format(input_text=raw_idea)
        response = self.generate_text(prompt, response_schema=Idea)
        idea = Idea(**json.loads(response))
        return idea

class Critic(LLMWrapper):
    """Analyzes and refines ideas"""
    agent_name = "Critic"

    CRITIQUE_PROMPT = """You are a helpful AI that refines ideas.
    Current Idea:
    {idea}

    Please consider the above proposal. Offer critical feedback.
    Pointing out potential pitfalls as well as strengths. If the idea doesn't have a clear structure,
    or sufficient detail, please suggest how to improve it and ask for elaboration of specific points of interest.
    No additional text, just the critique."""

    REFINE_PROMPT = """Current Idea:
    {idea}

    Critique: {critique}

    Please review both, consider your own opinion and create your own proposal.
    This could be a refinement of the original proposal or a fresh take on it.
    No additional text, just the refined idea on its own.
    Try have a sensible structure to the idea with markdown headings.
    It should be sufficient detailed to convey main idea to someone in the field."""

    def __init__(self, **kwargs):
        super().__init__(agent_name=self.agent_name, temperature=0.4, **kwargs)

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
        Please review the proposals and give a once sentence pro and con for each.
        If a proposal is unsufficiently detailed or lacks a clear structure this should count against it.
        After this, please give the proposal that you think is the worst considering value, novelty, and feasibility.
        The proposals are:
        {idea_str}
        Please return the proposal that you think is the worst in the following format (with no other text following):
        Worst Proposal: <proposal number>"""

        result = self.generate_text(prompt)
        parsed_result = result.split("Worst Proposal:")[1].strip()
        try:
            idea_index = int(parsed_result) - 1
        except ValueError:
            print(f"Invalid proposal index: {parsed_result}")
            idea_index = np.random.randint(0, len(ideas))
        return [ideas[i] for i in range(len(ideas)) if i != idea_index]

    def compare_ideas(self, idea_a, idea_b, idea_type: str):
        if idea_type == "drabble":
            item_type = "stories"
            criteria = ["creativity", "originality", "completeness", "impact"]
        elif idea_type == "game_design":
            item_type = "game designs"
            criteria = ["originality and creativity", "simplicity", "fun factor", "feasability for simple standard browser implementation"]
        elif idea_type == "airesearch":
            item_type = "AI research proposals"
            criteria = ["originality and novelty", "potential impact and significance", "feasibility and practicality", "clarity and coherence"]
        else:
            raise ValueError(f"Invalid idea type: {idea_type}")
        f"""
        Compare two {item_type} using the LLM and determine which is better.
        Returns: "A", "B", "tie", or None if there was an error
        """
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