import google.generativeai as genai
import json
from pydantic import BaseModel
from typing import Type

from idea.models import Idea


class LLMWrapper:
    MAX_TOKENS = 8192
    """
    Abstracts away the details of talking to an LLM. Right now uses Google Generative AI
    but can be swapped for OpenAI, Anthropic, local models, etc.
    """

    def __init__(self, provider: str = "google_generative_ai", model_name: str = "gemini-1.5-flash"):
        self.provider = provider
        self.model_name = model_name

        if provider == "google_generative_ai":
            if genai is None:
                raise ImportError("google.generativeai is not installed or not importable.")
            # Note: user must set environment variable: PALM_API_KEY
            genai.configure(api_key=None)  # if None, it picks from environment
        else:
            # In principle, you could configure other providers here
            pass

    def generate_text(self, prompt: str, temperature: float = 0.7, max_tokens: int = MAX_TOKENS, response_schema: Type[BaseModel] = None) -> str:
        """Call the underlying LLM to generate text based on the prompt."""
        if self.provider == "google_generative_ai":
            generation_config = {
                "temperature": temperature,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": max_tokens,
            }

            if response_schema:
                generation_config["response_schema"] = response_schema
                generation_config["response_mime_type"] = "application/json"

            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
            )

            response = model.generate_content(
                prompt,
                generation_config=generation_config,
            )
            return response.text if response.text else "No response."
        else:
            # Implement other providers here
            return "Not implemented"

    def generate_idea(self, prompt: str, temperature: float = 0.7, max_tokens: int = MAX_TOKENS) -> Idea:
        """Generate an idea."""
        response = self.generate_text(prompt, temperature, max_tokens, Idea)
        try:
            data = json.loads(response)
            return Idea(**data) if data else "No response."
        except Exception as e:
            print(f"Error parsing idea: {e}")
            print(response)
            return Idea(title="", proposal=response)

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