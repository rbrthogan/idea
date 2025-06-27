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

        # Use the configured temperature unless explicitly overridden
        text = self.generate_text(prompts.CONTEXT_PROMPT)
        print(f"Text: {text}")

        # Extract concepts from the response
        if "CONCEPTS:" in text:
            context_text = text.split("CONCEPTS:")[1].strip()
        else:
            # Fallback if the expected format is not found
            context_text = text.strip()

        # Use more of the context - sample 30-50% instead of 10%
        words = [word.strip() for word in context_text.split(',')]
        # Use between 30-50% of the words to maintain diversity while avoiding overwhelming context
        sample_size = max(3, min(int(len(words) * 0.4), 15))  # 40% but cap at 15 words
        subset = random.sample(words, sample_size)
        return ", ".join(subset)

    def generate_specific_prompt(self, context_pool: str, idea_type: str) -> str:
        """Generate a specific idea prompt from the context pool using the translation layer"""
        prompts = get_prompts(idea_type)

        # Create the translation prompt with shuffled subset
        translation_prompt = prompts.SPECIFIC_PROMPT.format(context_pool=context_pool)

        specific_prompt = self.generate_text(translation_prompt)
        print(f"Generated specific prompt: {specific_prompt}")

        return specific_prompt

    def generate_context_from_parents(self, parent_genotypes: List[str]) -> str:
        """Generate context pool from parent genotypes by combining and sampling concepts

        Args:
            parent_genotypes: List of genotypes from parent ideas

        Returns:
            A sampled context pool string
        """
        # Combine all concepts from parent genotypes
        all_concepts = []
        for genotype in parent_genotypes:
            # Split genotype by semicolons and clean up
            concepts = [concept.strip() for concept in genotype.split(';') if concept.strip()]
            all_concepts.extend(concepts)

        # Remove duplicates while preserving order
        unique_concepts = list(dict.fromkeys(all_concepts))

        # Sample 50% at random as requested by user
        if len(unique_concepts) > 1:
            sample_size = max(1, len(unique_concepts) // 2)  # 50% but at least 1
            sampled_concepts = random.sample(unique_concepts, sample_size)
        else:
            sampled_concepts = unique_concepts

        context_pool = ", ".join(sampled_concepts)
        print(f"Generated context from parents: {context_pool}")
        return context_pool

    def generate_idea_from_context(self, context_pool: str, idea_type: str) -> tuple[str, str]:
        """Helper function to generate an idea from a context pool

        This function encapsulates steps 3 and 4 from the user's request:
        3. Using the sample to ask LLM to create specific prompt
        4. Generate an idea from specific prompt

        Args:
            context_pool: The context pool to generate from
            idea_type: Type of idea to generate

        Returns:
            tuple: (generated_idea, specific_prompt_used)
        """
        # Generate specific prompt from context pool
        specific_prompt = self.generate_specific_prompt(context_pool, idea_type)

        # Generate idea using the specific prompt
        response = self.generate_text(specific_prompt)

        return response, specific_prompt

    def get_idea_prompt(self, idea_type: str) -> str:
        """Get prompt template for specific idea type"""
        prompts = get_prompts(idea_type)
        return prompts.IDEA_PROMPT

    def seed_ideas(self, n: int, idea_type: str) -> tuple[List[str], List[str]]:
        """Generate n initial ideas

        Returns:
            tuple: (ideas, specific_prompts)
        """
        ideas = []
        specific_prompts = []

        for _ in tqdm(range(n), desc="Generating ideas"):
            # Generate context pool
            context_pool = self.generate_context(idea_type)

            response, specific_prompt = self.generate_idea_from_context(context_pool, idea_type)
            specific_prompts.append(specific_prompt)

            ideas.append({"id": uuid.uuid4(), "idea": response, "parent_ids": []})

        return ideas, specific_prompts


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

        # Check if this is an Oracle-generated idea
        is_oracle_idea = isinstance(raw_idea, dict) and raw_idea.get("oracle_generated", False)
        oracle_analysis = raw_idea.get("oracle_analysis", "") if is_oracle_idea else ""

        # If raw_idea is a dictionary with 'id' and 'idea' keys, extract just the idea text
        idea_text = raw_idea["idea"] if isinstance(raw_idea, dict) and "idea" in raw_idea else raw_idea
        print(f"Formatting idea:\n {idea_text}")

        if is_oracle_idea:
            # For Oracle ideas, use a different prompt that preserves the analysis context
            prompt = f"""Format the following idea into a structured format with title and content.

ORACLE CONTEXT: This idea was generated by the Oracle diversity agent to address patterns and gaps in the population.

IDEA TO FORMAT:
{idea_text}

Please format this into a structured format with a compelling title and well-organized content. The Oracle analysis will be preserved separately.
"""
        else:
            prompt = prompts.FORMAT_PROMPT.format(input_text=idea_text)

        print(f"Prompt:\n {prompt}")
        response = self.generate_text(prompt, response_schema=Idea)

        try:
            formatted_idea = Idea(**json.loads(response))
        except (json.JSONDecodeError, ValueError) as e:
            print(f"FORMATTER: JSON parsing failed: {e}")
            print(f"FORMATTER: Raw response: {response}")

            # Fallback: Try to extract title and content manually
            title = "Untitled"
            content = response.strip()

            # Try to parse "Title: X\nContent: Y" format
            if "Title:" in response and "Content:" in response:
                lines = response.split('\n')
                title_line = None
                content_lines = []
                found_content = False

                for line in lines:
                    if line.strip().startswith("Title:"):
                        title_line = line.replace("Title:", "").strip()
                    elif line.strip().startswith("Content:"):
                        content_lines.append(line.replace("Content:", "").strip())
                        found_content = True
                    elif found_content:
                        content_lines.append(line)

                if title_line:
                    title = title_line
                if content_lines:
                    content = "\n".join(content_lines).strip()

            print(f"FORMATTER: Fallback extracted - Title: '{title}', Content length: {len(content)}")
            formatted_idea = Idea(title=title, content=content)

        # If the input was a dictionary with an ID, preserve that ID and all metadata
        if isinstance(raw_idea, dict) and "id" in raw_idea:
            result = {"id": raw_idea["id"], "idea": formatted_idea}
            # Preserve parent_ids if they exist
            if "parent_ids" in raw_idea:
                result["parent_ids"] = raw_idea["parent_ids"]
            else:
                result["parent_ids"] = []

            # Preserve Oracle metadata
            if is_oracle_idea:
                result["oracle_generated"] = True
                result["oracle_analysis"] = oracle_analysis

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

        # Handle edge case: if there's only one idea or no ideas, return appropriate ranking
        if len(ideas) <= 1:
            return {0: 1500} if len(ideas) == 1 else {}

        max_elo_diff = 100
        ranks = {i: 1500 for i in range(len(ideas))}
        for k in range(comparisons):
            if k < len(ideas):
                # assert that each idea is one side of the comparison at least once
                idea_idx_a = k
                # Create a list of valid indices excluding idea_idx_a and ensure the difference is within max_elo_diff
                other_indices = [idx for idx in range(len(ideas)) if idx != idea_idx_a]

                # Handle edge case: if no other indices available (shouldn't happen with len check above, but safety first)
                if len(other_indices) == 0:
                    continue

                valid_indices = [idx for idx in other_indices if abs(ranks[idx] - ranks[idea_idx_a]) <= max_elo_diff]
                if len(valid_indices) == 0:
                    idea_idx_b = np.random.choice(other_indices, size=1)[0]
                else:
                    idea_idx_b = np.random.choice(valid_indices, size=1)[0]
            else:
                # Handle edge case: ensure we have at least 2 ideas for comparison
                if len(ideas) < 2:
                    continue
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
            # Use the configured temperature for comparisons
            response = self.generate_text(prompt)
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


class Breeder(LLMWrapper):
    """Breeds ideas and handles genotype encoding/decoding"""
    agent_name = "Breeder"
    parent_count = 2

    def __init__(self, **kwargs):
        # Don't set temperature directly here, let it come from kwargs
        super().__init__(agent_name=self.agent_name, **kwargs)

    def encode_to_genotype(self, idea: str, idea_type: str) -> str:
        """
        Convert a full idea (phenotype) to its basic elements (genotype)

        Args:
            idea: The full idea to encode (can be dict with 'idea' key or string)
            idea_type: Type of idea for customization

        Returns:
            A genotype string representing the basic elements
        """
        # Extract idea text if it's a dictionary
        idea_text = idea["idea"] if isinstance(idea, dict) and "idea" in idea else idea

        # Get the idea object if it's a formatted idea
        if hasattr(idea_text, 'title') and hasattr(idea_text, 'content'):
            idea_content = f"Title: {idea_text.title}\nContent: {idea_text.content}"
        else:
            idea_content = str(idea_text)

        # Create encoding prompt based on idea type
        prompts = get_prompts(idea_type)

        # Get genotype encoding prompt from template
        prompt_template = prompts.GENOTYPE_ENCODE_PROMPT

        prompt = prompt_template.format(idea_content=idea_content)

        response = self.generate_text(prompt)

        # Clean up the response
        genotype = response.strip()
        if genotype.startswith("Genotype:"):
            genotype = genotype[9:].strip()

        print(f"Encoded genotype: {genotype}")
        return genotype

    def breed(self, ideas: List[str], idea_type: str) -> str:
        """Breed ideas to create a new idea using the new approach that mirrors initial population generation

        This follows the same pattern as initial idea generation:
        1. Get concepts from the parents (their genotypes) and combine them
        2. Sample 50% at random (in python code)
        3. Using the sample to ask LLM to create specific prompt
        4. Generate an idea from specific prompt

        Args:
            ideas: List of parent ideas that have already been selected in the main evolution loop
            idea_type: Type of idea to breed

        Returns:
            A new idea with a unique ID and parent IDs
        """
        # Extract parent IDs
        parent_ids = []
        for parent in ideas:
            if isinstance(parent, dict) and "id" in parent:
                parent_ids.append(str(parent["id"]))

        # Step 1: Get concepts from the parents (their genotypes) and combine them
        parent_genotypes = []
        for parent in ideas:
            genotype = self.encode_to_genotype(parent, idea_type)
            parent_genotypes.append(genotype)

        # We need to get an instance of Ideator to use the new helper methods
        # Use the same configuration as this Breeder instance
        ideator = Ideator(
            provider=self.provider,
            model_name=self.model_name,
            temperature=self.temperature
        )

        # Step 2: Sample 50% at random (handled in generate_context_from_parents)
        # Step 3 & 4: Using the sample to create specific prompt and generate idea
        context_pool = ideator.generate_context_from_parents(parent_genotypes)
        new_idea, specific_prompt = ideator.generate_idea_from_context(context_pool, idea_type)

        print(f"Generated new idea from parent concepts: {new_idea[:100]}...")
        print(f"Using specific prompt: {specific_prompt[:100]}...")

        # Create a new idea with a unique ID and parent IDs
        # Also include the specific prompt used for breeding
        return {
            "id": uuid.uuid4(),
            "idea": new_idea,
            "parent_ids": parent_ids,
            "specific_prompt": specific_prompt  # Store the specific prompt for later display
        }


class Oracle(LLMWrapper):
    """Analyzes entire population history and introduces diversity by identifying overused patterns"""
    agent_name = "Oracle"

    def __init__(self, **kwargs):
        # Use a higher temperature to encourage creativity and diversity
        temp = kwargs.pop('temperature', 1.8)
        super().__init__(agent_name=self.agent_name, temperature=temp, **kwargs)

    def analyze_and_diversify(self, history: List[List[str]], idea_type: str) -> dict:
        """
        Analyze the entire evolution history and generate a new diverse idea to replace an existing one.

        The replacement selection is handled by embedding-based centroid distance calculation.

        Args:
            history: Complete evolution history, list of generations (each generation is a list of ideas)
            idea_type: Type of ideas being evolved

        Returns:
            Dictionary with new_idea and action type
        """
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt(history, idea_type)

        print(f"Oracle analyzing {len(history)} generations with {sum(len(gen) for gen in history)} total ideas...")

        # This is the "expensive" single query that does everything
        response = self.generate_text(analysis_prompt)

        return self._parse_oracle_response(response)

    def _build_analysis_prompt(self, history: List[List[str]], idea_type: str) -> str:
        """Build the comprehensive analysis prompt for the Oracle"""

        # Format all historical ideas by generation
        history_text = ""
        for gen_idx, generation in enumerate(history):
            history_text += f"\n--- GENERATION {gen_idx} ---\n"
            for idea_idx, idea in enumerate(generation):
                idea_content = self._extract_idea_content(idea)
                history_text += f"Idea {gen_idx}.{idea_idx}: {idea_content}\n"

        # Get prompts for the idea type
        prompts = get_prompts(idea_type)
        base_idea_prompt = prompts.IDEA_PROMPT

        # Get Oracle-specific prompts from the template
        mode_instruction = prompts.ORACLE_INSTRUCTION
        format_instructions = prompts.ORACLE_FORMAT_INSTRUCTIONS
        example_idea_prompts = getattr(prompts, 'EXAMPLE_IDEA_PROMPTS', '')

        # Build the Oracle's comprehensive prompt using the template
        oracle_constraints = getattr(prompts, 'ORACLE_CONSTRAINTS', '')
        prompt = prompts.ORACLE_MAIN_PROMPT.format(
            idea_type=idea_type,
            history_text=history_text,
            mode_instruction=mode_instruction,
            format_instructions=format_instructions,
            oracle_constraints=oracle_constraints,
            example_idea_prompts=example_idea_prompts
        )

        return prompt

    def _extract_idea_content(self, idea) -> str:
        """Extract readable content from an idea (handles various formats)"""
        if isinstance(idea, dict) and "idea" in idea:
            idea_obj = idea["idea"]
            if hasattr(idea_obj, 'title') and hasattr(idea_obj, 'content'):
                return f"{idea_obj.title}: {idea_obj.content}"
            else:
                return str(idea_obj)
        else:
            return str(idea)

    def _parse_oracle_response(self, response: str) -> dict:
        """Parse the Oracle's response and extract the analysis and new idea separately"""

        # Parse the structured response to separate analysis from new idea
        oracle_analysis = ""
        idea_prompt = ""

        try:
            # Split response into sections
            if "=== ORACLE ANALYSIS ===" in response and "=== IDEA PROMPT ===" in response:
                parts = response.split("=== ORACLE ANALYSIS ===")[1]
                analysis_part, idea_part = parts.split("=== IDEA PROMPT ===")
                oracle_analysis = analysis_part.strip()
                idea_prompt = idea_part.strip()
                print(f"ORACLE: Successfully parsed structured response - analysis: {len(oracle_analysis)} chars, idea prompt: {len(idea_prompt)} chars")
            else:
                # Fallback: treat entire response as new idea content but preserve Oracle metadata
                idea_prompt = response.strip()
                oracle_analysis = f"Oracle response was not properly formatted. Expected sections '=== ORACLE ANALYSIS ===' and '=== IDEA PROMPT ===' but got unstructured response. This indicates the LLM did not follow the required format."
                print(f"ORACLE: Fallback parsing - treating entire response as idea content. Response length: {len(response)} chars")
        except Exception as e:
            # Fallback parsing failed
            idea_prompt = response.strip()
            oracle_analysis = f"Oracle parsing error: {e}. Response was treated as idea content."
            print(f"ORACLE: Parsing exception - {e}")

        # Oracle only supports replace mode
        # Replacement selection is handled externally via embedding-based centroid distance
        print(f"ORACLE: Generated replacement idea. Selection of which idea to replace will be handled externally via embeddings.")

        return {
            "action": "replace",
            "idea_prompt": idea_prompt,
            "oracle_analysis": oracle_analysis
        }
