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
            # Generate specific prompt from context pool
            specific_prompt = self.generate_specific_prompt(context_pool, idea_type)
            specific_prompts.append(specific_prompt)
            # Generate idea using the specific prompt
            response = self.generate_text(specific_prompt)

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
        formatted_idea = Idea(**json.loads(response))

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

    def breed(self, ideas: List[str], idea_type: str, genotype_encoder: 'GenotypeEncoder') -> str:
        """Breed ideas to create a new idea using genotype-based breeding

        Args:
            ideas: List of parent ideas that have already been selected in the main evolution loop
            idea_type: Type of idea to breed
            genotype_encoder: GenotypeEncoder instance for genotype operations

        Returns:
            A new idea with a unique ID and parent IDs
        """
        # Extract parent IDs
        parent_ids = []
        for parent in ideas:
            if isinstance(parent, dict) and "id" in parent:
                parent_ids.append(str(parent["id"]))


        # Encode parent ideas to genotypes
        parent_genotypes = []
        for parent in ideas:
            genotype = genotype_encoder.encode_to_genotype(parent, idea_type)
            parent_genotypes.append(genotype)

        # Perform genetic crossover
        new_genotype = genotype_encoder.crossover_genotypes(parent_genotypes, idea_type)

        # Decode the new genotype back to a full idea
        new_idea = genotype_encoder.decode_from_genotype(new_genotype, idea_type)

        print(f"Generated new idea from genotype crossover: {new_idea[:100]}...")

        # Create a new idea with a unique ID and parent IDs
        return {"id": uuid.uuid4(), "idea": new_idea, "parent_ids": parent_ids}

class GenotypeEncoder(LLMWrapper):
    """Encodes ideas to genotypes (basic elements) and decodes genotypes back to ideas"""
    agent_name = "GenotypeEncoder"

    def __init__(self, **kwargs):
        # Use a moderate temperature for encoding/decoding
        temp = kwargs.pop('temperature', 1.2)
        super().__init__(agent_name=self.agent_name, temperature=temp, **kwargs)

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

    def decode_from_genotype(self, genotype: str, idea_type: str) -> str:
        """
        Convert basic elements (genotype) back to a full idea (phenotype)

        Args:
            genotype: The genotype string with basic elements
            idea_type: Type of idea for customization

        Returns:
            A full idea string
        """
        # Create decoding prompt based on idea type
        prompts = get_prompts(idea_type)

        # Get genotype decoding prompt from template
        prompt_template = prompts.GENOTYPE_DECODE_PROMPT

        prompt = prompt_template.format(genotype=genotype)

        response = self.generate_text(prompt)

        print(f"Decoded from genotype '{genotype}' to: {response[:100]}...")
        return response

    def crossover_genotypes(self, genotypes: List[str], idea_type: str) -> str:
        """
        Perform genetic crossover between multiple genotypes to create a new genotype

        Args:
            genotypes: List of parent genotypes
            idea_type: Type of idea for customization

        Returns:
            A new genotype created from crossover
        """
        # Create crossover prompt based on idea type
        prompts = get_prompts(idea_type)

        # Get genotype crossover prompt from template
        prompt_template = prompts.GENOTYPE_CROSSOVER_PROMPT

        # Format parent genotypes
        parent_str = "\n".join([f"Parent {i+1}: {genotype}" for i, genotype in enumerate(genotypes)])

        prompt = prompt_template.format(parent_genotypes=parent_str)

        response = self.generate_text(prompt)

        # Clean up the response
        new_genotype = response.strip()
        if ":" in new_genotype and new_genotype.startswith(("New genotype", "Result", "Output")):
            new_genotype = new_genotype.split(":", 1)[1].strip()

        print(f"Crossover result: {new_genotype}")
        return new_genotype


class Oracle(LLMWrapper):
    """Analyzes entire population history and introduces diversity by identifying overused patterns"""
    agent_name = "Oracle"

    def __init__(self, **kwargs):
        # Use a higher temperature to encourage creativity and diversity
        temp = kwargs.pop('temperature', 1.8)
        super().__init__(agent_name=self.agent_name, temperature=temp, **kwargs)

    def analyze_and_diversify(self, history: List[List[str]], current_generation: List[str], idea_type: str, oracle_mode: str = "add") -> dict:
        """
        Analyze the entire evolution history and either add a new diverse idea or replace a similar one

        Args:
            history: Complete evolution history, list of generations (each generation is a list of ideas)
            current_generation: Current generation's ideas
            idea_type: Type of ideas being evolved
            oracle_mode: "add" to grow population by 1, "replace" to replace similar idea

        Returns:
            Dictionary with either new_idea or replacement info
        """
        # Build comprehensive analysis prompt
        analysis_prompt = self._build_analysis_prompt(history, current_generation, idea_type, oracle_mode)

        print(f"Oracle analyzing {len(history)} generations with {sum(len(gen) for gen in history)} total ideas...")

        # This is the "expensive" single query that does everything
        response = self.generate_text(analysis_prompt)

        return self._parse_oracle_response(response, current_generation, oracle_mode)

    def _build_analysis_prompt(self, history: List[List[str]], current_generation: List[str], idea_type: str, oracle_mode: str) -> str:
        """Build the comprehensive analysis prompt for the Oracle"""

        # Format all historical ideas by generation
        history_text = ""
        for gen_idx, generation in enumerate(history):
            history_text += f"\n--- GENERATION {gen_idx} ---\n"
            for idea_idx, idea in enumerate(generation):
                idea_content = self._extract_idea_content(idea)
                history_text += f"Idea {gen_idx}.{idea_idx}: {idea_content}\n"

        # Format current generation
        current_text = "\n--- CURRENT GENERATION ---\n"
        for idea_idx, idea in enumerate(current_generation):
            idea_content = self._extract_idea_content(idea)
            current_text += f"Idea {idea_idx}: {idea_content}\n"

        # Get prompts for the idea type
        prompts = get_prompts(idea_type)
        base_idea_prompt = prompts.IDEA_PROMPT

        # Get Oracle-specific prompts from the template
        if oracle_mode == "add":
            mode_instruction = prompts.ORACLE_ADD_MODE_INSTRUCTION
            format_instructions = prompts.ORACLE_ADD_FORMAT_INSTRUCTIONS
        else:  # replace mode
            mode_instruction = prompts.ORACLE_REPLACE_MODE_INSTRUCTION
            format_instructions = prompts.ORACLE_REPLACE_FORMAT_INSTRUCTIONS

        # Build the Oracle's comprehensive prompt using the template
        oracle_constraints = getattr(prompts, 'ORACLE_CONSTRAINTS', '')
        prompt = prompts.ORACLE_MAIN_PROMPT.format(
            idea_type=idea_type,
            base_idea_prompt=base_idea_prompt,
            history_text=history_text,
            current_text=current_text,
            mode_instruction=mode_instruction,
            format_instructions=format_instructions,
            oracle_constraints=oracle_constraints
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

    def _parse_oracle_response(self, response: str, current_generation: List[str], oracle_mode: str) -> dict:
        """Parse the Oracle's response and extract the analysis and new idea separately"""

        # Parse the structured response to separate analysis from new idea
        oracle_analysis = ""
        new_idea_text = ""

        try:
            # Split response into sections
            if "=== ORACLE ANALYSIS ===" in response and "=== NEW IDEA ===" in response:
                parts = response.split("=== ORACLE ANALYSIS ===")[1]
                analysis_part, idea_part = parts.split("=== NEW IDEA ===")
                oracle_analysis = analysis_part.strip()
                new_idea_text = idea_part.strip()
                print(f"ORACLE: Successfully parsed structured response - analysis: {len(oracle_analysis)} chars, idea: {len(new_idea_text)} chars")
            else:
                # Fallback: treat entire response as new idea content but preserve Oracle metadata
                new_idea_text = response.strip()
                oracle_analysis = f"Oracle response was not properly formatted. Expected sections '=== ORACLE ANALYSIS ===' and '=== NEW IDEA ===' but got unstructured response. This indicates the LLM did not follow the required format."
                print(f"ORACLE: Fallback parsing - treating entire response as idea content. Response length: {len(response)} chars")
        except Exception as e:
            # Fallback parsing failed
            new_idea_text = response.strip()
            oracle_analysis = f"Oracle parsing error: {e}. Response was treated as idea content."
            print(f"ORACLE: Parsing exception - {e}")

        if oracle_mode == "add":
            # Create new idea with unique ID
            return {
                "action": "add",
                "new_idea": {
                    "id": str(uuid.uuid4()),
                    "idea": new_idea_text,
                    "parent_ids": [],
                    "oracle_generated": True,
                    "oracle_analysis": oracle_analysis
                }
            }

        else:  # replace mode
            # Look for REPLACE_INDEX in the oracle analysis section
            replace_index = 0  # default to first idea if parsing fails

            # Try to parse REPLACE_INDEX from the oracle analysis section
            analysis_lines = oracle_analysis.split('\n')
            for line in analysis_lines:
                if line.strip().startswith('REPLACE_INDEX:'):
                    try:
                        replace_index = int(line.split(':')[1].strip())
                        print(f"ORACLE: Found REPLACE_INDEX: {replace_index}")
                        break
                    except (ValueError, IndexError) as e:
                        print(f"ORACLE: Error parsing REPLACE_INDEX from line '{line}': {e}")
                        replace_index = 0

            # If we didn't find it in the analysis, also check the entire response as fallback
            if replace_index == 0:  # Only search if we didn't find it above
                lines = response.split('\n')
                for line in lines:
                    if line.strip().startswith('REPLACE_INDEX:'):
                        try:
                            replace_index = int(line.split(':')[1].strip())
                            print(f"ORACLE: Found REPLACE_INDEX in fallback search: {replace_index}")
                            break
                        except (ValueError, IndexError):
                            replace_index = 0

            # Ensure replace_index is valid
            if replace_index >= len(current_generation):
                print(f"ORACLE: REPLACE_INDEX {replace_index} exceeds generation size {len(current_generation)}, using last index")
                replace_index = len(current_generation) - 1

            print(f"ORACLE: Final REPLACE_INDEX: {replace_index}")

            return {
                "action": "replace",
                "replace_index": replace_index,
                "new_idea": {
                    "id": str(uuid.uuid4()),
                    "idea": new_idea_text,
                    "parent_ids": [],
                    "oracle_generated": True,
                    "oracle_analysis": oracle_analysis
                }
            }
