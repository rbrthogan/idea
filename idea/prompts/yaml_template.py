"""
YAML Template Wrapper - makes YAML templates behave like Python modules
"""

from typing import Optional, List
from .validation import PromptTemplate


def generate_comparison_prompt(item_type: str, criteria: List[str]) -> str:
    """
    Generate a standardized comparison prompt from item type and criteria.

    Args:
        item_type: The type of items being compared (e.g., "stories", "research proposals")
        criteria: List of criteria to evaluate (first is most important)

    Returns:
        Formatted comparison prompt string
    """
    # Handle None or empty criteria
    if not criteria:
        criteria = ["overall quality", "clarity", "originality"]

    criteria_text = "\n".join(f"{i+1}. {criterion}" for i, criterion in enumerate(criteria))

    return f"""You are an expert evaluator of {item_type}. You will be presented with two {item_type}, and your task is to determine which one is better.

Idea A:
Title: {{idea_a_title}}
{{idea_a_content}}

Idea B:
Title: {{idea_b_title}}
{{idea_b_content}}

Evaluate both ideas based on the following criteria:
{criteria_text}

Criterion 1 is the most important.

After your evaluation, respond with exactly one of these three options:
- "Result: A" if Idea A is better
- "Result: B" if Idea B is better
- "Result: tie" if both ideas are approximately equal in quality

Your response must contain exactly one of these three phrases and nothing else."""


class YAMLTemplateWrapper:
    """
    Wrapper that makes YAML templates behave like the original Python prompt modules
    This ensures backward compatibility with existing code
    """

    def __init__(self, template: PromptTemplate):
        self.template = template
        self._setup_attributes()

    def _setup_attributes(self):
        """Set up module-like attributes from the template"""
        # Basic metadata
        self.ITEM_TYPE = self.template.metadata.item_type
        self.COMPARISON_CRITERIA = self.template.comparison_criteria or []

        # Core prompts
        self.CONTEXT_PROMPT = self.template.prompts.context
        self.IDEA_PROMPT = self._interpolate_prompt(self.template.prompts.idea)
        self.SPECIFIC_PROMPT = self._interpolate_prompt(self.template.prompts.specific_prompt)

        self.FORMAT_PROMPT = self.template.prompts.format
        self.CRITIQUE_PROMPT = self.template.prompts.critique
        self.REFINE_PROMPT = self._interpolate_prompt(self.template.prompts.refine)
        self.BREED_PROMPT = self._interpolate_prompt(self.template.prompts.breed)

        self.GENOTYPE_ENCODE_PROMPT = self.template.prompts.genotype_encode

        # Generate comparison prompt dynamically from criteria
        self.COMPARISON_PROMPT = generate_comparison_prompt(
            self.ITEM_TYPE,
            self.COMPARISON_CRITERIA
        )

        # Oracle prompts will be set by the loader

    def _interpolate_prompt(self, prompt_text: str) -> str:
        """
        Interpolate template-specific requirements into prompt text
        """
        result = prompt_text

        if self.template.special_requirements and '{requirements}' in result:
            result = result.replace('{requirements}', self.template.special_requirements)

        return result

    @property
    def name(self) -> str:
        """Get template name"""
        return self.template.name

    @property
    def description(self) -> str:
        """Get template description"""
        return self.template.description

    @property
    def version(self) -> str:
        """Get template version"""
        return self.template.version

    @property
    def author(self) -> str:
        """Get template author"""
        return self.template.author

    def get_info(self) -> dict:
        """Get template information"""
        return {
            'name': self.template.name,
            'description': self.template.description,
            'version': self.template.version,
            'author': self.template.author,
            'created_date': self.template.created_date,
            'item_type': self.template.metadata.item_type
        }