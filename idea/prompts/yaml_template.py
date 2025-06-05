"""
YAML Template Wrapper - makes YAML templates behave like Python modules
"""

from typing import Optional
from .validation import PromptTemplate

DEFAULT_COMPARISON_PROMPT = (
    "You are an expert evaluator of {item_type}. You will be presented with two {item_type}, and your task is to determine which one is better.\n\n"
    "Idea A:\n"
    "Title: {idea_a_title}\n"
    "{idea_a_proposal}\n\n"
    "Idea B:\n"
    "Title: {idea_b_title}\n"
    "{idea_b_proposal}\n\n"
    "Evaluate both ideas based on the following criteria:\n"
    "{criteria}\n\n"
    "Criterion 1 is the most important.\n\n"
    "After your evaluation, respond with exactly one of these three options:\n"
    "- \"Result: A\" if Idea A is better\n"
    "- \"Result: B\" if Idea B is better\n"
    "- \"Result: tie\" if both ideas are approximately equal in quality\n\n"
    "Your response must contain exactly one of these three phrases and nothing else."
)


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
        self.COMPARISON_CRITERIA = self.template.comparison_criteria

        # Core prompts
        self.CONTEXT_PROMPT = self.template.prompts.context
        self.IDEA_PROMPT = self._interpolate_prompt(self.template.prompts.idea)
        self.NEW_IDEA_PROMPT = self._interpolate_prompt(self.template.prompts.new_idea)
        self.FORMAT_PROMPT = self.template.prompts.format
        self.CRITIQUE_PROMPT = self.template.prompts.critique
        self.REFINE_PROMPT = self._interpolate_prompt(self.template.prompts.refine)
        self.BREED_PROMPT = self._interpolate_prompt(self.template.prompts.breed)

        # Optional comparison prompt with default for backward compatibility
        comparison_prompt = getattr(self.template.prompts, 'comparison_prompt', None)
        if comparison_prompt is None:
            comparison_prompt = getattr(self.template, 'comparison_prompt', None)
        if comparison_prompt is None:
            comparison_prompt = DEFAULT_COMPARISON_PROMPT

        self.COMPARISON_PROMPT = self._interpolate_prompt(comparison_prompt)

        # Special requirements
        if self.template.special_requirements:
            self.SPECIAL_REQUIREMENTS = self.template.special_requirements

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