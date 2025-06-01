"""
YAML Template Wrapper - makes YAML templates behave like Python modules
"""

from typing import Optional
from .validation import PromptTemplate


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