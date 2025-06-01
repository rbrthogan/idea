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

        # Optional template-specific attributes
        if self.template.format_requirements:
            self.DRABBLE_FORMAT_PROMPT = self.template.format_requirements  # Legacy name for drabble
            self.FORMAT_REQUIREMENTS = self.template.format_requirements

        if self.template.design_requirements:
            self.DESIGN_REQUIREMENTS = self.template.design_requirements

    def _interpolate_prompt(self, prompt_text: str) -> str:
        """
        Interpolate template-specific requirements into prompt text
        """
        result = prompt_text

        # Interpolate format requirements if they exist and are referenced
        if self.template.format_requirements and '{format_requirements}' in result:
            result = result.replace('{format_requirements}', self.template.format_requirements)

        # Interpolate design requirements if they exist and are referenced
        if self.template.design_requirements and '{design_requirements}' in result:
            result = result.replace('{design_requirements}', self.template.design_requirements)

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