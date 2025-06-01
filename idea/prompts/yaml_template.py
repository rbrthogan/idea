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

        # Unified special requirements with legacy compatibility
        requirements = self.template.requirements
        if requirements:
            self.SPECIAL_REQUIREMENTS = requirements

            # Legacy compatibility attributes
            if self.template.format_requirements:
                self.DRABBLE_FORMAT_PROMPT = requirements  # Legacy name for drabble
                self.FORMAT_REQUIREMENTS = requirements
            elif self.template.design_requirements:
                self.DESIGN_REQUIREMENTS = requirements
            else:
                # For new templates using special_requirements
                self.FORMAT_REQUIREMENTS = requirements
                self.DESIGN_REQUIREMENTS = requirements

    def _interpolate_prompt(self, prompt_text: str) -> str:
        """
        Interpolate template-specific requirements into prompt text
        Supports both new {requirements} placeholder and legacy {format_requirements}/{design_requirements}
        """
        result = prompt_text
        requirements = self.template.requirements

        if requirements:
            # Handle new unified placeholder
            if '{requirements}' in result:
                result = result.replace('{requirements}', requirements)

            # Handle legacy placeholders for backward compatibility
            if '{format_requirements}' in result:
                result = result.replace('{format_requirements}', requirements)

            if '{design_requirements}' in result:
                result = result.replace('{design_requirements}', requirements)

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