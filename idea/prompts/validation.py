"""
Validation for YAML prompt templates
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field, validator
import yaml
from datetime import datetime


class PromptMetadata(BaseModel):
    """Metadata for a prompt template"""
    item_type: str = Field(..., description="Type of items generated (e.g., 'stories', 'game designs')")


class PromptSet(BaseModel):
    """Set of prompts required for idea generation"""
    context: str = Field(..., description="Prompt for generating contextual concepts")
    idea: str = Field(..., description="Prompt for generating initial ideas")
    new_idea: str = Field(..., description="Prompt for generating new ideas from existing ones")
    format: str = Field(..., description="Prompt for formatting raw ideas")
    critique: str = Field(..., description="Prompt for critiquing ideas")
    refine: str = Field(..., description="Prompt for refining ideas based on critique")
    breed: str = Field(..., description="Prompt for breeding/combining ideas")


class PromptTemplate(BaseModel):
    """Complete prompt template structure"""
    name: str = Field(..., description="Human-readable name of the template")
    description: str = Field(..., description="Description of what this template generates")
    version: str = Field(..., description="Template version (semantic versioning)")
    author: str = Field(..., description="Template author")
    created_date: str = Field(..., description="Creation date (YYYY-MM-DD)")

    metadata: PromptMetadata = Field(..., description="Template metadata")
    prompts: PromptSet = Field(..., description="Set of prompts")
    comparison_criteria: List[str] = Field(..., description="Criteria for comparing generated ideas")

    # Unified special requirements field (replaces format_requirements and design_requirements)
    special_requirements: Optional[str] = Field(None, description="Special requirements for this template type")

    # Legacy fields for backward compatibility - will be deprecated
    format_requirements: Optional[str] = Field(None, description="Legacy: use special_requirements instead")
    design_requirements: Optional[str] = Field(None, description="Legacy: use special_requirements instead")

    def __init__(self, **data):
        """Initialize template with special requirements migration"""
        # Auto-migrate from legacy fields to special_requirements
        if 'special_requirements' not in data or data['special_requirements'] is None:
            if data.get('format_requirements'):
                data['special_requirements'] = data['format_requirements']
            elif data.get('design_requirements'):
                data['special_requirements'] = data['design_requirements']

        super().__init__(**data)

    @property
    def requirements(self) -> Optional[str]:
        """Get the special requirements, handling legacy compatibility"""
        if self.special_requirements:
            return self.special_requirements
        elif self.format_requirements:
            return self.format_requirements
        elif self.design_requirements:
            return self.design_requirements
        return None

    @validator('version')
    def validate_version(cls, v):
        """Validate semantic versioning format"""
        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError('Version must be in format X.Y.Z')
        for part in parts:
            if not part.isdigit():
                raise ValueError('Version parts must be numeric')
        return v

    @validator('created_date')
    def validate_date(cls, v):
        """Validate date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @validator('comparison_criteria')
    def validate_criteria(cls, v):
        """Ensure at least one comparison criterion"""
        if not v or len(v) == 0:
            raise ValueError('At least one comparison criterion is required')
        return v


class TemplateValidator:
    """Validates YAML prompt templates"""

    @staticmethod
    def load_and_validate(file_path: str) -> PromptTemplate:
        """Load and validate a YAML template file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML syntax: {e}")
        except FileNotFoundError:
            raise ValueError(f"Template file not found: {file_path}")

        try:
            return PromptTemplate(**data)
        except Exception as e:
            raise ValueError(f"Template validation failed: {e}")

    @staticmethod
    def validate_dict(data: Dict[str, Any]) -> PromptTemplate:
        """Validate a template from a dictionary"""
        try:
            return PromptTemplate(**data)
        except Exception as e:
            raise ValueError(f"Template validation failed: {e}")

    @staticmethod
    def check_prompt_interpolation(template: PromptTemplate) -> List[str]:
        """Check for potential interpolation issues in prompts"""
        warnings = []

        # Check for missing format placeholders
        prompts_dict = template.prompts.dict()

        # Check format prompt for {input_text}
        if '{input_text}' not in prompts_dict['format']:
            warnings.append("Format prompt missing {input_text} placeholder")

        # Check critique prompt for {idea}
        if '{idea}' not in prompts_dict['critique']:
            warnings.append("Critique prompt missing {idea} placeholder")

        # Check refine prompt for {idea} and {critique}
        refine_prompt = prompts_dict['refine']
        if '{idea}' not in refine_prompt:
            warnings.append("Refine prompt missing {idea} placeholder")
        if '{critique}' not in refine_prompt:
            warnings.append("Refine prompt missing {critique} placeholder")

        # Check breed prompt for {ideas}
        if '{ideas}' not in prompts_dict['breed']:
            warnings.append("Breed prompt missing {ideas} placeholder")

        # Check for template-specific requirements interpolation
        if template.requirements:
            for prompt_name, prompt_text in prompts_dict.items():
                if '{requirements}' in prompt_text and not template.requirements:
                    warnings.append(f"{prompt_name} references requirements but it's not defined")

        return warnings


def validate_template_file(file_path: str) -> tuple[PromptTemplate, List[str]]:
    """
    Validate a template file and return the template and any warnings

    Returns:
        tuple: (validated_template, list_of_warnings)
    """
    template = TemplateValidator.load_and_validate(file_path)
    warnings = TemplateValidator.check_prompt_interpolation(template)

    return template, warnings