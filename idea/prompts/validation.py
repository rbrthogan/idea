"""
Validation for YAML prompt templates
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, ConfigDict, Field, StrictStr, field_validator, model_validator
import yaml
from datetime import datetime


class PromptMetadata(BaseModel):
    """Metadata for a prompt template"""
    item_type: str = Field(..., description="Type of items generated (e.g., 'stories', 'game designs')")


class TemplateEvaluationCapabilities(BaseModel):
    """Managed evaluation features a template may request."""

    artifact_generation: bool = False
    fitness_score: bool = False


class OutputWordCount(BaseModel):
    """Requested body length, excluding the title. Equal bounds mean exactly."""

    model_config = ConfigDict(extra="forbid")
    min: Optional[int] = Field(None, gt=0, strict=True)
    max: Optional[int] = Field(None, gt=0, strict=True)

    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min is None and self.max is None:
            raise ValueError("Word count needs a minimum or maximum.")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError("Minimum word count cannot exceed maximum word count.")
        return self


class TemplateOutput(BaseModel):
    """Content guidance for the app's fixed title/content result structure."""

    model_config = ConfigDict(extra="forbid")
    title: StrictStr = ""
    content: StrictStr = ""
    word_count: Optional[OutputWordCount] = None


class PromptSet(BaseModel):
    """Set of prompts required for idea generation"""
    context: str = Field(..., description="Prompt for generating contextual concepts")
    specific_prompt: str = Field(..., description="Prompt for turning contextual concepts into concrete idea directions")

    format: str = Field(..., description="Prompt for formatting raw ideas")
    critique: str = Field(..., description="Prompt for critiquing ideas")
    refine: str = Field(..., description="Prompt for refining ideas based on critique")
    breed: str = Field(..., description="Prompt for breeding/combining ideas")
    genotype_encode: str = Field(..., description="Prompt for encoding ideas to genotypes")
    # Optional comparison prompt - will be generated dynamically if not provided
    comparison_prompt: Optional[str] = Field(None, description="Prompt for comparing ideas (optional - auto-generated)")
    oracle_constraints: Optional[str] = Field(None, description="Template-specific goals and boundaries for Oracle diversity analysis")
    example_idea_prompts: Optional[str] = Field(None, description="Illustrative directions for Oracle analysis")


class PromptTemplate(BaseModel):
    """Complete prompt template structure"""
    name: str = Field(..., description="Human-readable name of the template")
    description: str = Field(..., description="Description of what this template generates")
    version: str = Field(..., description="Template version (semantic versioning)")
    author: str = Field(..., description="Template author")
    created_date: str = Field(..., description="Creation date (YYYY-MM-DD)")

    metadata: PromptMetadata = Field(..., description="Template metadata")
    evaluation: TemplateEvaluationCapabilities = Field(
        default_factory=TemplateEvaluationCapabilities,
        description="Managed evaluation features used by this template",
    )
    prompts: PromptSet = Field(..., description="Set of prompts")
    comparison_criteria: List[str] = Field(
        default_factory=list,
        description=(
            "Optional criteria for the legacy pairwise LLM evaluator. "
            "Templates that use other evaluators do not need to define them."
        ),
    )
    # Optional comparison prompt at root level for backward compatibility
    comparison_prompt: Optional[str] = Field(None, description="Prompt for comparing ideas (optional - auto-generated)")

    # Special requirements for this template type
    special_requirements: Optional[str] = Field(None, description="Special requirements for this template type")
    requirements: List[StrictStr] = Field(
        default_factory=list,
        description="Individual content and domain requirements applied throughout evolution",
    )
    output: Optional[TemplateOutput] = Field(
        None,
        description="Title/body guidance and requested body length; the app owns the transport format",
    )

    @field_validator('requirements')
    @classmethod
    def validate_requirements(cls, value):
        if any(not rule.strip() for rule in value):
            raise ValueError('Requirements must be non-empty strings')
        return [rule.strip() for rule in value]

    @field_validator('version')
    @classmethod
    def validate_version(cls, v):
        """Validate semantic versioning format"""
        parts = v.split('.')
        if len(parts) != 3:
            raise ValueError('Version must be in format X.Y.Z')
        for part in parts:
            if not part.isdigit():
                raise ValueError('Version parts must be numeric')
        return v

    @field_validator('created_date')
    @classmethod
    def validate_date(cls, v):
        """Validate date format"""
        try:
            datetime.strptime(v, '%Y-%m-%d')
            return v
        except ValueError:
            raise ValueError('Date must be in YYYY-MM-DD format')

    @field_validator('comparison_criteria')
    @classmethod
    def validate_criteria(cls, v):
        """Reject blank criteria while allowing non-LLM evaluation templates."""
        if v is None:
            return []
        if any(not isinstance(criterion, str) or not criterion.strip() for criterion in v):
            raise ValueError('Comparison criteria must be non-empty strings')
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
        prompts_dict = template.prompts.model_dump()

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
        has_requirements = bool(template.special_requirements or template.requirements or template.output)
        for prompt_name, prompt_text in prompts_dict.items():
            if prompt_text and any(token in prompt_text for token in ('{requirements}', '{requirements_digest}')) and not has_requirements:
                warnings.append(f"{prompt_name} references requirements but none are defined")

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
