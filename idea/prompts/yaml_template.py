"""
YAML Template Wrapper - makes YAML templates behave like Python modules
"""

from typing import List
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
{item_type} A:
Title: {{idea_a_title}}
{{idea_a_content}}

{item_type} B:
Title: {{idea_b_title}}
{{idea_b_content}}

Evaluate both {item_type}s based on the following criteria:
{criteria_text}

Criterion 1 is the most important.

After your evaluation, respond with exactly one of these three options:
- "Result: A" if {item_type} A is better
- "Result: B" if {item_type} B is better
- "Result: tie" if both {item_type}s are approximately equal in quality

First pick a winner along each criterion (A or B) -- brief or no explanation required. Then provide 1-2 sentence summary of your evaluation. Then, provide your final verdict.
Your response must contain exactly one of these three phrases"""


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
        self.REQUIREMENTS_DIGEST = self._build_requirements_digest()

        # Core prompts
        self.CONTEXT_PROMPT = self.template.prompts.context
        self.IDEA_PROMPT = self._interpolate_prompt(self.template.prompts.idea)
        self.SPECIFIC_PROMPT = self._interpolate_prompt(self.template.prompts.specific_prompt)

        self.FORMAT_PROMPT = self.template.prompts.format
        self.CRITIQUE_PROMPT = self._interpolate_prompt(self.template.prompts.critique)
        self.REFINE_PROMPT = self._interpolate_prompt(self.template.prompts.refine)
        self.BREED_PROMPT = self._interpolate_prompt(self.template.prompts.breed)

        self.GENOTYPE_ENCODE_PROMPT = self._interpolate_prompt(self.template.prompts.genotype_encode)

        # Generate comparison prompt dynamically from criteria
        self.COMPARISON_PROMPT = generate_comparison_prompt(
            self.ITEM_TYPE,
            self.COMPARISON_CRITERIA
        )

        # Oracle prompts will be set by the loader

    def _build_requirements_digest(self, max_lines: int = 6) -> str:
        """Build a short bullet digest of special requirements for lightweight injection."""
        requirements = self.template.special_requirements or ""
        if not requirements.strip():
            return ""

        candidates = []
        for line in requirements.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lowered = stripped.lower()
            if lowered.startswith("format your response as"):
                continue
            if lowered.startswith("title:") or lowered.startswith("content:"):
                continue
            cleaned = stripped.lstrip("- ").strip()
            if cleaned:
                candidates.append(cleaned)

        if not candidates:
            return ""

        unique = list(dict.fromkeys(candidates))[:max_lines]
        return "\n".join(f"- {item}" for item in unique)

    def _interpolate_prompt(self, prompt_text: str) -> str:
        """
        Interpolate template-specific requirements into prompt text
        """
        result = prompt_text

        if self.template.special_requirements and '{requirements}' in result:
            result = result.replace('{requirements}', self.template.special_requirements)
        if self.REQUIREMENTS_DIGEST and '{requirements_digest}' in result:
            result = result.replace('{requirements_digest}', self.REQUIREMENTS_DIGEST)

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
