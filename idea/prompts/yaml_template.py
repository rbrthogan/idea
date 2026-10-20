"""
YAML Template Wrapper - makes YAML templates behave like Python modules
"""

from typing import List
from .validation import PromptTemplate
from idea.template_runtime import compile_prompt, requirements_text, render_prompt, add_oracle_guidance, bind_prompt


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

    source = """You are an expert evaluator of {item_type}. You will be presented with two {item_type}, and your task is to determine which one is better.
{item_type} A:
Title: {idea_a_title}
{idea_a_content}

{item_type} B:
Title: {idea_b_title}
{idea_b_content}

Evaluate both {item_type}s based on the following criteria:
{criteria_text}

Criterion 1 is the most important.

After your evaluation, respond with exactly one of these three options:
- "Result: A" if {item_type} A is better
- "Result: B" if {item_type} B is better
- "Result: tie" if both {item_type}s are approximately equal in quality

First pick a winner along each criterion (A or B) -- brief or no explanation required. Then provide 1-2 sentence summary of your evaluation. Then, provide your final verdict.
Your response must contain exactly one of these three phrases"""
    return bind_prompt(source, item_type=item_type, criteria_text=criteria_text)


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
        self.CONTEXT_PROMPT = compile_prompt(self.template.prompts.context, self.template, "context")
        self.SPECIFIC_PROMPT = compile_prompt(self.template.prompts.specific_prompt, self.template, "direction")

        self.FORMAT_PROMPT = compile_prompt(self.template.prompts.format, self.template, "format")
        self.CRITIQUE_PROMPT = compile_prompt(self.template.prompts.critique, self.template, "critique")
        self.REFINE_PROMPT = compile_prompt(self.template.prompts.refine, self.template, "refine")
        self.BREED_PROMPT = compile_prompt(self.template.prompts.breed, self.template, "breed")

        self.GENOTYPE_ENCODE_PROMPT = compile_prompt(self.template.prompts.genotype_encode, self.template, "genotype")
        self.ORACLE_CONSTRAINTS = (
            self._interpolate_prompt(self.template.prompts.oracle_constraints)
            if self.template.prompts.oracle_constraints is not None else None
        )
        self.EXAMPLE_IDEA_PROMPTS = self._interpolate_prompt(self.template.prompts.example_idea_prompts or "")

        comparison = (self.template.prompts.comparison_prompt or self.template.comparison_prompt
                      or generate_comparison_prompt(self.ITEM_TYPE, self.COMPARISON_CRITERIA))
        # Older custom comparison prose may omit candidate slots entirely.
        # Every judge must still receive the complete candidates it evaluates.
        import re
        for label in ("a", "b"):
            has = lambda name: re.search(r"(?<!\{)\{" + name + r"\}(?!\})", comparison)
            if not has(f"idea_{label}"):
                if not has(f"idea_{label}_title"):
                    comparison += f"\n\nCandidate {label.upper()} title: {{idea_{label}_title}}"
                if not has(f"idea_{label}_content"):
                    comparison += f"\nCandidate {label.upper()} body:\n{{idea_{label}_content}}"
        self.COMPARISON_PROMPT = compile_prompt(comparison, self.template, "comparison")

        # Oracle prompts will be set by the loader

    def _build_requirements_digest(self, max_lines: int = 6) -> str:
        """Legacy alias, now complete so later constraints cannot disappear."""
        return requirements_text(self.template)

    def _interpolate_prompt(self, prompt_text: str) -> str:
        """
        Interpolate template-specific requirements into prompt text
        """
        return compile_prompt(prompt_text, self.template, "auxiliary", inject=False)

    def build_generation_prompt(self, direction: str) -> str:
        return compile_prompt("{direction}", self.template, "generate").format(direction=direction)

    def build_breeding_prompt(self, direction: str, context_pool: str) -> str:
        ideas = ("Sampled parent/mutation concepts (only the concepts retained by selection):\n"
                 f"{context_pool}\n\nChild direction:\n{direction}")
        return render_prompt(self.BREED_PROMPT, ideas=ideas)

    def build_oracle_prompt(self, rendered_prompt: str) -> str:
        return add_oracle_guidance(rendered_prompt, self.template)

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
