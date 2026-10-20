"""
Integration tests for YAML templates with the LLM system
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.prompts.loader import get_prompts


class TestLLMSystemIntegration:
    """Test that YAML templates integrate properly with the LLM system"""

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_yaml_template_llm_compatibility(self, template_name):
        """Test that YAML templates are compatible with LLM agents"""
        prompts = get_prompts(template_name, use_yaml=True)

        # Test that all required attributes are available for LLM agents
        required_attrs = [
            'ITEM_TYPE', 'CONTEXT_PROMPT', 'SPECIFIC_PROMPT',
            'FORMAT_PROMPT', 'CRITIQUE_PROMPT', 'REFINE_PROMPT', 'BREED_PROMPT',
            'COMPARISON_CRITERIA', 'COMPARISON_PROMPT'
        ]

        for attr in required_attrs:
            assert hasattr(prompts, attr), f"Missing {attr}"
            value = getattr(prompts, attr)
            assert value is not None, f"{attr} is None"

            if isinstance(value, str):
                assert len(value) > 0, f"{attr} is empty string"
            elif isinstance(value, list):
                assert len(value) > 0, f"{attr} is empty list"
        assert not hasattr(prompts, 'IDEA_PROMPT')


class TestPromptInterpolation:
    """Test that prompt interpolation works correctly"""

    def test_structured_drabble_contract_reaches_refine_and_breed(self):
        """Length and content rules come from one structured definition."""
        prompts = get_prompts('drabble', use_yaml=True)
        for prompt in (prompts.REFINE_PROMPT, prompts.BREED_PROMPT):
            assert 'Body length: exactly 100 words, excluding the title.' in prompt
            assert all(rule in prompt for rule in prompts.template.requirements)
            assert prompts.template.output.content in prompt

    def test_game_design_structured_requirements_reach_refine_and_breed(self):
        """All game rules survive without manual requirements placeholders."""
        prompts = get_prompts('game_design', use_yaml=True)
        for prompt in (prompts.REFINE_PROMPT, prompts.BREED_PROMPT):
            assert all(rule in prompt for rule in prompts.template.requirements)
            assert prompts.template.output.content in prompt
            assert '{requirements}' not in prompt

    def test_airesearch_prompts_load_without_idea_prompt(self):
        """Test that airesearch template loads without the deprecated idea prompt"""
        prompts = get_prompts('airesearch', use_yaml=True)

        # Just verify the prompts are loaded correctly
        assert not hasattr(prompts, 'IDEA_PROMPT')
        assert len(prompts.SPECIFIC_PROMPT) > 0
        assert len(prompts.REFINE_PROMPT) > 0
        assert len(prompts.BREED_PROMPT) > 0


class TestPromptPlaceholders:
    """Test that prompt placeholders are correctly present"""

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_required_placeholders_present(self, template_name):
        """Test that all required placeholders are present in prompts"""
        prompts = get_prompts(template_name, use_yaml=True)

        # Test placeholder presence
        assert '{input_text}' in prompts.FORMAT_PROMPT
        assert '{idea}' in prompts.CRITIQUE_PROMPT
        assert '{idea}' in prompts.REFINE_PROMPT
        assert '{critique}' in prompts.REFINE_PROMPT
        assert '{ideas}' in prompts.BREED_PROMPT

    def test_no_unresolved_placeholders(self):
        """Test that there are no unresolved placeholders in interpolated prompts"""
        templates = ["drabble", "airesearch", "game_design"]

        for template_name in templates:
            prompts = get_prompts(template_name, use_yaml=True)

            # Check that template-specific placeholders were resolved
            # (but keep the required runtime placeholders)
            interpolated_prompts = [
                prompts.SPECIFIC_PROMPT,
                prompts.REFINE_PROMPT,
                prompts.BREED_PROMPT
            ]

            for prompt in interpolated_prompts:
                # These should not contain unresolved template placeholders
                assert '{requirements}' not in prompt or template_name == 'airesearch', f"Unresolved requirements in {template_name}"


class TestSystemCompatibility:
    """Test compatibility with the existing evolution system"""

    def test_item_type_consistency(self):
        """Test that item types are consistent and appropriate"""
        expected_item_types = {
            'drabble': 'stories',
            'airesearch': 'GPU poor AI research proposals',
            'game_design': 'game designs'
        }

        for template_name, expected_type in expected_item_types.items():
            prompts = get_prompts(template_name, use_yaml=True)
            assert prompts.ITEM_TYPE == expected_type

    def test_comparison_criteria_format(self):
        """Test that comparison criteria are in the expected format"""
        for template_name in ["drabble", "airesearch", "game_design"]:
            prompts = get_prompts(template_name, use_yaml=True)
            criteria = prompts.COMPARISON_CRITERIA

            assert isinstance(criteria, list)
            assert len(criteria) >= 3  # Should have at least 3 criteria

            for criterion in criteria:
                assert isinstance(criterion, str)
                assert len(criterion.strip()) > 5  # Should be meaningful descriptions
