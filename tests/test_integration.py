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
            'ITEM_TYPE', 'CONTEXT_PROMPT', 'IDEA_PROMPT', 'NEW_IDEA_PROMPT',
            'FORMAT_PROMPT', 'CRITIQUE_PROMPT', 'REFINE_PROMPT', 'BREED_PROMPT',
            'COMPARISON_CRITERIA'
        ]

        for attr in required_attrs:
            assert hasattr(prompts, attr), f"Missing {attr}"
            value = getattr(prompts, attr)
            assert value is not None, f"{attr} is None"

            if isinstance(value, str):
                assert len(value) > 0, f"{attr} is empty string"
            elif isinstance(value, list):
                assert len(value) > 0, f"{attr} is empty list"

    def test_drabble_legacy_compatibility(self):
        """Test that drabble template maintains legacy compatibility"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Should have legacy DRABBLE_FORMAT_PROMPT attribute
        assert hasattr(prompts, 'DRABBLE_FORMAT_PROMPT')
        assert isinstance(prompts.DRABBLE_FORMAT_PROMPT, str)
        assert len(prompts.DRABBLE_FORMAT_PROMPT) > 0

    def test_template_metadata_access(self):
        """Test that template metadata is accessible"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Test template metadata properties
        if hasattr(prompts, 'get_info'):
            info = prompts.get_info()
            assert isinstance(info, dict)

            # Test individual properties
            assert hasattr(prompts, 'name')
            assert hasattr(prompts, 'description')
            assert hasattr(prompts, 'version')
            assert hasattr(prompts, 'author')

            assert isinstance(prompts.name, str)
            assert isinstance(prompts.description, str)
            assert isinstance(prompts.version, str)
            assert isinstance(prompts.author, str)


class TestPromptInterpolation:
    """Test that prompt interpolation works correctly"""

    def test_format_requirements_interpolation(self):
        """Test that format_requirements are properly interpolated"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Check that format_requirements was interpolated into relevant prompts
        idea_prompt = prompts.IDEA_PROMPT
        assert 'A drabble is a short work of fiction' in idea_prompt

        new_idea_prompt = prompts.NEW_IDEA_PROMPT
        assert 'A drabble is a short work of fiction' in new_idea_prompt

        refine_prompt = prompts.REFINE_PROMPT
        assert 'A drabble is a short work of fiction' in refine_prompt

        breed_prompt = prompts.BREED_PROMPT
        assert 'A drabble is a short work of fiction' in breed_prompt

    def test_design_requirements_interpolation(self):
        """Test that design_requirements are properly interpolated"""
        prompts = get_prompts('game_design', use_yaml=True)

        # Check that design_requirements was interpolated into relevant prompts
        idea_prompt = prompts.IDEA_PROMPT
        assert 'The game should be simple enough' in idea_prompt

        new_idea_prompt = prompts.NEW_IDEA_PROMPT
        assert 'The game should be simple enough' in new_idea_prompt

        refine_prompt = prompts.REFINE_PROMPT
        assert 'The game should be simple enough' in refine_prompt

        breed_prompt = prompts.BREED_PROMPT
        assert 'The game should be simple enough' in breed_prompt

    def test_airesearch_no_special_requirements(self):
        """Test that airesearch template works without special requirements"""
        prompts = get_prompts('airesearch', use_yaml=True)

        # AI research template doesn't have special requirements
        # Just verify the prompts are loaded correctly
        assert len(prompts.IDEA_PROMPT) > 0
        assert len(prompts.NEW_IDEA_PROMPT) > 0
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
                prompts.IDEA_PROMPT,
                prompts.NEW_IDEA_PROMPT,
                prompts.REFINE_PROMPT,
                prompts.BREED_PROMPT
            ]

            for prompt in interpolated_prompts:
                # These should not contain unresolved template placeholders
                assert '{format_requirements}' not in prompt, f"Unresolved format_requirements in {template_name}"
                assert '{design_requirements}' not in prompt, f"Unresolved design_requirements in {template_name}"


class TestSystemCompatibility:
    """Test compatibility with the existing evolution system"""

    def test_item_type_consistency(self):
        """Test that item types are consistent and appropriate"""
        expected_item_types = {
            'drabble': 'stories',
            'airesearch': 'AI research proposals',
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