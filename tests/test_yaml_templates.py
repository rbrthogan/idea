"""
Unit tests for YAML prompt templates
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.prompts.loader import get_prompts, list_available_templates, validate_template


class TestYAMLTemplateLoading:
    """Test loading YAML templates"""

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_yaml_template_loads_successfully(self, template_name):
        """Test that YAML templates can be loaded"""
        prompts = get_prompts(template_name, use_yaml=True)
        assert prompts is not None
        assert hasattr(prompts, 'ITEM_TYPE')
        assert hasattr(prompts, 'COMPARISON_CRITERIA')

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_required_prompts_available(self, template_name):
        """Test that all required prompts are available"""
        prompts = get_prompts(template_name, use_yaml=True)

        required_prompts = [
            'CONTEXT_PROMPT', 'IDEA_PROMPT', 'NEW_IDEA_PROMPT',
            'FORMAT_PROMPT', 'CRITIQUE_PROMPT', 'REFINE_PROMPT', 'BREED_PROMPT'
        ]

        for prompt_name in required_prompts:
            assert hasattr(prompts, prompt_name), f"Missing {prompt_name}"
            prompt_text = getattr(prompts, prompt_name)
            assert isinstance(prompt_text, str), f"{prompt_name} should be a string"
            assert len(prompt_text) > 0, f"{prompt_name} should not be empty"

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_comparison_criteria_present(self, template_name):
        """Test that comparison criteria are present and non-empty"""
        prompts = get_prompts(template_name, use_yaml=True)
        assert hasattr(prompts, 'COMPARISON_CRITERIA')
        criteria = prompts.COMPARISON_CRITERIA
        assert isinstance(criteria, list)
        assert len(criteria) > 0
        for criterion in criteria:
            assert isinstance(criterion, str)
            assert len(criterion.strip()) > 0

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_template_metadata_available(self, template_name):
        """Test that template metadata is available"""
        prompts = get_prompts(template_name, use_yaml=True)

        if hasattr(prompts, 'get_info'):
            info = prompts.get_info()
            assert 'name' in info
            assert 'version' in info
            assert 'author' in info
            assert 'created_date' in info
            assert 'item_type' in info

            # Basic validation of fields
            assert info['version'].count('.') == 2  # Semantic versioning
            assert len(info['name']) > 0
            assert len(info['author']) > 0


class TestSpecialFeatures:
    """Test special features like prompt interpolation"""

    def test_special_requirements_interpolation(self):
        """Test that special requirements are properly interpolated"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Check that special requirements were interpolated into prompts
        idea_prompt = prompts.IDEA_PROMPT
        assert 'A drabble is a short work of fiction' in idea_prompt

        # Check that special requirements attribute is available
        assert hasattr(prompts, 'SPECIAL_REQUIREMENTS')

    def test_game_design_special_requirements_interpolation(self):
        """Test that game design special requirements are properly interpolated"""
        prompts = get_prompts('game_design', use_yaml=True)

        idea_prompt = prompts.IDEA_PROMPT
        assert 'The game should be simple enough' in idea_prompt

        # Check that special requirements attribute is available
        assert hasattr(prompts, 'SPECIAL_REQUIREMENTS')

    def test_placeholder_validation(self):
        """Test that required placeholders are present in prompts"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Check that required placeholders are present
        assert '{input_text}' in prompts.FORMAT_PROMPT
        assert '{idea}' in prompts.CRITIQUE_PROMPT
        assert '{idea}' in prompts.REFINE_PROMPT
        assert '{critique}' in prompts.REFINE_PROMPT
        assert '{ideas}' in prompts.BREED_PROMPT


class TestTemplateValidation:
    """Test template validation functionality"""

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_template_validation_passes(self, template_name):
        """Test that existing templates pass validation"""
        is_valid, warnings = validate_template(template_name)
        assert is_valid, f"Template {template_name} should be valid"

    def test_validation_detects_missing_placeholders(self):
        """Test that validation detects missing required placeholders"""
        from idea.prompts.validation import TemplateValidator

        # Create invalid template with missing placeholders
        invalid_template_data = {
            "name": "Invalid Template",
            "description": "Test template",
            "version": "1.0.0",
            "author": "Test",
            "created_date": "2024-01-01",
            "metadata": {"item_type": "test"},
            "prompts": {
                "context": "Context prompt",
                "idea": "Idea prompt",
                "new_idea": "New idea prompt",
                "format": "Format prompt missing input_text placeholder",  # Missing {input_text}
                "critique": "Critique prompt missing idea placeholder",  # Missing {idea}
                "refine": "Refine prompt missing placeholders",  # Missing {idea} and {critique}
                "breed": "Breed prompt missing ideas placeholder"  # Missing {ideas}
            },
            "comparison_criteria": ["test"]
        }

        # Should validate successfully but have warnings
        template = TemplateValidator.validate_dict(invalid_template_data)
        warnings = TemplateValidator.check_prompt_interpolation(template)

        assert len(warnings) > 0, "Should have warnings for missing placeholders"


class TestTemplateManagerCompatibility:
    """Test compatibility with template manager"""

    def test_list_templates_includes_yaml(self):
        """Test that list_available_templates includes YAML templates"""
        templates = list_available_templates()

        # Should include our test templates
        assert 'drabble' in templates
        assert 'airesearch' in templates
        assert 'game_design' in templates

        # Templates should have required metadata
        for template_id in ['drabble', 'airesearch', 'game_design']:
            template_info = templates[template_id]
            assert 'name' in template_info
            assert 'type' in template_info
            assert template_info['type'] == 'yaml'