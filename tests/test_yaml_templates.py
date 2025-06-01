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


class TestYAMLTemplateValidation:
    """Test template validation"""

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_template_validation_passes(self, template_name):
        """Test that templates pass validation"""
        is_valid, warnings_or_errors = validate_template(template_name)
        assert is_valid, f"Template {template_name} failed validation: {warnings_or_errors}"

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_no_critical_validation_errors(self, template_name):
        """Test that templates have no critical validation errors"""
        is_valid, warnings_or_errors = validate_template(template_name)
        if not is_valid:
            pytest.fail(f"Template {template_name} has validation errors: {warnings_or_errors}")


class TestTemplateListing:
    """Test template listing functionality"""

    def test_template_listing_returns_templates(self):
        """Test that template listing returns available templates"""
        templates = list_available_templates()
        assert isinstance(templates, dict)
        assert len(templates) >= 3  # At least our 3 YAML templates

        # Check that our YAML templates are present
        expected_templates = ['drabble', 'airesearch', 'game_design']
        for template_name in expected_templates:
            assert template_name in templates
            template_info = templates[template_name]
            assert template_info['type'] == 'yaml'
            assert 'name' in template_info
            assert 'description' in template_info

    def test_template_listing_handles_errors_gracefully(self):
        """Test that template listing handles errors without crashing"""
        templates = list_available_templates()

        # Should not raise an exception even if some templates have errors
        for template_name, template_info in templates.items():
            assert isinstance(template_info, dict)
            assert 'type' in template_info


class TestYAMLOnlySystem:
    """Test the fully migrated YAML-only system"""

    def test_yaml_templates_are_default(self):
        """Test that YAML templates are loaded by default"""
        prompts = get_prompts('drabble')  # No use_yaml parameter, should default to True
        assert prompts is not None
        assert hasattr(prompts, 'ITEM_TYPE')
        assert prompts.ITEM_TYPE == 'stories'

        # Should be a YAML template wrapper
        assert hasattr(prompts, 'get_info')

    def test_all_templates_are_yaml(self):
        """Test that all available templates are now YAML-based"""
        templates = list_available_templates()

        for template_name, template_info in templates.items():
            if 'error' not in template_info:  # Skip any errored templates
                assert template_info['type'] == 'yaml', f"Template {template_name} is not YAML-based"

    def test_no_python_modules_remain(self):
        """Test that we've successfully migrated away from Python modules"""
        templates = list_available_templates()

        # Should not have any Python-type templates for our main template types
        main_templates = ['drabble', 'airesearch', 'game_design']
        for template_name in main_templates:
            if template_name in templates:
                assert templates[template_name]['type'] == 'yaml'


class TestSpecialFeatures:
    """Test special features like prompt interpolation"""

    def test_drabble_format_requirements_interpolation(self):
        """Test that drabble format requirements are properly interpolated"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Check that format requirements were interpolated into prompts
        idea_prompt = prompts.IDEA_PROMPT
        assert 'A drabble is a short work of fiction' in idea_prompt

        # Check legacy compatibility
        assert hasattr(prompts, 'DRABBLE_FORMAT_PROMPT')

    def test_game_design_requirements_interpolation(self):
        """Test that game design requirements are properly interpolated"""
        prompts = get_prompts('game_design', use_yaml=True)

        idea_prompt = prompts.IDEA_PROMPT
        assert 'The game should be simple enough' in idea_prompt

        # Check that design requirements attribute is available
        assert hasattr(prompts, 'DESIGN_REQUIREMENTS')

    def test_placeholder_validation(self):
        """Test that required placeholders are present in prompts"""
        prompts = get_prompts('drabble', use_yaml=True)

        # Check that required placeholders are present
        assert '{input_text}' in prompts.FORMAT_PROMPT
        assert '{idea}' in prompts.CRITIQUE_PROMPT
        assert '{idea}' in prompts.REFINE_PROMPT
        assert '{critique}' in prompts.REFINE_PROMPT
        assert '{ideas}' in prompts.BREED_PROMPT