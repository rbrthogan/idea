"""
Unit tests for YAML prompt templates
"""

import pytest
import sys
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.prompts.loader import get_prompts, list_available_templates, validate_template
from idea.prompts.validation import TemplateValidator
from idea.prompts.yaml_template import YAMLTemplateWrapper
import yaml


class TestYAMLTemplateLoading:
    """Test loading YAML templates"""

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_yaml_template_loads_successfully(self, template_name):
        """Test that YAML templates can be loaded"""
        prompts = get_prompts(template_name, use_yaml=True)
        assert prompts is not None
        assert hasattr(prompts, 'ITEM_TYPE')
        assert hasattr(prompts, 'COMPARISON_CRITERIA')
        assert hasattr(prompts, 'COMPARISON_PROMPT')

    @pytest.mark.parametrize("template_name", ["drabble", "airesearch", "game_design"])
    def test_required_prompts_available(self, template_name):
        """Test that all required prompts are available"""
        prompts = get_prompts(template_name, use_yaml=True)

        required_prompts = [
            'CONTEXT_PROMPT', 'SPECIFIC_PROMPT',
            'FORMAT_PROMPT', 'CRITIQUE_PROMPT', 'REFINE_PROMPT', 'BREED_PROMPT',
            'COMPARISON_PROMPT'
        ]

        for prompt_name in required_prompts:
            assert hasattr(prompts, prompt_name), f"Missing {prompt_name}"
            prompt_text = getattr(prompts, prompt_name)
            assert isinstance(prompt_text, str), f"{prompt_name} should be a string"
            assert len(prompt_text) > 0, f"{prompt_name} should not be empty"
        assert not hasattr(prompts, 'IDEA_PROMPT')

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

        assert not hasattr(prompts, 'IDEA_PROMPT')


    def test_game_design_special_requirements_interpolation(self):
        """Test that game design special requirements are properly interpolated"""
        prompts = get_prompts('game_design', use_yaml=True)

        assert not hasattr(prompts, 'IDEA_PROMPT')


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
                "specific_prompt": "Direction builder prompt",
                "format": "Format prompt missing input_text placeholder",  # Missing {input_text}
                "critique": "Critique prompt missing idea placeholder",  # Missing {idea}
                "refine": "Refine prompt missing placeholders",  # Missing {idea} and {critique}
                "breed": "Breed prompt missing ideas placeholder",  # Missing {ideas}
                "genotype_encode": "Genotype encode prompt"
            },
            "comparison_criteria": ["test"]
        }

        # Should validate successfully but have warnings
        template = TemplateValidator.validate_dict(invalid_template_data)
        warnings = TemplateValidator.check_prompt_interpolation(template)

        assert len(warnings) > 0, "Should have warnings for missing placeholders"
        assert "Format prompt missing {input_text} placeholder" in warnings
        assert "Critique prompt missing {idea} placeholder" in warnings
        assert "Refine prompt missing {idea} placeholder" in warnings
        assert "Refine prompt missing {critique} placeholder" in warnings
        assert "Breed prompt missing {ideas} placeholder" in warnings


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

    def test_missing_comparison_prompt_uses_default(self, tmp_path):
        """Templates without comparison_prompt should still load with a default"""
        template_path = Path('idea/prompts/templates/drabble.yaml')
        data = yaml.safe_load(template_path.read_text())
        if 'comparison_prompt' in data.get('prompts', {}):
            del data['prompts']['comparison_prompt']

        template = TemplateValidator.validate_dict(data)
        wrapper = YAMLTemplateWrapper(template)

        assert hasattr(wrapper, 'COMPARISON_PROMPT')
        assert isinstance(wrapper.COMPARISON_PROMPT, str)
        assert len(wrapper.COMPARISON_PROMPT) > 0

    def test_comparison_criteria_are_optional_for_non_llm_evaluators(self):
        """A prompt template should not force every run to use an LLM judge."""
        template_path = Path('idea/prompts/templates/drabble.yaml')
        data = yaml.safe_load(template_path.read_text())
        data.pop('comparison_criteria', None)

        template = TemplateValidator.validate_dict(data)
        wrapper = YAMLTemplateWrapper(template)

        assert template.comparison_criteria == []
        assert wrapper.COMPARISON_CRITERIA == []
        # The compatibility wrapper still exposes a default prompt for callers
        # that explicitly select the legacy LLM evaluator.
        assert isinstance(wrapper.COMPARISON_PROMPT, str)
        assert wrapper.COMPARISON_PROMPT


class TestPrimeVisualizationTemplate:
    """Contract tests for the artifact-ready prime experiment preset."""

    def test_prime_visualization_template_validates_and_loads(self):
        is_valid, warnings = validate_template('prime_visualization')
        assert is_valid
        assert warnings == []

        prompts = get_prompts('prime_visualization', use_yaml=True)
        assert prompts.ITEM_TYPE == 'prime visualization experiment proposals'
        assert prompts.get_info()['version'] == '1.2.0'

        required_prompts = [
            'CONTEXT_PROMPT',
            'SPECIFIC_PROMPT',
            'FORMAT_PROMPT',
            'CRITIQUE_PROMPT',
            'REFINE_PROMPT',
            'BREED_PROMPT',
            'GENOTYPE_ENCODE_PROMPT',
        ]
        for prompt_name in required_prompts:
            assert getattr(prompts, prompt_name).strip()

    def test_prime_visualization_contract_is_explicit_and_honest(self):
        template_path = Path('idea/prompts/templates/prime_visualization.yaml')
        data = yaml.safe_load(template_path.read_text())
        requirements = data['special_requirements']
        all_prompts = '\n'.join(data['prompts'].values())

        for contract_term in [
            'Ulam square spiral',
            'Sacks number spiral',
            'residue grids',
            'cyclic multiplication maps',
            'prime-race curves',
            'Pascal/binomial arrays',
            'recursive dot or polygon groupings',
            'falsifiable hypothesis',
            'deterministic sieve',
            'fixed seed of 0',
            'visualization.png',
            'metadata.json',
            'result.json',
            'media_type="image/png"',
        ]:
            assert contract_term in requirements

        assert 'never claim that code was executed' in requirements
        assert 'Do not claim that the experiment has been run' in all_prompts
        assert 'seeds, not recipes' in requirements
        assert 'At least 48 of the 60' in data['prompts']['context']
        assert 'At least 40 must use a single mapping rule' in data['prompts']['context']
        assert 'select exactly one primary ingredient' in data['prompts']['specific_prompt']
        assert 'one or two plain-language sentences' in requirements
        assert any(
            'conceptual simplicity and elegance' in criterion
            for criterion in data['comparison_criteria']
        )
        complete_template_text = '\n'.join([
            requirements,
            all_prompts,
            *data['comparison_criteria'],
        ])
        assert complete_template_text.count('Ulam') <= 2
        assert '{context_pool}' in data['prompts']['specific_prompt']
        assert '{input_text}' in data['prompts']['format']

    def test_prime_visualization_is_catalogued_with_opt_in_llm_criteria(self):
        templates = list_available_templates()
        template_info = templates['prime_visualization']
        assert template_info['type'] == 'yaml'
        assert template_info['name'] == 'Prime Visualization Experiments'
        assert template_info['warnings'] == []

        prompts = get_prompts('prime_visualization', use_yaml=True)
        assert len(prompts.COMPARISON_CRITERIA) >= 5
        assert any('cited precedents' in criterion for criterion in prompts.COMPARISON_CRITERIA)
        assert any('abstracted precedent principle' in criterion for criterion in prompts.COMPARISON_CRITERIA)
        assert any('fabricated' in criterion for criterion in prompts.COMPARISON_CRITERIA)
