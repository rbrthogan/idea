"""
Utility functions for loading prompts based on idea type
Supports both YAML templates and legacy Python modules
"""

from importlib import import_module
from pathlib import Path
from typing import Optional
from .validation import validate_template_file
from .yaml_template import YAMLTemplateWrapper


def get_prompts(idea_type: str, use_yaml: bool = True):
    """
    Load prompts for the specified idea type

    Args:
        idea_type (str): Type of idea (airesearch [GPU poor AI research], game_design, drabble)
        use_yaml (bool): Whether to prefer YAML templates over Python modules

    Returns:
        module or YAMLTemplateWrapper: Module/wrapper containing the prompts for the specified idea type
    """

    # First try YAML templates if requested
    if use_yaml:
        yaml_path = _get_yaml_template_path(idea_type)
        if yaml_path and yaml_path.exists():
            try:
                template, warnings = validate_template_file(str(yaml_path))
                if warnings:
                    print(f"Template warnings for {idea_type}: {warnings}")

                # Load and merge Oracle prompts from core oracle template
                wrapper = YAMLTemplateWrapper(template)
                _load_oracle_prompts(wrapper)
                return wrapper
            except Exception as e:
                print(f"Failed to load YAML template for {idea_type}: {e}")
                print("Falling back to Python module...")

    # Fallback to Python modules (existing behavior)
    try:
        return import_module(f"idea.prompts.{idea_type}")
    except ImportError:
        raise ValueError(f"No prompts found for idea type: {idea_type}")


def _get_yaml_template_path(idea_type: str) -> Optional[Path]:
    """Get the path to a YAML template file"""
    # Get the directory where this loader.py file is located
    prompts_dir = Path(__file__).parent
    template_path = prompts_dir / "templates" / f"{idea_type}.yaml"
    return template_path


def list_available_templates() -> dict:
    """
    List all available prompt templates (both YAML and Python)

    Returns:
        dict: Template information keyed by idea_type
    """
    templates = {}
    prompts_dir = Path(__file__).parent

    # Check for YAML templates
    templates_dir = prompts_dir / "templates"
    if templates_dir.exists():
        for yaml_file in templates_dir.glob("*.yaml"):
            idea_type = yaml_file.stem
            try:
                template, warnings = validate_template_file(str(yaml_file))
                wrapper = YAMLTemplateWrapper(template)
                templates[idea_type] = {
                    'type': 'yaml',
                    'name': wrapper.name,
                    'description': wrapper.description,
                    'version': wrapper.version,
                    'author': wrapper.author,
                    'warnings': warnings,
                    'path': str(yaml_file)
                }
            except Exception as e:
                templates[idea_type] = {
                    'type': 'yaml',
                    'name': idea_type,
                    'description': 'Invalid template',
                    'error': str(e),
                    'path': str(yaml_file)
                }

    # Check for Python modules
    for py_file in prompts_dir.glob("*.py"):
        if py_file.name in ['__init__.py', 'loader.py', 'validation.py', 'yaml_template.py', 'oracle_prompts.py']:
            continue

        idea_type = py_file.stem
        if idea_type not in templates:  # Don't override YAML templates
            try:
                module = import_module(f"idea.prompts.{idea_type}")
                templates[idea_type] = {
                    'type': 'python',
                    'name': idea_type.replace('_', ' ').title(),
                    'description': getattr(module, '__doc__', '').strip() or f"Python module for {idea_type}",
                    'item_type': getattr(module, 'ITEM_TYPE', 'Unknown'),
                    'path': str(py_file)
                }
            except Exception as e:
                templates[idea_type] = {
                    'type': 'python',
                    'name': idea_type,
                    'description': 'Invalid module',
                    'error': str(e),
                    'path': str(py_file)
                }

    return templates


def _load_oracle_prompts(wrapper: YAMLTemplateWrapper):
    """
    Load Oracle prompts from the core oracle_prompts module and add them to the wrapper
    """
    try:
        from .oracle_prompts import (
            ORACLE_INSTRUCTION,
            ORACLE_FORMAT_INSTRUCTIONS,
            ORACLE_MAIN_PROMPT,
            ORACLE_CONSTRAINTS
        )

        # Add Oracle prompts to the wrapper
        wrapper.ORACLE_INSTRUCTION = ORACLE_INSTRUCTION
        wrapper.ORACLE_FORMAT_INSTRUCTIONS = ORACLE_FORMAT_INSTRUCTIONS
        wrapper.ORACLE_MAIN_PROMPT = ORACLE_MAIN_PROMPT

        # Check if the idea-specific template has Oracle constraints customization
        # For now, we'll use the default constraints but this allows future customization
        wrapper.ORACLE_CONSTRAINTS = ORACLE_CONSTRAINTS

    except Exception as e:
        print(f"Warning: Failed to load Oracle prompts: {e}")
        # Set empty Oracle prompts as fallback
        wrapper.ORACLE_INSTRUCTION = ''
        wrapper.ORACLE_FORMAT_INSTRUCTIONS = ''
        wrapper.ORACLE_MAIN_PROMPT = ''
        wrapper.ORACLE_CONSTRAINTS = ''


def validate_template(idea_type: str) -> tuple[bool, list]:
    """
    Validate a specific template

    Args:
        idea_type (str): The template to validate

    Returns:
        tuple: (is_valid, list_of_warnings_or_errors)
    """
    yaml_path = _get_yaml_template_path(idea_type)

    if yaml_path and yaml_path.exists():
        try:
            template, warnings = validate_template_file(str(yaml_path))
            return True, warnings
        except Exception as e:
            return False, [str(e)]
    else:
        # For Python modules, just try to import them
        try:
            module = import_module(f"idea.prompts.{idea_type}")
            # Basic check for required attributes
            required_attrs = ['ITEM_TYPE', 'CONTEXT_PROMPT', 'IDEA_PROMPT', 'COMPARISON_CRITERIA']
            missing = [attr for attr in required_attrs if not hasattr(module, attr)]
            if missing:
                return False, [f"Missing required attributes: {missing}"]
            return True, []
        except ImportError as e:
            return False, [f"Failed to import: {e}"]
