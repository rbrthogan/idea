"""
Utility functions for loading prompts based on idea type
"""

from importlib import import_module

def get_prompts(idea_type):
    """
    Load prompts for the specified idea type

    Args:
        idea_type (str): Type of idea (airesearch, game_design, drabble)

    Returns:
        module: Module containing the prompts for the specified idea type
    """
    try:
        return import_module(f"idea.prompts.{idea_type}")
    except ImportError:
        raise ValueError(f"No prompts found for idea type: {idea_type}")

def get_field_name(idea_type):
    """
    Get the field name for the specified idea type

    Args:
        idea_type (str): Type of idea

    Returns:
        str: Field name for the specified idea type
    """
    field_map = {
        "airesearch": "AI Research",
        "game_design": "2D arcade game design",
        "drabble": "Creative writing"
    }
    return field_map.get(idea_type, "Unknown field")