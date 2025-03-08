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
