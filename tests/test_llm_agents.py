import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Critic, Oracle


class DummyPrompts:
    ITEM_TYPE = "idea"
    COMPARISON_CRITERIA = ["quality"]
    COMPARISON_PROMPT = ""
    ORACLE_INSTRUCTION = ""
    ORACLE_FORMAT_INSTRUCTIONS = ""
    ORACLE_MAIN_PROMPT = "{mode_instruction}{format_instructions}{example_idea_prompts}"
    IDEA_PROMPT = ""
    EXAMPLE_IDEA_PROMPTS = ""


def create_critic():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Critic()


def create_oracle():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Oracle()


def test_elo_update_cases():
    critic = create_critic()
    elo_a, elo_b = critic._elo_update(1500, 1500, "A")
    assert round(elo_a, 1) == 1516.0
    assert round(elo_b, 1) == 1484.0

    elo_a, elo_b = critic._elo_update(1600, 1400, "B")
    assert elo_a < 1600
    assert elo_b > 1400

    elo_a_tie, elo_b_tie = critic._elo_update(1500, 1500, "tie")
    assert elo_a_tie == 1500
    assert elo_b_tie == 1500


def test_get_tournament_ranks_deterministic():
    critic = create_critic()
    ideas = ["idea0", "idea1"]
    with patch.object(Critic, 'compare_ideas', side_effect=["A", "B"]):
        ranks = critic.get_tournament_ranks(ideas, "airesearch", comparisons=2)
    assert ranks[0] > ranks[1]


def test_compare_ideas_response_parsing():
    critic = create_critic()
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Critic, 'generate_text', return_value="Result: B"):
        result = critic.compare_ideas({'title': 'A'}, {'title': 'B'}, "airesearch")
    assert result == "B"



def test_compare_ideas_response_parsing_tie():
    critic = create_critic()
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Critic, 'generate_text', return_value="Result: TIE"):
        result = critic.compare_ideas({'title': 'A'}, {'title': 'B'}, "airesearch")
    assert result == "tie"


def test_compare_ideas_response_parsing_simple():
    critic = create_critic()
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Critic, 'generate_text', return_value="A"):
        result = critic.compare_ideas({'title': 'A'}, {'title': 'B'}, "airesearch")
    assert result == "A"


def test_compare_ideas_response_parsing_complex():
    critic = create_critic()
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Critic, 'generate_text', return_value="I think B is better"):
        result = critic.compare_ideas({'title': 'A'}, {'title': 'B'}, "airesearch")
    assert result == "B"


def test_oracle_response_parsing():
    oracle = create_oracle()
    response_text = """
=== ORACLE ANALYSIS ===
This is the analysis.

=== IDEA PROMPT ===
This is the new idea prompt.
"""
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Oracle, 'generate_text', return_value=response_text):
        result = oracle.analyze_and_diversify([], [], "airesearch")

    assert result["action"] == "replace"
    assert result["oracle_analysis"] == "This is the analysis."
    assert result["idea_prompt"] == "This is the new idea prompt."