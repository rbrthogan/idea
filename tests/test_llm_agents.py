import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Critic


class DummyPrompts:
    ITEM_TYPE = "idea"
    COMPARISON_CRITERIA = ["quality"]
    COMPARISON_PROMPT = ""
    REMOVE_WORST_IDEA_PROMPT = "{ideas}\nCriteria: {criteria}\nWorst Entry:"


def create_critic():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Critic()


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


def test_remove_worst_idea_parsing():
    critic = create_critic()
    ideas = ["a", "b", "c"]
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Critic, 'generate_text', return_value="Some text\nWorst Entry: 2"):
        remaining = critic.remove_worst_idea(ideas, "airesearch")
    assert remaining == ["a", "c"]


def test_compare_ideas_response_parsing():
    critic = create_critic()
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()), \
         patch.object(Critic, 'generate_text', return_value="Result: B"):
        result = critic.compare_ideas({'title': 'A'}, {'title': 'B'}, "airesearch")
    assert result == "B"

