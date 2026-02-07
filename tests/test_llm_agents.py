import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Critic, Oracle, Breeder


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
    with patch.object(Critic, 'compare_ideas', side_effect=["A"]):
        ranks = critic.get_tournament_ranks(ideas, "airesearch", rounds=1)
    assert ranks[0] > ranks[1]


def test_swiss_pairings_even_no_repeats():
    critic = create_critic()
    ranks = {i: 1500 for i in range(6)}
    match_history = set()
    bye_counts = {}
    all_pairs = set()

    for _ in range(3):
        pairs, bye = critic._generate_swiss_round_pairs(ranks, match_history, bye_counts)
        assert bye is None
        for a, b in pairs:
            key = tuple(sorted((a, b)))
            assert key not in all_pairs
            all_pairs.add(key)

    assert len(all_pairs) == 3 * (len(ranks) // 2)


def test_swiss_pairings_odd_rotates_byes():
    critic = create_critic()
    ranks = {i: 1500 for i in range(5)}
    match_history = set()
    bye_counts = {}
    bye_history = set()

    for _ in range(2):
        pairs, bye = critic._generate_swiss_round_pairs(ranks, match_history, bye_counts)
        assert len(pairs) == len(ranks) // 2
        assert bye is not None
        bye_history.add(bye)

    assert len(bye_history) == 2


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


def test_multi_tournament_segmentation_resets_pairing_state():
    critic = create_critic()
    ideas = ["idea0", "idea1", "idea2", "idea3"]
    observed_match_history_sizes = []

    original_pairer = critic._generate_swiss_round_pairs

    def wrapped_pairer(ranks, match_history, bye_counts):
        observed_match_history_sizes.append(len(match_history))
        return original_pairer(ranks, match_history, bye_counts)

    critic._generate_swiss_round_pairs = wrapped_pairer

    with patch.object(Critic, "compare_ideas", return_value="A"):
        critic.get_tournament_ranks(
            ideas,
            "airesearch",
            rounds=4,
            full_tournament_rounds=2,
        )

    assert observed_match_history_sizes[0] == 0
    assert observed_match_history_sizes[1] > 0
    assert observed_match_history_sizes[2] == 0


def test_breeder_keeps_mutation_context_per_child():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        breeder = Breeder(mutation_rate=0.5)

    fake_ideator = SimpleNamespace(
        generate_context=patch,
        generate_context_from_parents=lambda parent_genotypes, mutation_rate, mutation_context_pool: "ctx",
        generate_idea_from_context=lambda context_pool, idea_type: ("child idea", "child prompt"),
    )

    # Replace methods with deterministic fakes
    fake_ideator.generate_context = lambda idea_type: f"mutation-{idea_type}"
    breeder._get_thread_ideator = lambda: fake_ideator
    breeder.encode_to_genotype = lambda parent, idea_type: "gene"

    parents = [
        {"id": "p1", "idea": "idea1"},
        {"id": "p2", "idea": "idea2"},
    ]

    # Track mutation-context calls across two child generations
    calls = {"count": 0}
    def counting_generate_context(idea_type):
        calls["count"] += 1
        return f"mutation-{calls['count']}"

    fake_ideator.generate_context = counting_generate_context

    breeder.breed(parents, "airesearch")
    breeder.breed(parents, "airesearch")

    assert calls["count"] == 2


def test_oracle_response_parsing():
    oracle = create_oracle()
    response_text = """
=== ORACLE ANALYSIS ===
This is the analysis.

=== IDEA PROMPT ===
This is the new idea prompt.
"""
    with patch('idea.llm.get_prompts', return_value=DummyPrompts()),         patch.object(Oracle, 'generate_text', return_value=response_text):
        result = oracle.analyze_and_diversify([], "airesearch")

    assert result["action"] == "replace"
    assert result["oracle_analysis"] == "This is the analysis."
    assert result["idea_prompt"] == "This is the new idea prompt."
