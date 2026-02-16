import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Breeder, Critic, Ideator, Oracle


class DummyPrompts:
    ITEM_TYPE = "idea"
    COMPARISON_CRITERIA = ["quality"]
    COMPARISON_PROMPT = ""
    ORACLE_INSTRUCTION = ""
    ORACLE_FORMAT_INSTRUCTIONS = ""
    ORACLE_MAIN_PROMPT = "{mode_instruction}{format_instructions}{example_idea_prompts}"
    IDEA_PROMPT = ""
    SPECIFIC_PROMPT = "{context_pool}"
    BREED_PROMPT = "{ideas}"
    EXAMPLE_IDEA_PROMPTS = ""


def create_critic():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Critic()


def create_oracle():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Oracle()


def create_ideator(**kwargs):
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Ideator(**kwargs)


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


def test_context_from_parents_uses_weighted_parent_mix_and_non_overlapping_mutation():
    ideator = create_ideator()
    ideator._random_sample = lambda population, k: list(population)[:k]
    ideator._random_uniform = lambda _a, _b: 0.5
    ideator._random_shuffle = lambda _seq: None

    context_pool = ideator.generate_context_from_parents(
        parent_genotypes=[
            "alpha; beta; gamma; theta",
            "beta; delta; epsilon; zeta",
        ],
        mutation_rate=0.5,
        mutation_context_pool="beta, omega",
    )
    concepts = [c.strip() for c in context_pool.split(",") if c.strip()]

    # target_child_size = 4, primary ratio = 50% => 2 from parent 1, then 2 from parent 2.
    # secondary overlap on "beta" is filtered and not backfilled, then one mutation is injected.
    assert concepts == ["alpha", "beta", "delta", "omega"]


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


def test_generate_context_parses_line_based_concepts():
    ideator = create_ideator()
    llm_response = """CONCEPTS:
- Base=Snake | Twist=wrapping + ghost trail | Constraints=grid-only / no sprites
- Base=Tetris | Twist=rotating board | Constraints=single screen / geometric blocks
- Base=Pong | Twist=timed gates | Constraints=one ball / deterministic speed
- Base=Frogger | Twist=lane swap | Constraints=discrete lanes / keyboard only
"""
    with patch.object(Ideator, "generate_text", return_value=llm_response):
        context_pool = ideator.generate_context("game_design")

    concepts = [c.strip() for c in context_pool.split(",") if c.strip()]
    assert concepts
    assert all("Base=" in concept for concept in concepts)


def test_generate_idea_from_context_appends_template_constraints():
    ideator = create_ideator()
    prompts = SimpleNamespace(
        SPECIFIC_PROMPT="SPEC:{context_pool}",
        IDEA_PROMPT="UNUSED",
        BREED_PROMPT="UNUSED",
        template=SimpleNamespace(special_requirements="must include title and content"),
    )

    with patch("idea.llm.get_prompts", return_value=prompts), patch.object(
        Ideator,
        "generate_text",
        side_effect=["specific direction", "final idea"],
    ) as mocked_generate:
        idea, specific = ideator.generate_idea_from_context("alpha, beta", "drabble")

    assert idea == "final idea"
    assert specific == "specific direction"
    generation_prompt = mocked_generate.call_args_list[1].args[0]
    assert generation_prompt == (
        "specific direction\n\n"
        "Constraints:\n"
        "must include title and content"
    )


def test_generate_idea_from_context_without_requirements_uses_specific_prompt_only():
    ideator = create_ideator()
    prompts = SimpleNamespace(
        SPECIFIC_PROMPT="SPEC:{context_pool}",
        IDEA_PROMPT="UNUSED",
        BREED_PROMPT="UNUSED",
    )

    with patch("idea.llm.get_prompts", return_value=prompts), patch.object(
        Ideator,
        "generate_text",
        side_effect=["specific direction", "bred idea"],
    ) as mocked_generate:
        idea, _specific = ideator.generate_idea_from_context("alpha, beta", "drabble")

    assert idea == "bred idea"
    generation_prompt = mocked_generate.call_args_list[1].args[0]
    assert generation_prompt == "specific direction"


def test_seed_context_pool_size_override_controls_context_call_count():
    ideator = create_ideator(seed_context_pool_size=4)
    with patch.object(Ideator, "generate_context", return_value="alpha, beta, gamma") as mocked_context:
        concept_bank = ideator._build_seed_context_bank(n=5, idea_type="drabble")

    assert mocked_context.call_count == 4
    assert concept_bank == ["alpha", "beta", "gamma"]


def test_diagnostic_events_store_raw_details():
    ideator = create_ideator()
    ideator._increment_diagnostic("generation_errors", detail="timeout while generating")

    events = ideator.get_diagnostic_events()
    assert len(events) == 1
    assert events[0]["key"] == "generation_errors"
    assert "timeout" in events[0]["detail"]
