import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Breeder, Critic, Formatter, Ideator, Oracle


class DummyPrompts:
    ITEM_TYPE = "idea"
    COMPARISON_CRITERIA = ["quality"]
    COMPARISON_PROMPT = ""
    ORACLE_INSTRUCTION = ""
    ORACLE_FORMAT_INSTRUCTIONS = ""
    ORACLE_MAIN_PROMPT = "{mode_instruction}{format_instructions}{example_idea_prompts}"
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


def create_formatter():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        return Formatter()


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


def test_tournament_ranks_start_from_persisted_global_elo():
    critic = create_critic()
    ideas = [
        {"id": "high", "idea": "high", "selection_metrics": {"elo": 1800}},
        {"id": "low", "idea": "low", "selection_metrics": {"elo": 1200}},
    ]

    with patch.object(Critic, 'compare_ideas', side_effect=["tie"]):
        ranks = critic.get_tournament_ranks(ideas, "airesearch", rounds=1)

    assert 1780 < ranks[0] < 1800
    assert 1200 < ranks[1] < 1220
    assert ranks[0] > ranks[1]


def test_single_idea_tournament_preserves_persisted_global_elo():
    critic = create_critic()
    ranks = critic.get_tournament_ranks(
        [{"id": "solo", "idea": "solo", "selection_metrics": {"elo": 1712}}],
        "airesearch",
        rounds=1,
    )

    assert ranks == {0: 1712.0}


def test_tournament_reuses_cached_pair_without_rejudging_reversed_order():
    critic = create_critic()
    idea_a = {"id": "a", "idea": {"title": "A", "content": "alpha"}}
    idea_b = {"id": "b", "idea": {"title": "B", "content": "beta"}}
    comparison_cache = {}
    details = []

    with patch.object(Critic, "compare_ideas", return_value="A") as compare:
        first_ranks = critic.get_tournament_ranks(
            [idea_a, idea_b],
            "airesearch",
            rounds=1,
            comparison_cache=comparison_cache,
            details=details,
        )

    assert compare.call_count == 1
    assert first_ranks[0] > first_ranks[1]
    assert comparison_cache
    assert details[0]["pairs"][0]["cached"] is False
    assert details[0]["pairs"][0]["winner_source"] == "llm"

    reversed_details = []
    with patch.object(Critic, "compare_ideas", side_effect=AssertionError("cache miss")) as compare:
        reversed_ranks = critic.get_tournament_ranks(
            [idea_b, idea_a],
            "airesearch",
            rounds=1,
            comparison_cache=comparison_cache,
            details=reversed_details,
        )

    assert compare.call_count == 0
    assert reversed_ranks[1] > reversed_ranks[0]
    assert reversed_details[0]["pairs"][0]["cached"] is True
    assert reversed_details[0]["pairs"][0]["winner_source"] == "cache"
    assert reversed_details[0]["pairs"][0]["winner"] == "B"


def test_tournament_records_evaluation_metadata_on_ideas():
    critic = create_critic()
    ideas = [
        {"id": "a", "idea": {"title": "A", "content": "alpha"}},
        {"id": "b", "idea": {"title": "B", "content": "beta"}},
    ]

    with patch.object(Critic, "compare_ideas", return_value="tie"):
        critic.get_tournament_ranks(
            ideas,
            "airesearch",
            rounds=1,
            comparison_cache={},
            evaluation_generation=3,
        )

    for idea in ideas:
        metadata = idea["rating_evaluations"]
        assert metadata["comparison_count"] == 1
        assert metadata["llm_comparison_count"] == 1
        assert metadata["cached_comparison_count"] == 0
        assert metadata["last_evaluated_generation"] == 3
        assert idea["auto_match_count"] == 1


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


def test_swiss_pairing_prefers_nearest_elo_unseen_partner():
    critic = create_critic()
    ranks = {0: 2000, 1: 1000, 2: 1500, 3: 1990}
    match_history = {(0, 3)}
    bye_counts = {}

    pairs, bye = critic._generate_swiss_round_pairs(ranks, match_history, bye_counts)

    assert bye is None
    assert (0, 2) in {tuple(sorted(pair)) for pair in pairs}


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


def test_tournament_stops_early_when_survivor_boundary_is_stable():
    critic = create_critic()
    ideas = [
        {"id": "a", "idea": "a", "selection_metrics": {"elo": 1800}},
        {"id": "b", "idea": "b", "selection_metrics": {"elo": 1700}},
        {"id": "c", "idea": "c", "selection_metrics": {"elo": 1300}},
        {"id": "d", "idea": "d", "selection_metrics": {"elo": 1200}},
    ]
    details = []

    with patch.dict("os.environ", {"ENABLE_PARALLEL_TOURNAMENT": "0"}), \
         patch.object(Critic, "compare_ideas", return_value="A") as compare:
        critic.get_tournament_ranks(
            ideas,
            "airesearch",
            rounds=5,
            details=details,
            early_stop_top_n=2,
            early_stop_min_rounds=2,
        )

    assert compare.call_count < 10
    assert len(details) == 2
    assert details[-1]["early_stopped"] is True
    assert details[-1]["early_stop_reason"] == "stable_survivor_boundary"


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


def test_breeder_returns_parent_genotypes_for_display():
    with patch('idea.llm.LLMWrapper._setup_provider', return_value=None):
        breeder = Breeder()

    fake_ideator = SimpleNamespace(
        generate_context_from_parents=lambda parent_genotypes, mutation_rate, mutation_context_pool: "ctx",
        generate_idea_from_context=lambda context_pool, idea_type: ("child idea", "child prompt"),
    )

    breeder._get_thread_ideator = lambda: fake_ideator
    breeder.encode_to_genotype = lambda parent, idea_type: f"gene-{parent['id']}"

    parents = [
        {"id": "p1", "idea": "idea1"},
        {"id": "p2", "idea": "idea2"},
    ]

    result = breeder.breed(parents, "airesearch")

    assert result["parent_genotypes"] == [
        {"parent_id": "p1", "genotype": "gene-p1"},
        {"parent_id": "p2", "genotype": "gene-p2"},
    ]


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


def test_generate_content_coerces_empty_text_to_no_response():
    ideator = create_ideator()

    class _FakeResponse:
        text = ""
        usage_metadata = None

    class _FakeClient:
        class models:
            @staticmethod
            def generate_content(**_kwargs):
                return _FakeResponse()

    with patch.object(ideator, "_get_client", return_value=_FakeClient()):
        result = ideator._generate_content("prompt", None, None, None)

    assert result == "No response."
    assert ideator.get_diagnostics().get("blocked_or_empty_responses", 0) >= 1


def test_gemini_25_level_preset_maps_to_budget_without_fallback_diagnostic():
    ideator = create_ideator(model_name="gemini-2.5-flash", thinking_level="high")
    captured = {}

    class _FakeResponse:
        text = "done"
        usage_metadata = None

    class _FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                captured.update(kwargs)
                return _FakeResponse()

    with patch.object(ideator, "_get_client", return_value=_FakeClient()):
        result = ideator._generate_content("prompt", None, None, None)

    assert result == "done"
    dumped_config = captured["config"].model_dump(by_alias=True, exclude_none=True)
    assert dumped_config["thinkingConfig"]["thinkingBudget"] == -1
    assert "thinking_level_budget_fallbacks" not in ideator.get_diagnostics()


def test_gemini_31_pro_sends_explicit_thinking_level():
    ideator = create_ideator(model_name="gemini-3.1-pro-preview", thinking_level="low")
    captured = {}

    class _FakeResponse:
        text = "done"
        usage_metadata = None

    class _FakeClient:
        class models:
            @staticmethod
            def generate_content(**kwargs):
                captured.update(kwargs)
                return _FakeResponse()

    with patch.object(ideator, "_get_client", return_value=_FakeClient()):
        result = ideator._generate_content("prompt", None, None, None)

    assert result == "done"
    dumped_config = captured["config"].model_dump(by_alias=True, exclude_none=True, mode="json")
    assert dumped_config["thinkingConfig"]["thinkingLevel"] == "LOW"
    assert "thinking_level_budget_fallbacks" not in ideator.get_diagnostics()
    assert "thinking_level_sdk_unsupported" not in ideator.get_diagnostics()


def test_generate_content_tracks_thinking_cached_and_tiered_costs():
    ideator = create_ideator(model_name="gemini-2.5-pro")

    class _FakeResponse:
        text = "done"
        usage_metadata = SimpleNamespace(
            prompt_token_count=210_000,
            cached_content_token_count=50_000,
            candidates_token_count=1_000,
            thoughts_token_count=2_000,
            total_token_count=213_000,
        )

    class _FakeClient:
        class models:
            @staticmethod
            def generate_content(**_kwargs):
                return _FakeResponse()

    with patch.object(ideator, "_get_client", return_value=_FakeClient()):
        result = ideator._generate_content("prompt", None, None, None)

    assert result == "done"
    assert ideator.input_token_count == 210_000
    assert ideator.cached_input_token_count == 50_000
    assert ideator.uncached_input_token_count == 160_000
    assert ideator.output_token_count == 1_000
    assert ideator.thought_token_count == 2_000
    assert ideator.total_token_count == 213_000

    expected_uncached_input_cost = 160_000 * 2.50 / 1_000_000
    expected_cached_input_cost = 50_000 * 0.25 / 1_000_000
    expected_output_cost = 1_000 * 15.00 / 1_000_000
    expected_thought_cost = 2_000 * 15.00 / 1_000_000
    assert ideator.uncached_input_cost == expected_uncached_input_cost
    assert ideator.cached_input_cost == expected_cached_input_cost
    assert ideator.input_cost == expected_uncached_input_cost + expected_cached_input_cost
    assert ideator.output_cost == expected_output_cost
    assert ideator.thought_cost == expected_thought_cost
    assert ideator.total_cost == (
        expected_uncached_input_cost
        + expected_cached_input_cost
        + expected_output_cost
        + expected_thought_cost
    )

    summary = ideator.get_token_usage_summary()
    assert summary["thoughts"] == 2_000
    assert summary["cached_input"] == 50_000
    assert summary["uncached_input"] == 160_000
    assert summary["requests"] == 1
    assert summary["large_prompt_requests"] == 1
    assert summary["cost"] == ideator.total_cost


def test_formatter_recovers_empty_response_with_source_idea_text():
    formatter = create_formatter()
    prompts = SimpleNamespace(FORMAT_PROMPT="FORMAT:{input_text}")

    with patch("idea.llm.get_prompts", return_value=prompts), patch.object(
        Formatter,
        "generate_text",
        return_value="",
    ):
        result = formatter.format_idea(
            {
                "id": "idea-1",
                "idea": "Original idea body from upstream generation",
                "parent_ids": [],
            },
            "airesearch",
        )

    assert result["idea"].title == "Untitled"
    assert result["idea"].content == "Original idea body from upstream generation"
