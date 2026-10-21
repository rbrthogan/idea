"""Comparison budgets exercise request -> group scheduling -> actual pairings offline."""
import asyncio
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from idea.evolution import EvolutionEngine
from idea.evolution_orchestrator import EvolutionOrchestrator
from idea.evolution_types import EvolutionRunConfig
from idea.run_requests import parse_run_request
from idea.swiss_tournament import tournament_round_budget
from idea.viewer import _resolve_tournament_settings
from tests.test_llm_agents import create_critic
from tests.test_rater_async_recovery import run_rater


@pytest.mark.parametrize("size,rounds,pairs", [
    (1, 1, 0), (2, 1, 1), (3, 2, 2), (5, 3, 6),
    (8, 3, 12), (9, 4, 16), (16, 4, 32), (32, 5, 80),
])
def test_default_budget_bounds_actual_judge_calls_and_balances_participation(size, rounds, pairs):
    config = parse_run_request({"popSize": size, "islandCount": 1})
    assert config.tournament_count == 1.0
    assert config.tournament_rounds == config.full_tournament_rounds == rounds
    assert config.tournament_rounds_policy == "swiss_log2"
    assert _resolve_tournament_settings(size, 1, None)[1:] == (rounds, rounds)
    critic = create_critic()
    details = []
    with patch.dict("os.environ", {"ENABLE_PARALLEL_TOURNAMENT": "0"}), \
         patch.object(critic, "compare_ideas", return_value="A") as compare:
        critic.get_tournament_ranks(
            [f"unique idea {i}" for i in range(size)], "airesearch",
            rounds=rounds, full_tournament_rounds=rounds, details=details,
        )
    assert compare.call_count == pairs
    assert len(details) == (rounds if size > 1 else 0)
    observed = [tuple(sorted((p["a_idx"], p["b_idx"]))) for r in details for p in r["pairs"]]
    assert len(observed) == len(set(observed)) == pairs
    participation = Counter(i for pair in observed for i in pair)
    counts = [participation[i] for i in range(size)]
    assert max(counts) - min(counts) <= 1
    assert all(r["ratings_committed"] for r in details)


@pytest.mark.parametrize("population,islands,expected", [
    (10, 2, [3, 3]), (17, 2, [4, 3]), (24, 3, [3, 3, 3]),
    (32, 1, [5]), (32, 2, [4, 4]),
])
def test_group_budget_uses_actual_island_size(population, islands, expected):
    config = parse_run_request({"popSize": population, "islandCount": islands})
    engine = SimpleNamespace(**config.to_persistence_dict())
    orchestrator = EvolutionOrchestrator(engine, lambda _: asyncio.sleep(0))
    sizes = [population // islands + (i < population % islands) for i in range(islands)]
    assert [orchestrator._local_tournament_rounds(size) for size in sizes] == expected
    assert orchestrator._planned_generation_tournament_rounds() == sum(expected)


def test_old_saved_counts_keep_their_meaning_on_resume():
    old = {"pop_size": 16, "island_count": 2, "tournament_count": 1.0,
           "tournament_rounds": 15, "full_tournament_rounds": 15}
    saved = EvolutionRunConfig.from_stored_mapping(old)
    assert saved.tournament_rounds_policy == "legacy_linear"
    assert saved.full_tournament_rounds == saved.tournament_rounds == 15
    restored = EvolutionRunConfig.from_stored_mapping(saved.to_persistence_dict())
    orchestrator = EvolutionOrchestrator(SimpleNamespace(**restored.to_persistence_dict()), lambda _: asyncio.sleep(0))
    assert orchestrator._local_tournament_rounds(8) == 7
    assert restored == saved
    fresh = parse_run_request({"popSize": 16, "islandCount": 2, "tournamentCount": 1.0})
    fresh_orchestrator = EvolutionOrchestrator(SimpleNamespace(**fresh.to_persistence_dict()), lambda _: asyncio.sleep(0))
    assert fresh_orchestrator._local_tournament_rounds(8) == 3


def test_constructor_default_and_explicit_round_override():
    with patch("idea.llm.LLMWrapper._setup_provider", return_value=None):
        default = EvolutionEngine(pop_size=16, island_count=1)
        explicit = EvolutionEngine(pop_size=16, island_count=1, tournament_rounds=2)
    assert default.full_tournament_rounds == default.tournament_rounds == 4
    assert default.tournament_count == 1
    assert explicit.tournament_rounds == 2
    assert explicit.tournament_count == .5
    assert EvolutionRunConfig.from_stored_mapping(default.config_state.to_persistence_dict()) == default.config_state


@pytest.mark.parametrize("size,count,expected_rounds,expected_pairs", [
    (0, 1, 1, 0), (1, 1, 1, 0), (5, 1, 3, 6),
    (16, 1, 4, 32), (16, .5, 2, 16), (16, 1.5, 6, 48),
])
def test_browser_auto_rating_estimate_matches_backend(size, count, expected_rounds, expected_pairs):
    result = run_rater(f"""
ideasDb = Object.fromEntries(Array.from({{length:{size}}}, (_, i) => [i, {{id:i}}]));
nodes.autoTournamentCount.value = '{count}';
syncAutoTournamentCountUI();
process.stdout.write(JSON.stringify({{estimate:getAutoTournamentEstimate(),hint:nodes.autoTournamentRoundsHint.textContent}}));
""")
    assert result["estimate"]["targetRounds"] == expected_rounds
    assert result["estimate"]["estimatedTotalMatches"] == expected_pairs
    assert result["estimate"]["fullTournamentRounds"] == tournament_round_budget(size)
    assert str(expected_pairs) + " pairings" in result["hint"]
