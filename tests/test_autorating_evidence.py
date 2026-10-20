"""Endpoint regressions using the real Critic's applied-evidence ledger."""
import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from idea import viewer
from idea.auth import UserInfo
from idea.config import DEFAULT_MODEL, DEFAULT_CREATIVE_TEMP, DEFAULT_TOP_P
from idea.llm import Critic
from idea.user_state import UserEvolutionState


def judge():
    level, budget = viewer._get_default_thinking_settings(DEFAULT_MODEL)
    with patch("idea.llm.LLMWrapper._setup_provider"):
        return Critic(model_name=DEFAULT_MODEL, temperature=DEFAULT_CREATIVE_TEMP,
                      top_p=DEFAULT_TOP_P, thinking_level=level, thinking_budget=budget)


def run_endpoint(critic, ideas, cache, *, rounds=1):
    state = UserEvolutionState()
    evolution = {"config": {"idea_type": "airesearch"}, "history": [copy.deepcopy(ideas)],
                 "comparison_cache": copy.deepcopy(cache)}
    class Request:
        async def json(self):
            return {"evolutionId": "e", "numRounds": rounds, "ideaType": "airesearch"}
    saved = AsyncMock()
    with patch.object(viewer.user_states, "get", AsyncMock(return_value=state)), \
         patch.object(viewer, "_load_rating_evolution", AsyncMock(return_value=(None, evolution, "firestore"))), \
         patch.object(viewer, "_live_or_persisted_artifact_store", return_value=None), \
         patch.object(viewer, "_resolve_user_provider_key", AsyncMock(return_value="fake-test-key")), \
         patch.object(viewer, "Critic", return_value=critic), \
         patch.object(viewer.db, "save_billing_event", AsyncMock()), \
         patch.object(viewer, "_save_rating_evolution", saved):
        response = asyncio.run(viewer.auto_rate(Request(), UserInfo(uid="owner", email="owner@example.com")))
    assert response.status_code == 200, response.body.decode()
    saved.assert_awaited_once()
    return json.loads(response.body), evolution, state


def seed():
    critic = judge()
    ideas = [{"id": "a", "idea": {"title": "A", "content": "alpha"}},
             {"id": "b", "idea": {"title": "B", "content": "beta"}}]
    cache = {}
    with patch.object(critic, "compare_ideas", return_value="A"):
        critic.get_tournament_ranks(ideas, "airesearch", rounds=1, comparison_cache=cache)
    for idea in ideas:
        idea["match_count"] = 1
    return ideas, cache


@pytest.mark.parametrize("legacy", [False, True])
def test_applied_cached_evidence_does_not_change_endpoint_ratings_or_counts(legacy):
    ideas, cache = seed()
    if legacy:
        for entry in cache.values():
            for field in ("applied", "post_ratings", "application_sequence", "evidence_version"):
                entry.pop(field, None)
    critic = judge()
    with patch.object(critic, "compare_ideas", side_effect=AssertionError("cached evidence must not make a new judge call")):
        result, persisted, state = run_endpoint(critic, ideas, cache)
    assert result["new_comparisons"] == 0
    assert result["results"] == []
    assert result["completed_comparisons"] == 1
    ratings = {idea["id"]: idea["ratings"]["auto"] for idea in result["ideas"]}
    assert ratings == {"a": 1516, "b": 1484}
    assert all(idea["match_count"] == 1 and idea["auto_match_count"] == 1 for idea in result["ideas"])
    assert {idea["id"]: idea["ratings"]["auto"] for idea in persisted["history"][0]} == ratings
    assert state.autorating_status["new_comparisons"] == 0


def test_fresh_pair_is_applied_once_and_result_uses_critic_rating():
    critic = judge()
    ideas = [{"id": "a", "idea": "alpha"}, {"id": "b", "idea": "beta"}]
    with patch.object(critic, "compare_ideas", return_value="A") as compare:
        result, _, _ = run_endpoint(critic, ideas, {}, rounds=2)
    assert compare.call_count == 1  # Second round reuses this now-applied observation.
    assert result["new_comparisons"] == 1
    assert len(result["results"]) == 1
    assert result["results"][0]["new_elo_a"] == 1516
    assert result["results"][0]["new_elo_b"] == 1484
    assert {idea["id"]: idea["ratings"]["auto"] for idea in result["ideas"]} == {"a": 1516, "b": 1484}
    assert all(idea["match_count"] == 1 and idea["auto_match_count"] == 1 for idea in result["ideas"])


def test_cached_unapplied_observation_commits_once_without_new_judge_call():
    ideas, cache = seed()
    for idea in ideas:
        idea["ratings"]["auto"] = idea["elo"] = 1500
        idea["match_count"] = idea["auto_match_count"] = 0
        idea.pop("rating_evaluations", None)
    for entry in cache.values():
        entry["applied"] = False
        entry.pop("post_ratings", None)
        entry.pop("application_sequence", None)
    critic = judge()
    with patch.object(critic, "compare_ideas", side_effect=AssertionError("unapplied observation is already judged")):
        result, persisted, _ = run_endpoint(critic, ideas, cache, rounds=2)
        replay, _, _ = run_endpoint(critic, persisted["history"][0], persisted["comparison_cache"])
    assert result["new_comparisons"] == 1
    assert result["results"][0]["new_elo_a"] == 1516
    assert result["results"][0]["new_elo_b"] == 1484
    assert all(idea["match_count"] == 1 for idea in result["ideas"])
    assert replay["new_comparisons"] == 0
    assert replay["results"] == []
    assert all(idea["match_count"] == 1 for idea in replay["ideas"])


@pytest.mark.parametrize("legacy", [False, True])
def test_mixed_cached_and_new_pairs_count_only_one_new_observation_across_rounds(legacy):
    ideas, cache = seed()
    if legacy:
        for entry in cache.values():
            for field in ("applied", "post_ratings", "application_sequence", "evidence_version"):
                entry.pop(field, None)
    ideas.extend([{"id": "c", "idea": "gamma"}, {"id": "d", "idea": "delta"}])
    critic = judge()
    with patch.object(critic, "_generate_swiss_round_pairs", return_value=([(0, 1), (2, 3)], None)), \
         patch.object(critic, "compare_ideas", side_effect=lambda a, b, _: "A" if "gamma" in str(a) else "B") as compare:
        result, _, _ = run_endpoint(critic, ideas, cache, rounds=2)
    assert compare.call_count == 1
    assert result["new_comparisons"] == 1
    assert result["completed_comparisons"] == 2
    assert len(result["results"]) == 1
    assert result["results"][0]["idea_a"] == "c"
    assert result["results"][0]["idea_b"] == "d"
    assert result["results"][0]["new_elo_a"] == 1516
    assert result["results"][0]["new_elo_b"] == 1484
    assert all(idea["match_count"] == 1 and idea["auto_match_count"] == 1 for idea in result["ideas"])
    assert {idea["id"]: idea["ratings"]["auto"] for idea in result["ideas"]} == {"a": 1516, "b": 1484, "c": 1516, "d": 1484}


@pytest.mark.parametrize("rating_type", ["auto", "all", "manual"])
def test_reset_auto_invalidates_applied_evidence_but_manual_reset_preserves_it(rating_type):
    ideas, cache = seed()
    for idea in ideas:
        idea["ratings"]["manual"] = 1700
        idea["manual_match_count"] = 2
        idea["match_count"] = 3
    evolution = {"config": {"idea_type": "airesearch"}, "history": [copy.deepcopy(ideas)],
                 "comparison_cache": copy.deepcopy(cache), "rating_swiss_state": {"auto": {}, "manual": {}}}
    class Request:
        async def json(self):
            return {"evolutionId": "e", "ratingType": rating_type}
    saved = AsyncMock()
    with patch.object(viewer, "_load_rating_evolution", AsyncMock(return_value=(None, evolution, "firestore"))), \
         patch.object(viewer, "_save_rating_evolution", saved):
        response = asyncio.run(viewer.reset_ratings(Request(), UserInfo(uid="owner", email="owner@example.com")))
    assert response.status_code == 200
    saved.assert_awaited_once()
    critic = judge()
    with patch.object(critic, "compare_ideas", return_value="A") as compare:
        result, _, _ = run_endpoint(critic, evolution["history"][0], evolution["comparison_cache"])
    if rating_type == "manual":
        assert evolution["comparison_cache"] == cache
        assert compare.call_count == 0
        assert result["new_comparisons"] == 0
        assert all(idea["match_count"] == 1 for idea in result["ideas"])
    else:
        assert evolution["comparison_cache"] == {}
        assert compare.call_count == 1
        assert result["new_comparisons"] == 1
        assert all(idea["auto_match_count"] == 1 for idea in result["ideas"])
        expected_total = 1 if rating_type == "all" else 3
        assert all(idea["match_count"] == expected_total for idea in result["ideas"])
    assert {idea["id"]: idea["ratings"]["auto"] for idea in result["ideas"]} == {"a": 1516, "b": 1484}


@pytest.mark.parametrize("rating_type", ["auto", "all", "manual"])
def test_continuation_population_and_archive_restore_from_consistently_reset_ratings(rating_type):
    from idea.evolution_persistence import EvolutionSerializer
    ideas, cache = seed()
    for idea in ideas:
        idea["ratings"]["manual"] = 1700
        idea["manual_match_count"] = 2
        idea["match_count"] = 3
        idea["owner_notes"] = {"keep": "private annotation"}
    evolution = {"evolution_id": "e", "history": [copy.deepcopy(ideas)],
                 "population": copy.deepcopy(ideas), "quality_archive": copy.deepcopy(ideas),
                 "comparison_cache": copy.deepcopy(cache)}
    evolution["quality_archive"][0]["archive_quality_score"] = .87
    class Request:
        async def json(self):
            return {"evolutionId": "e", "ratingType": rating_type}
    with patch.object(viewer, "_load_rating_evolution", AsyncMock(return_value=(None, evolution, "firestore"))), \
         patch.object(viewer, "_save_rating_evolution", AsyncMock()):
        response = asyncio.run(viewer.reset_ratings(Request(), UserInfo(uid="owner", email="owner@example.com")))
    assert response.status_code == 200
    restored = SimpleNamespace()
    EvolutionSerializer.restore_runtime_state(restored, evolution)
    expected_auto = [1516, 1484] if rating_type == "manual" else [1500, 1500]
    expected_manual = 1700 if rating_type == "auto" else 1500
    for candidates in (restored.history[0], restored.population, restored.quality_archive):
        assert [idea["ratings"]["auto"] for idea in candidates] == expected_auto
        assert all(idea["ratings"]["manual"] == expected_manual for idea in candidates)
        assert all(idea["owner_notes"] == {"keep": "private annotation"} for idea in candidates)
        if rating_type != "manual":
            assert all("rating_evaluations" not in idea for idea in candidates)
    assert restored.quality_archive[0]["archive_quality_score"] == .87
    critic = judge()
    with patch.object(critic, "compare_ideas", return_value="A") as compare:
        ranks = critic.get_tournament_ranks(restored.population, "airesearch", comparison_cache=restored.comparison_cache)
    assert ranks == {0: 1516, 1: 1484}  # Continuation uses population, not its history copy.
    assert compare.call_count == (0 if rating_type == "manual" else 1)
