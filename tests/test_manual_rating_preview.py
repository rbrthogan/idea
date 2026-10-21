"""Manual pair previews are reads; only a confirmed vote advances Swiss state."""

import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from idea import viewer
from idea.auth import UserInfo


class RatingStore:
    def __init__(self, size=5, *, revision=True):
        self.data = {
            "history": [[{
                "id": chr(97 + index),
                "idea": {"title": f"Candidate {index}", "content": "Original body"},
                "generation": 0,
                "elo": 1500,
            } for index in range(size)]],
        }
        if revision:
            self.data["_save_revision"] = "initial-revision"
        self.writes = 0
        self.owner = UserInfo(uid="test-owner", email="owner@example.com")
        self.load = AsyncMock(side_effect=lambda *_args, **_kwargs: (None, self.data, "firestore"))
        self.editable = AsyncMock()
        self.save = AsyncMock(side_effect=self.commit)

    async def commit(self, _owner, _evolution, data):
        if data.get("_save_revision") != self.data.get("_save_revision"):
            raise viewer.db.EvolutionSaveConflict("A concurrent save changed the experiment")
        self.writes += 1
        self.data = copy.deepcopy(data)
        self.data["_save_revision"] = f"revision-{self.writes}"

    def call(self, endpoint, payload):
        request = SimpleNamespace(json=AsyncMock(return_value=payload))
        with patch.object(viewer, "_load_rating_evolution", self.load), \
             patch.object(viewer.db, "ensure_evolution_editable", self.editable), \
             patch.object(viewer.db, "save_evolution", self.save):
            return asyncio.run(endpoint(request, self.owner))

    def preview(self):
        return self.call(viewer.get_efficient_pair, {"evolution_id": "run"})

    def vote(self, preview, **overrides):
        payload = {
            "evolution_id": "run",
            "idea_a_id": preview["idea_a"]["id"],
            "idea_b_id": preview["idea_b"]["id"],
            "pair_token": preview["pair_token"],
            "outcome": "A",
        }
        payload.update(overrides)
        return self.call(viewer.submit_rating, payload)


@pytest.mark.parametrize("pending", [False, True])
def test_preview_is_repeatable_and_never_mutates_or_saves_loaded_snapshot(pending):
    store = RatingStore()
    if pending:
        store.data["rating_swiss_state"] = {"manual": {
            "round_number": 3,
            "pending_pairs": [{"a_id": "c", "b_id": "d"}, {"a_id": "a", "b_id": "b"}],
            "match_history": [], "bye_counts": {"e": 1}, "last_bye_id": "e",
        }}
    original = copy.deepcopy(store.data)
    first = store.preview()
    assert first == store.preview() == store.preview()
    assert store.data == original
    assert len(first["pair_token"]) == 64
    assert first["idea_a"]["title"].startswith("Candidate")
    store.save.assert_not_awaited()
    store.editable.assert_not_awaited()


def test_votes_commit_exact_preview_and_advance_round_and_bye_only_when_saved():
    store = RatingStore()
    first = store.preview()
    assert first["swiss_status"]["round_number"] == 1
    assert "rating_swiss_state" not in store.data
    assert store.vote(first).status_code == 200
    state = store.data["rating_swiss_state"]["manual"]
    assert state["round_number"] == 1
    assert sum(state["bye_counts"].values()) == 1
    assert len(state["pending_pairs"]) == 1
    assert len(state["match_history"]) == 1
    counts = {idea["id"]: idea["manual_match_count"] for idea in store.data["history"][0]}
    assert {key for key, count in counts.items() if count} == {first["idea_a"]["id"], first["idea_b"]["id"]}

    persisted = copy.deepcopy(store.data)
    second = store.preview()
    assert second == store.preview()
    assert store.data == persisted
    assert second["swiss_status"]["round_number"] == 1
    assert store.vote(second).status_code == 200
    state = store.data["rating_swiss_state"]["manual"]
    assert not state["pending_pairs"]
    assert len(state["match_history"]) == 2
    assert sum(state["bye_counts"].values()) == 1
    assert store.preview()["swiss_status"]["round_number"] == 2
    assert state["round_number"] == 1
    assert store.writes == 2


@pytest.mark.parametrize("revision", [False, True])
def test_replayed_vote_is_rejected_even_when_next_round_has_same_two_candidates(revision):
    store = RatingStore(size=2, revision=revision)
    first = store.preview()
    assert store.vote(first).status_code == 200
    persisted = copy.deepcopy(store.data)
    assert store.vote(first).status_code == 409
    assert store.data == persisted
    assert store.writes == 1
    next_pair = store.preview()
    assert next_pair["pair_token"] != first["pair_token"]
    assert {next_pair["idea_a"]["id"], next_pair["idea_b"]["id"]} == {first["idea_a"]["id"], first["idea_b"]["id"]}
    assert store.vote(next_pair).status_code == 200


@pytest.mark.parametrize("overrides", [{"pair_token": None}, {"pair_token": "stale"}, {"idea_a_id": "c", "idea_b_id": "d"}])
def test_stale_or_unselected_pair_cannot_change_saved_ratings(overrides):
    store = RatingStore()
    original = copy.deepcopy(store.data)
    response = store.vote(store.preview(), **overrides)
    assert response.status_code == 409
    assert "Reload the pair" in json.loads(response.body)["message"]
    assert store.data == original
    store.save.assert_not_awaited()


def test_reversed_pair_preserves_outcome_orientation():
    store = RatingStore(size=2)
    pair = store.preview()
    response = store.vote(pair, idea_a_id=pair["idea_b"]["id"], idea_b_id=pair["idea_a"]["id"], outcome="A")
    assert response.status_code == 200
    elos = json.loads(response.body)["updated_elos"]
    assert elos[pair["idea_b"]["id"]] == 1516
    assert elos[pair["idea_a"]["id"]] == 1484


@pytest.mark.parametrize("conflict", ["active_worker", "concurrent_revision"])
def test_vote_keeps_active_worker_and_revision_fences(conflict):
    store = RatingStore()
    pair = store.preview()
    original = copy.deepcopy(store.data)
    if conflict == "active_worker":
        store.editable.side_effect = viewer.db.ActiveEvolutionWriteConflict("The owning worker is active")
    else:
        store.save.side_effect = viewer.db.EvolutionSaveConflict("A concurrent save changed the experiment")
    assert store.vote(pair).status_code == 409
    assert store.data == original
    store.editable.assert_awaited_once_with("test-owner", "run")
    if conflict == "active_worker":
        store.save.assert_not_awaited()
    else:
        assert store.save.await_args.args[2]["_save_revision"] == "initial-revision"


def test_same_candidate_cannot_be_rated_against_itself():
    store = RatingStore()
    pair = store.preview()
    assert store.vote(pair, idea_b_id=pair["idea_a"]["id"]).status_code == 400
    store.save.assert_not_awaited()


def test_stopping_mid_round_saves_only_played_matches_and_resumes_pending_pair():
    store = RatingStore(size=6)
    first = store.preview()
    assert store.vote(first).status_code == 200
    persisted = copy.deepcopy(store.data)
    counts = [idea['manual_match_count'] for idea in persisted['history'][0]]
    assert sorted(counts) == [0, 0, 0, 0, 1, 1]
    assert len(persisted['rating_swiss_state']['manual']['pending_pairs']) == 2

    # Simulate a new session: reload saved state, then preview repeatedly.
    resumed = RatingStore(size=6)
    resumed.data = copy.deepcopy(persisted)
    next_pair = resumed.preview()
    assert next_pair == resumed.preview()
    assert resumed.data == persisted
    assert resumed.writes == 0
    assert {next_pair['idea_a']['id'], next_pair['idea_b']['id']}.isdisjoint(
        {first['idea_a']['id'], first['idea_b']['id']})
    for _ in range(2):
        assert resumed.vote(resumed.preview()).status_code == 200
    assert [idea['manual_match_count'] for idea in resumed.data['history'][0]] == [1] * 6
    assert not resumed.data['rating_swiss_state']['manual']['pending_pairs']


@pytest.mark.parametrize('size,rounds,matches_each', [(6, 3, 3), (5, 5, 4)])
def test_completed_swiss_rounds_balance_participation_including_rotating_byes(size, rounds, matches_each):
    store = RatingStore(size=size)
    for _ in range(rounds * (size // 2)):
        assert store.vote(store.preview()).status_code == 200
    state = store.data['rating_swiss_state']['manual']
    assert state['round_number'] == rounds
    assert not state['pending_pairs']
    assert [idea['manual_match_count'] for idea in store.data['history'][0]] == [matches_each] * size
