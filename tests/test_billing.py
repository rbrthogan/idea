import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

from idea import database, viewer
from idea.auth import UserInfo, require_auth
from idea.billing import compact_usage, build_billing_response
from idea.llm import Ideator, Breeder
from idea.evaluation.artifacts import GeminiCodeExecutionRunner


def agent(cls=Ideator, **kwargs):
    with patch("idea.llm.LLMWrapper._setup_provider"):
        return cls(**kwargs)


def test_per_request_tier_and_captured_price_do_not_reprice_later():
    llm = agent(model_name="gemini-2.5-pro")
    usage = {"prompt_token_count": 150_000, "candidates_token_count": 100,
             "thoughts_token_count": 20, "cached_content_token_count": 50_000}
    llm._record_generation_usage(usage)
    llm._record_generation_usage(usage)
    original = llm.get_token_usage_summary()
    assert len(original["usage_by_price"]) == 1
    assert original["usage_by_price"][0]["large_prompt_tier"] is False
    with patch("idea.llm.estimate_generation_token_cost", side_effect=AssertionError("repriced")):
        summary = llm.get_token_usage_summary()
    assert summary["cost"] == original["cost"]
    assert summary["cache_savings"] > 0
    assert summary["usage_by_price"][0]["requests"] == 2


def test_reasoning_and_cache_write_cost_are_counted_once():
    llm = agent(provider="openai", model_name="gpt-6.1-sol")
    llm._record_openai_generation_usage({"input_tokens": 1000, "output_tokens": 200,
        "input_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 50},
        "output_tokens_details": {"reasoning_tokens": 80}})
    summary = llm.get_token_usage_summary()
    assert summary["output"] == 120
    assert summary["thoughts"] == 80
    assert summary["uncached_input"] == 850
    assert summary["total"] == 1200
    snapshot = summary["usage_by_price"][0]
    expected = (850 * snapshot["input_rate"] + 100 * snapshot["cached_input_rate"]
                + 50 * snapshot["cache_write_input_rate"] + 200 * snapshot["output_rate"]) / 1_000_000
    assert summary["cost"] == pytest.approx(expected)


def test_breeder_child_usage_charged_even_when_breed_does_not_finish():
    breeder = agent(Breeder, model_name="gemini-2.5-pro")
    with patch("idea.llm.LLMWrapper._setup_provider"):
        child = breeder._get_thread_ideator()
    child._record_generation_usage({"prompt_token_count": 500, "candidates_token_count": 80})
    child._record_generation_usage(None)
    parent_summary = breeder.get_token_usage_summary()
    assert parent_summary["cost"] == child.get_token_usage_summary()["cost"]
    assert parent_summary["input"] == 500
    assert parent_summary["requests"] == 1
    assert parent_summary["missing_usage_requests"] == 1


def test_artifact_usage_aliases_and_per_response_cost_snapshot():
    runner = GeminiCodeExecutionRunner(enabled=True)
    with patch("idea.evaluation.artifacts.estimate_generation_token_cost", side_effect=lambda *args, **kwargs: {
        "total_cost": 1, "input_cost": .4, "uncached_input_cost": .4, "cached_input_cost": 0,
        "cache_write_input_cost": 0, "output_cost": .4, "thought_cost": .2,
        "uncached_input_tokens": kwargs["input_tokens"], "cached_input_tokens": 0,
        "cache_write_input_tokens": 0, "pricing_known": True, "price_version": "old",
        "input_rate": 2, "output_rate": 4}):
        runner._record_usage({"input_tokens": 15, "output_tokens": 9, "thought_tokens": 3})
        runner._record_usage({"prompt_token_count": 18, "candidates_token_count": 7, "thoughts_token_count": 2})
    with patch("idea.evaluation.artifacts.estimate_generation_token_cost", side_effect=AssertionError("repriced")):
        summary = runner.get_token_usage_summary()
    assert summary["input"] == 33
    assert summary["output"] == 16
    assert summary["thoughts"] == 5
    assert summary["cost"] == 2
    assert summary["usage_by_price"][0]["price_version"] == "old"


def usage(cost=2):
    return {"critic": {"model": "gemini-historical", "input": 100, "output": 20,
                       "thoughts": 5, "cost": cost, "cached_input": 30},
            "total_input": 100, "total_output": 20, "total_thoughts": 5,
            "total_cached_input": 30, "cost": {"total_cost": cost}}


def test_billing_deduplicates_cumulative_runs_and_rerating_events():
    runs = [{"id": "checkpoint", "evolution_id": "e", "updated_at": "2026-01-01", "token_counts": usage(1)},
            {"id": "e", "evolution_id": "e", "updated_at": "2026-02-01", "token_counts": usage(2)}]
    event = {"id": "session", "evolution_id": "e", "summary": compact_usage(usage(.5))}
    payload = build_billing_response(runs, [event, event])
    assert payload["totals"]["estimated_cost"] == 2.5
    assert payload["totals"]["rerating_cost"] == .5
    assert payload["totals"]["run_count"] == 1
    assert payload["totals"]["input_tokens"] == 200
    assert payload["by_provider"][0]["estimated_cost"] == 2.5
    assert payload["coverage"]["legacy_price_runs"] == 1
    assert payload["coverage"]["invoice_available"] is False


def test_missing_usage_does_not_invent_free_cost():
    response = build_billing_response([{"id": "old"}], [])
    assert response["totals"]["estimated_cost"] is None
    assert response["coverage"]["runs_missing_cost"] == 1
    assert response["coverage"]["runs_missing_usage"] == 1


@pytest.mark.parametrize("metadata", [{"unexpected": "malformed"}, {"prompt_token_count": float("inf")}, {"prompt_token_count": float("nan")}])
def test_malformed_provider_usage_is_a_coverage_gap(metadata):
    llm = agent()
    llm._record_generation_usage(metadata)
    summary = llm.get_token_usage_summary()
    assert summary["requests"] == 0
    assert summary["missing_usage_requests"] == 1
    assert not summary["usage_by_price"]


def test_failed_transport_with_no_usage_is_reported_as_a_gap():
    llm = agent(provider="openai", model_name="gpt-6.1-sol")
    llm.client = SimpleNamespace(post=lambda *args, **kwargs: (_ for _ in ()).throw(TimeoutError("fake timeout")))
    with pytest.raises(TimeoutError):
        llm._generate_openai_response("prompt", 0, 1, None)
    assert llm.get_token_usage_summary()["missing_usage_requests"] == 1


def test_total_only_and_partially_nonfinite_usage_remain_incomplete():
    llm = agent()
    llm._record_generation_usage({"total_token_count": 20})
    llm._record_generation_usage({"prompt_token_count": 10, "candidates_token_count": float("inf")})
    assert llm.get_token_usage_summary()["missing_usage_requests"] == 2


@pytest.mark.parametrize("metadata", [{"unexpected": "malformed"}, {"input_tokens": float("inf")}])
def test_artifact_malformed_usage_is_a_gap(metadata):
    runner = GeminiCodeExecutionRunner()
    runner._record_usage(metadata)
    summary = runner.get_token_usage_summary()
    assert summary["requests"] == 0
    assert summary["missing_usage_requests"] == 1


@pytest.mark.parametrize("snapshots", ["oops", {"price": 1}, [None], ["not a snapshot"]])
def test_malformed_historical_price_snapshots_do_not_break_cost_view(snapshots):
    tokens = usage(2)
    tokens["critic"]["usage_by_price"] = snapshots
    response = build_billing_response([{"id": "legacy", "token_counts": tokens}], [])
    assert response["totals"]["estimated_cost"] == 2
    assert response["coverage"]["legacy_price_runs"] == 1
    assert response["coverage"]["unpriced_models"] == ["gemini-historical"]


@pytest.mark.parametrize("cost", ["invalid", float("nan"), float("inf"), -1])
def test_invalid_recorded_cost_is_missing_not_free(cost):
    tokens = usage(cost)
    response = build_billing_response([{"id": "legacy", "token_counts": tokens}], [])
    assert response["totals"]["estimated_cost"] is None
    assert response["coverage"]["runs_missing_cost"] == 1


@pytest.mark.parametrize("aggregate, expected", [
    ({"total_input": 10, "total_tool_input": 5}, 15),
    ({"total_tool_input": 5}, 15),
    ({}, 15),
    ({"total_tool_input": 8}, 18),
])
def test_legacy_tool_input_is_counted_once_with_or_without_aggregate_input(aggregate, expected):
    summary = compact_usage({
        **aggregate,
        "cost": {"total_cost": 0.25},
        "ideator": {"model": "unknown", "input": 10, "tool_input": 5, "cost": 0.25},
    })
    assert summary["input_tokens"] == expected
    assert summary["models"][0]["input_tokens"] == 15


def test_overflowing_historical_token_counts_do_not_break_billing():
    oversized = 10 ** 10000
    summary = compact_usage({
        "cost": {"total_cost": 0.25},
        "total_input": oversized,
        "ideator": {"model": "unknown", "input": oversized, "cost": 0.25},
    })
    response = build_billing_response([
        {"evolution_id": "old-run", "_billing_summary": summary},
    ], [])
    assert response["totals"]["estimated_cost"] == 0.25
    assert response["totals"]["input_tokens"] == 0
    assert response["coverage"]["unpriced_models"] == ["unknown"]


def test_manifest_summary_excludes_candidates_and_preserves_recorded_price():
    data, chunks = database._prepare_evolution_save_data({"history": [[{"idea": "private long candidate"}]],
        "token_counts": usage(9)}, "owner", "today")
    assert data["_billing_summary"]["estimated_cost"] == 9
    assert "private long candidate" not in json.dumps(data["_billing_summary"])
    assert "token_counts" not in data
    assert chunks


def test_checkpoint_restore_preserves_model_attribution_through_multiple_resumes():
    from idea.evolution import EvolutionEngine
    engine = EvolutionEngine.__new__(EvolutionEngine)
    engine._persisted_token_counts = usage(2)
    once = engine.get_total_token_count()
    engine._persisted_token_counts = once
    twice = engine.get_total_token_count()
    compact = compact_usage(twice)
    assert compact["estimated_cost"] == 2
    assert compact["models"][0]["model"] == "gemini-historical"
    assert compact["models"][0]["estimated_cost"] == 2
    assert compact["input_tokens"] == 100
    assert len(twice["restored_usage"]["components"]) == 1


class Snapshot:
    def __init__(self, identity, data, reference=None):
        self.id, self.data, self.reference = identity, data, reference
    def to_dict(self):
        return dict(self.data)


class BillingFake:
    def __init__(self, docs, path=(), reads=None):
        self.docs, self.path, self.reads = docs, path, reads if reads is not None else []
    def collection(self, name):
        return BillingFake(self.docs, (*self.path, name), self.reads)
    def document(self, name):
        return BillingFake(self.docs, (*self.path, name), self.reads)
    def order_by(self, *args, **kwargs):
        return self
    def limit(self, count):
        self.count = count
        return self
    def stream(self):
        self.reads.append(self.path)
        return [Snapshot(path[-1], data, BillingFake(self.docs, path, self.reads))
                for path, data in self.docs.items() if path[:-1] == self.path][:self.count]
    def create(self, value):
        from google.api_core.exceptions import AlreadyExists
        if self.path in self.docs:
            raise AlreadyExists("duplicate")
        self.docs[self.path] = dict(value)


def test_historical_billing_reads_only_owner_token_chunks_and_records_missing_coverage(monkeypatch):
    fake = BillingFake({("users", "owner", "evolutions", "legacy"): {
        "_save_revision": "r", "_externalized_fields": {"history": {"large": True}, "token_counts": {"tiny": True}}},
        ("users", "other", "evolutions", "private"): {"token_counts": usage(100)}})
    monkeypatch.setattr(database, "get_db", lambda: fake)
    loaded = []
    def load(ref, data):
        loaded.append((ref.path, data["_externalized_fields"]))
        data["token_counts"] = json.dumps(usage(3))
    monkeypatch.setattr(database, "_load_externalized_evolution_fields", load)
    records = asyncio.run(database.list_billing_records("owner"))
    assert loaded == [(("users", "owner", "evolutions", "legacy"), {"token_counts": {"tiny": True}})]
    assert records["runs"][0]["_billing_summary"]["estimated_cost"] == 3
    assert all(path[1] == "owner" for path in fake.reads)
    assert len(records["runs"]) == 1


def test_rerating_event_is_owner_scoped_and_idempotent(monkeypatch):
    fake = BillingFake({})
    monkeypatch.setattr(database, "get_db", lambda: fake)
    assert asyncio.run(database.save_billing_event("owner", "request", {"summary": compact_usage(usage(.2))}))
    assert not asyncio.run(database.save_billing_event("owner", "request", {"summary": compact_usage(usage(100))}))
    assert fake.docs[("users", "owner", "billing_events", "request")]["summary"]["estimated_cost"] == .2


def test_billing_endpoint_only_reads_authenticated_owner():
    async def user():
        return UserInfo(uid="owner", email="owner@example.com")
    app = viewer.app
    app.dependency_overrides[require_auth] = user
    records = AsyncMock(return_value={"runs": [], "events": [], "truncated": False})
    try:
        with patch.object(viewer.db, "list_billing_records", records):
            response = TestClient(app).get("/api/billing?user_id=other")
    finally:
        app.dependency_overrides.pop(require_auth, None)
    assert response.status_code == 200
    records.assert_awaited_once_with("owner")


def request(host="localhost", client="127.0.0.1"):
    return Request({"type": "http", "headers": [(b"host", host.encode())], "client": (client, 1234)})


def test_environment_credentials_only_for_enabled_loopback_dev_user(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "example-test-key-not-real")
    dev = UserInfo(uid="local-dev-user", email="dev@example.local")
    ordinary = UserInfo(uid="ordinary", email="ordinary@example.com")
    with patch.object(viewer.db, "get_user_api_key", AsyncMock(return_value=None)):
        assert asyncio.run(viewer._user_provider_credential(dev, "google_generative_ai", request()))[1] == "environment"
        assert asyncio.run(viewer._user_provider_credential(dev, "google_generative_ai", request(client="203.0.113.8")))[1] == "missing"
        assert asyncio.run(viewer._user_provider_credential(dev, "google_generative_ai", request(host="example.com")))[1] == "missing"
        assert asyncio.run(viewer._user_provider_credential(ordinary, "google_generative_ai", request()))[1] == "missing"
        assert asyncio.run(viewer._user_provider_credential(dev, "google_generative_ai", None))[1] == "missing"
        monkeypatch.setenv("IDEA_DEV_AUTH", "false")
        assert asyncio.run(viewer._user_provider_credential(dev, "google_generative_ai", request()))[1] == "missing"


def test_environment_settings_never_expose_key_prefix_or_suffix(monkeypatch):
    monkeypatch.setenv("IDEA_DEV_AUTH", "true")
    secret = "example-test-secret-not-real"
    monkeypatch.setenv("GEMINI_API_KEY", secret)
    dev = UserInfo(uid="local-dev-user", email="dev@example.local")
    with patch.object(viewer.db, "get_user_api_key", AsyncMock(return_value=None)), \
         patch.object(viewer.db, "get_user_settings", AsyncMock(return_value={})):
        response = asyncio.run(viewer.get_settings_status(user=dev, request=request()))
    payload = json.loads(response.body)
    assert payload["api_keys"]["google_generative_ai"] == {"configured": True, "masked_key": "***", "source": "environment"}
    assert secret not in response.body.decode()
    assert secret[:4] not in response.body.decode()


def test_template_generation_uses_shared_local_credential_and_offloads_provider(monkeypatch):
    import threading
    from idea import template_manager
    monkeypatch.setenv("IDEA_DEV_AUTH", "true")
    monkeypatch.setenv("GEMINI_API_KEY", "fake-local-test-key")
    captured = {}
    main_thread = threading.get_ident()
    class FakeLLM:
        def __init__(self, **kwargs):
            captured["key"] = kwargs["api_key"]
        def generate_text(self, prompt):
            captured["thread"] = threading.get_ident()
            return "fake draft"
    with patch.object(template_manager.db, "get_user_api_key", AsyncMock(return_value=None)), \
         patch.object(template_manager, "LLMWrapper", FakeLLM), \
         patch.object(template_manager, "_load_template_generation_examples", return_value={"a": {}, "b": {}, "c": {}}), \
         patch.object(template_manager, "create_template_generation_prompt", return_value="prompt"), \
         patch.object(template_manager, "parse_generated_template", return_value={"name": "Test"}):
        response = asyncio.run(template_manager._generate_template_from_inputs("ideas",
            UserInfo(uid="local-dev-user", email="dev@example.local"), request=request()))
    assert response.status_code == 200
    assert captured["key"] == "fake-local-test-key"
    assert captured["thread"] != main_thread


@pytest.mark.parametrize("evaluation_fails,storage_fails", [(False, False), (True, False), (False, True), (True, True)])
def test_autorating_spend_is_persisted_on_success_failure_and_skip_save(evaluation_fails, storage_fails):
    from idea.user_state import UserEvolutionState
    llm = agent()
    state = UserEvolutionState()
    def evaluate(critic, *args, **kwargs):
        critic._record_generation_usage({"prompt_token_count": 100, "candidates_token_count": 20})
        if evaluation_fails:
            raise RuntimeError("fake downstream failure after paid usage")
        return SimpleNamespace(batch=SimpleNamespace(records=[]))
    class Body:
        async def json(self):
            return {"evolutionId": "e", "numRounds": 1, "skipSave": True}
    evolution = {"config": {"idea_type": "drabble"}, "history": [[
        {"id": "a", "idea": {"title": "A", "content": "alpha"}},
        {"id": "b", "idea": {"title": "B", "content": "beta"}}]]}
    user = UserInfo(uid="owner", email="owner@example.com")
    saved = AsyncMock(side_effect=RuntimeError("storage unavailable") if storage_fails else None)
    with patch.object(viewer.user_states, "get", AsyncMock(return_value=state)), \
         patch.object(viewer, "_load_rating_evolution", AsyncMock(return_value=(None, evolution, "firestore"))), \
         patch.object(viewer, "_live_or_persisted_artifact_store", return_value=None), \
         patch.object(viewer, "_autorating_pairwise_evaluator", return_value=SimpleNamespace(evaluate_population=evaluate)), \
         patch.object(viewer, "_resolve_user_provider_key", AsyncMock(return_value="fake-test-key")), \
         patch.object(viewer, "Critic", return_value=llm), \
         patch.object(viewer.db, "save_billing_event", saved), \
         patch.object(viewer, "_save_rating_evolution", AsyncMock()) as save_candidates:
        response = asyncio.run(viewer.auto_rate(Body(), user))
    assert response.status_code == (500 if evaluation_fails else 200)
    saved.assert_awaited_once()
    assert saved.call_args.args[0] == "owner"
    assert saved.call_args.args[2]["summary"]["estimated_cost"] > 0
    assert saved.call_args.args[2]["evolution_id"] == "e"
    save_candidates.assert_not_awaited()
    if storage_fails:
        assert "could not be saved" in json.loads(response.body)["billing_warning"]
        assert state.autorating_status["billing_warning"]
