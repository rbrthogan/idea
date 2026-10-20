import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.diversity import DiversityCalculator


def test_embedding_2_separate_candidates_survive_real_sdk_batch_normalization():
    from google import genai
    from unittest.mock import patch
    client = genai.Client(api_key="fake-offline-test-key")
    calculator = DiversityCalculator(api_key=None)
    calculator.client = client
    texts = ["alpha candidate", "beta candidate", "gamma candidate"]
    captured = {}
    def request(method, path, payload, options):
        captured.update(method=method, path=path, payload=payload)
        # Model the API's actual one-vector-per-Content semantics. A raw
        # list[str] normalizes to one Content and would produce just one vector.
        return SimpleNamespace(headers={}, body=json.dumps({"embeddings": [
            {"values": [float(index + 1), .5]} for index in range(len(payload["requests"]))]}))
    try:
        with patch.object(client.models._api_client, "request", side_effect=request):
            vectors = asyncio.run(calculator._get_embeddings_batch(texts))
    finally:
        client.close()
    assert len(captured["payload"]["requests"]) == 3
    assert [item["content"]["parts"] for item in captured["payload"]["requests"]] == [[{"text": text}] for text in texts]
    assert all(vector is not None and len(vector) == 2 for vector in vectors)
    assert len(calculator._embedding_cache) == 3
    summary = calculator.get_token_usage_summary()
    assert summary["requests"] == 1
    assert summary["estimated_input"] > 0


class FakeEmbeddingModels:
    def __init__(self):
        self.calls = 0

    def embed_content(self, **_kwargs):
        self.calls += 1
        return SimpleNamespace(
            embedding=SimpleNamespace(values=[0.1, 0.2, 0.3]),
            usage_metadata=SimpleNamespace(prompt_token_count=12),
        )


def test_embedding_costs_are_tracked_and_cached():
    models = FakeEmbeddingModels()
    calculator = DiversityCalculator(api_key=None)
    calculator.client = SimpleNamespace(models=models)

    embedding = asyncio.run(calculator._get_embedding("A short idea"))
    assert embedding.tolist() == [0.1, 0.2, 0.3]

    summary = calculator.get_token_usage_summary()
    assert summary["input"] == 12
    assert summary["total"] == 12
    assert summary["requests"] == 1
    assert summary["per_model"]["gemini-embedding-2"]["input"] == 12
    assert summary["cost"] == pytest.approx(12 * 0.20 / 1_000_000)

    cached_embedding = asyncio.run(calculator._get_embedding("A short idea"))
    assert cached_embedding.tolist() == [0.1, 0.2, 0.3]
    assert models.calls == 1
    assert calculator.get_token_usage_summary()["input"] == 12


def test_openai_embeddings_use_the_selected_provider_and_report_exact_usage():
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "data": [{"embedding": [0.4, 0.5, 0.6]}],
                "usage": {"prompt_tokens": 9, "total_tokens": 9},
            }

    class FakeClient:
        def post(self, path, json):
            captured["path"] = path
            captured["json"] = json
            return FakeResponse()

    calculator = DiversityCalculator(api_key="test-key", provider="openai")
    calculator.client = FakeClient()

    embedding = asyncio.run(calculator._get_embedding("An OpenAI idea"))

    assert embedding.tolist() == [0.4, 0.5, 0.6]
    assert captured == {
        "path": "/v1/embeddings",
        "json": {"model": "text-embedding-3-small", "input": "An OpenAI idea"},
    }
    summary = calculator.get_token_usage_summary()
    assert summary["input"] == 9
    assert summary["model"] == "text-embedding-3-small"
    assert summary["cost"] == pytest.approx(9 * 0.02 / 1_000_000)
