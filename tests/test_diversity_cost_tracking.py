import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.diversity import DiversityCalculator


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
