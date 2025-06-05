import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.evolution import EvolutionEngine
from idea.config import model_prices_per_million_tokens


class DummyAgent:
    def __init__(self, model_name, input_tokens, output_tokens):
        self.model_name = model_name
        self.input_token_count = input_tokens
        self.output_token_count = output_tokens
        self.total_token_count = input_tokens + output_tokens


def test_get_total_token_count():
    engine = EvolutionEngine(pop_size=1, generations=1)
    engine.ideator = DummyAgent("gemini-2.0-flash-lite", 100, 200)
    engine.formatter = DummyAgent("gemini-2.0-flash", 10, 20)
    engine.critic = DummyAgent("gemini-2.0-flash-lite", 30, 40)
    engine.breeder = DummyAgent("gemini-2.0-flash", 50, 60)

    result = engine.get_total_token_count()

    assert result["total_input"] == 100 + 10 + 30 + 50
    assert result["total_output"] == 200 + 20 + 40 + 60
    assert result["total"] == result["total_input"] + result["total_output"]

    def calc_cost(model, inp, out):
        pricing = model_prices_per_million_tokens.get(model)
        return (
            pricing["input"] * inp / 1_000_000
            + pricing["output"] * out / 1_000_000
        )

    expected_total_cost = sum(
        [
            calc_cost("gemini-2.0-flash-lite", 100, 200),
            calc_cost("gemini-2.0-flash", 10, 20),
            calc_cost("gemini-2.0-flash-lite", 30, 40),
            calc_cost("gemini-2.0-flash", 50, 60),
        ]
    )

    assert abs(result["cost"]["total_cost"] - expected_total_cost) < 1e-9
