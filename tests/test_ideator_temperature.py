import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Ideator


def test_ideator_uses_configured_temperature(monkeypatch):
    used_temps = []

    # Avoid actual provider setup
    monkeypatch.setattr(Ideator, "_setup_provider", lambda self: None)

    def fake_generate_text(self, prompt, temperature=None, response_schema=None):
        used_temps.append(temperature if temperature is not None else self.temperature)
        return "stub"

    monkeypatch.setattr(Ideator, "generate_text", fake_generate_text)

    ideator = Ideator(temperature=0.42)

    ideator.generate_context("airesearch")
    assert used_temps[-1] == 0.42

    used_temps.clear()
    ideator.seed_ideas(2, "airesearch")
    assert all(t == 0.42 for t in used_temps)

    used_temps.clear()
    ideator.generate_new_idea([{"id": 1, "idea": "A"}], "airesearch")
    assert used_temps[-1] == 0.42
