import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.llm import Critic


def test_critic_uses_configured_temperature(monkeypatch):
    used_temps = []

    # Avoid actual provider setup
    monkeypatch.setattr(Critic, "_setup_provider", lambda self: None)

    def fake_generate_text(self, prompt, temperature=None, response_schema=None):
        used_temps.append(temperature if temperature is not None else self.temperature)
        # Provide a predictable response so compare_ideas can parse it
        return "RESULT: A"

    monkeypatch.setattr(Critic, "generate_text", fake_generate_text)

    critic = Critic(temperature=0.25)

    result = critic.compare_ideas({"title": "Idea A", "content": ""}, {"title": "Idea B", "content": ""}, "airesearch")
    assert result == "A"
    assert used_temps[-1] == 0.25
