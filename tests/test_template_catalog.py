import sys
import asyncio
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.auth import UserInfo
from idea.viewer import app, require_auth
from idea.evolution import EvolutionEngine


def test_template_types_includes_system_and_user_templates():
    client = TestClient(app)

    async def mock_require_auth():
        return UserInfo(uid="template-user", email="template@example.com", is_admin=False)

    async def mock_list_user_templates(_user_id):
        return [
            {
                "id": "my_custom_template",
                "name": "My Custom Template",
                "description": "User template",
                "author": "Me",
            }
        ]

    app.dependency_overrides[require_auth] = mock_require_auth
    with patch("idea.viewer.list_available_templates", return_value={
        "airesearch": {
            "name": "AI Research",
            "description": "System template",
            "type": "yaml",
            "author": "Original Idea App",
        }
    }), patch("idea.viewer.db.list_user_templates", side_effect=mock_list_user_templates):
        response = client.get("/api/template-types")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"

        templates = {item["id"]: item for item in data["templates"]}
        assert "airesearch" in templates
        assert "my_custom_template" in templates
        assert templates["airesearch"]["is_system"] is True
        assert templates["my_custom_template"]["is_system"] is False

    app.dependency_overrides.pop(require_auth, None)


def test_load_evolution_rehydrates_custom_template_snapshot():
    evolution_state = {
        "evolution_id": "evo-123",
        "name": "Custom Evolution",
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
        "checkpoint_id": "cp-123",
        "current_generation": 1,
        "config": {
            "idea_type": "custom_story",
            "pop_size": 2,
            "generations": 3,
            "model_type": "gemini-2.0-flash",
            "creative_temp": 1.0,
            "top_p": 0.9,
            "tournament_rounds": 1,
            "mutation_rate": 0.2,
        },
        "population": [],
        "history": [],
        "contexts": [],
        "specific_prompts": [],
        "breeding_prompts": [],
        "tournament_history": [],
        "diversity_history": [],
    }

    custom_template = {
        "name": "Custom Story",
        "description": "Story generator",
        "version": "1.0.0",
        "author": "User",
        "created_date": "2026-01-01",
        "metadata": {"item_type": "stories"},
        "prompts": {
            "context": "Generate concepts",
            "specific_prompt": "{context_pool}",
            "idea": "{requirements}",
            "format": "{input_text}",
            "critique": "{idea}",
            "refine": "{idea} {critique}",
            "breed": "{ideas}",
            "genotype_encode": "{idea_content}",
        },
        "comparison_criteria": ["originality"],
        "special_requirements": "Keep it concise",
    }

    async def mock_get_evolution(_user_id, _evolution_id):
        return evolution_state

    async def mock_get_user_template(_user_id, template_id):
        if template_id == "custom_story":
            return custom_template
        return None

    with patch("idea.evolution.db.get_evolution", side_effect=mock_get_evolution), \
         patch("idea.evolution.db.get_user_template", side_effect=mock_get_user_template), \
         patch("idea.evolution.list_available_templates", return_value={
             "airesearch": {"name": "AI Research"}
         }):
        engine = asyncio.run(EvolutionEngine.load_evolution_for_user("template-user", "evo-123"))

    assert engine is not None
    assert engine.template_data is not None
    assert engine.template_data["name"] == "Custom Story"
    assert "custom_story" in engine.ideator._custom_templates
