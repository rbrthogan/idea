import sys
import asyncio
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.template_manager import get_template_starter, _normalize_template_id, _generate_unique_template_id


def test_get_template_starter_structure():
    starter = get_template_starter()
    assert starter["name"] == "Custom Template"
    assert "prompts" in starter and isinstance(starter["prompts"], dict)
    required_prompts = ["context", "idea", "format", "critique", "refine", "breed"]
    for p in required_prompts:
        assert p in starter["prompts"]
    assert isinstance(starter["comparison_criteria"], list) and starter["comparison_criteria"]


def test_normalize_template_id_sanitizes_and_falls_back():
    assert _normalize_template_id("My Fancy-Template!") == "my_fancy_template"
    assert _normalize_template_id("###") == "custom_template"


def test_generate_unique_template_id_avoids_collisions():
    async def fake_get_user_template(_user_id, template_id):
        # Pretend the first user candidate is also taken.
        return {"name": "taken"} if template_id == "my_template_3" else None

    with patch("idea.template_manager.list_available_templates", return_value={
        "my_template": {"name": "system"},
        "my_template_2": {"name": "system"},
    }), patch("idea.template_manager.db.get_user_template", side_effect=fake_get_user_template):
        template_id = asyncio.run(_generate_unique_template_id("user-123", "My Template"))

    assert template_id == "my_template_4"
