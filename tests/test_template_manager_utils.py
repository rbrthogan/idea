import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.template_manager import get_template_starter


def test_get_template_starter_structure():
    starter = get_template_starter()
    assert starter["name"] == "Custom Template"
    assert "prompts" in starter and isinstance(starter["prompts"], dict)
    required_prompts = ["context", "idea", "format", "critique", "refine", "breed"]
    for p in required_prompts:
        assert p in starter["prompts"]
    assert isinstance(starter["comparison_criteria"], list) and starter["comparison_criteria"]
