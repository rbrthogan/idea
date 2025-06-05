import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.viewer import idea_to_dict
from idea.models import Idea


def test_idea_to_dict_from_model():
    idea = Idea(title="Title", content="Proposal")
    result = idea_to_dict(idea)
    assert result == {
        "title": "Title",
        "content": "Proposal",
        "parent_ids": [],
        "match_count": 0,
        "auto_match_count": 0,
        "manual_match_count": 0,
    }


def test_idea_to_dict_from_dict_with_model():
    idea = Idea(title="T", content="P")
    input_data = {
        "id": 123,
        "idea": idea,
        "parent_ids": [1, 2],
        "match_count": 1,
        "auto_match_count": 2,
        "manual_match_count": 3,
    }
    result = idea_to_dict(input_data)
    assert result == {
        "id": "123",
        "title": "T",
        "content": "P",
        "parent_ids": [1, 2],
        "match_count": 1,
        "auto_match_count": 2,
        "manual_match_count": 3,
    }


def test_idea_to_dict_from_dict_with_string():
    input_data = {
        "id": 5,
        "idea": "raw idea",
    }
    result = idea_to_dict(input_data)
    assert result == {
        "id": "5",
        "title": "Untitled",
        "content": "raw idea",
        "parent_ids": [],
        "match_count": 0,
        "auto_match_count": 0,
        "manual_match_count": 0,
    }


def test_idea_to_dict_from_dict_with_dict():
    input_data = {
        "id": 1,
        "idea": {"title": "A", "content": "B"},
        "match_count": 9,
    }
    result = idea_to_dict(input_data)
    assert result == {
        "id": "1",
        "title": "A",
        "content": "B",
        "parent_ids": [],
        "match_count": 9,
        "auto_match_count": 0,
        "manual_match_count": 0,
    }


def test_idea_to_dict_from_string():
    result = idea_to_dict("hello")
    assert result == {
        "title": "Untitled",
        "content": "hello",
        "parent_ids": [],
        "match_count": 0,
    }
