import sys
import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).parent.parent))

from idea.auth import UserInfo
from idea.template_manager import (
    TemplateContextFile,
    TemplateContextFileError,
    _generate_unique_template_id,
    _normalize_template_id,
    _parse_multipart_template_generation_request,
    _read_template_context_upload,
    _generate_template_from_inputs,
    get_template_starter,
)


def test_get_template_starter_structure():
    starter = get_template_starter()
    assert starter["name"] == "Custom Template"
    assert "prompts" in starter and isinstance(starter["prompts"], dict)
    required_prompts = ["context", "specific_prompt", "format", "critique", "refine", "breed"]
    for p in required_prompts:
        assert p in starter["prompts"]
    assert "idea" not in starter["prompts"]
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


def test_generate_template_uses_gemini_31_pro_with_high_thinking_level():
    captured_kwargs = {}

    class FakeLLM:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

        def generate_text(self, _prompt):
            return """
Name: Follow-up Paper Ideas
Description: Generates research directions for follow-up papers.
Item Type: research idea
Special Requirements: None
Context Prompt: Generate 50 concepts for the research domain.
Direction Builder Prompt: Use {context_pool} to define a focused research angle.
Format Prompt: Format {input_text} as a concise research idea.
Critique Prompt: Critique {idea} for novelty and feasibility.
Refine Prompt: Refine {idea} using {critique}.
Breed Prompt: Combine {ideas} into a stronger research direction.
Genotype Encode Prompt: Extract reusable research DNA from {idea_content}.
Comparison Criteria: novelty, feasibility, scientific value
"""

    async def fake_get_user_api_key(_uid):
        return "test-key"

    with patch("idea.template_manager.db.get_user_api_key", side_effect=fake_get_user_api_key), patch(
        "idea.template_manager.LLMWrapper",
        FakeLLM,
    ):
        response = asyncio.run(
            _generate_template_from_inputs(
                "follow-up papers",
                UserInfo(uid="user-123"),
            )
        )

    assert response.status_code == 200
    assert captured_kwargs["model_name"] == "gemini-3.1-pro-preview"
    assert captured_kwargs["thinking_budget"] is None
    assert captured_kwargs["thinking_level"] == "high"


def test_generate_template_uploads_context_file_and_passes_it_to_gemini():
    captured = {}

    class FakeFiles:
        def upload(self, *, file, config=None):
            captured["upload_bytes"] = file.read()
            captured["upload_config"] = config
            return SimpleNamespace(name="files/paper", uri="gemini://paper", mime_type="application/pdf")

        def delete(self, *, name):
            captured["deleted_file"] = name

    class FakeLLM:
        def __init__(self, **kwargs):
            captured["llm_kwargs"] = kwargs
            self.client = SimpleNamespace(files=FakeFiles())

        def generate_text(self, contents):
            captured["contents"] = contents
            return """
Name: Paper Follow-ups
Description: Generates follow-up paper ideas grounded in an uploaded paper.
Item Type: research idea
Special Requirements: None
Context Prompt: Generate 50 concepts from the attached paper.
Direction Builder Prompt: Use {context_pool} to focus on one research direction.
Format Prompt: Format {input_text} as a research idea.
Critique Prompt: Critique {idea} for novelty and feasibility.
Refine Prompt: Refine {idea} using {critique}.
Breed Prompt: Combine {ideas} into one stronger proposal.
Genotype Encode Prompt: Extract the core research pattern from {idea_content}.
Comparison Criteria: novelty, feasibility, relevance to source paper
"""

    async def fake_get_user_api_key(_uid):
        return "test-key"

    context_file = TemplateContextFile(
        filename="paper.pdf",
        content_type="application/pdf",
        data=b"%PDF test paper",
    )

    with patch("idea.template_manager.db.get_user_api_key", side_effect=fake_get_user_api_key), patch(
        "idea.template_manager.LLMWrapper",
        FakeLLM,
    ):
        response = asyncio.run(
            _generate_template_from_inputs(
                "follow-up paper ideas",
                UserInfo(uid="user-123"),
                context_file=context_file,
            )
        )

    assert response.status_code == 200
    assert captured["upload_bytes"] == b"%PDF test paper"
    assert captured["upload_config"] == {
        "mime_type": "application/pdf",
        "display_name": "paper.pdf",
    }
    assert captured["contents"][0].name == "files/paper"
    assert "paper.pdf" in captured["contents"][1]
    assert captured["deleted_file"] == "files/paper"


def test_template_context_upload_rejects_unsupported_file_type():
    class FakeUpload:
        filename = "archive.zip"
        content_type = "application/zip"

        async def read(self):
            return b"not supported"

    try:
        asyncio.run(_read_template_context_upload(FakeUpload()))
    except TemplateContextFileError as exc:
        assert "Unsupported context file type" in str(exc)
    else:
        raise AssertionError("Expected unsupported template context file to be rejected")


def test_parse_multipart_template_generation_request_without_extra_parser_dependency():
    boundary = "----idea-template-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="idea_type_suggestion"\r\n\r\n'
        "Follow-up paper ideas\r\n"
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="context_file"; filename="paper.pdf"\r\n'
        "Content-Type: application/pdf\r\n\r\n"
        "%PDF test paper\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    class FakeRequest:
        headers = {"content-type": f"multipart/form-data; boundary={boundary}"}

        async def body(self):
            return body

    idea_type, context_file = asyncio.run(
        _parse_multipart_template_generation_request(FakeRequest())
    )

    assert idea_type == "Follow-up paper ideas"
    assert context_file == TemplateContextFile(
        filename="paper.pdf",
        content_type="application/pdf",
        data=b"%PDF test paper",
    )
