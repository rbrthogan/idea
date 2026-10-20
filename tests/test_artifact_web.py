import asyncio
import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
import pytest

from idea.artifact_store import ArtifactStore
from idea.artifact_store import ArtifactStoreError
from idea.auth import UserInfo
from idea.evaluation import LLMCodeArtifactComponentConfig
from idea.models import Idea
from idea.user_state import UserEvolutionState
from idea.viewer import (
    ARTIFACT_STORE_ROOT_ENV,
    ENABLE_GEMINI_CODE_ARTIFACTS_ENV,
    ENABLE_LOCAL_ARTIFACT_REPAIR_ENV,
    ENABLE_LOCAL_ARTIFACTS_ENV,
    _artifact_store_for_evolution,
    _delete_owned_artifact_objects,
    _stored_evolution_artifact_context,
    _web_artifact_runtime,
    _web_evaluation_component_available,
    app,
    idea_to_dict,
    require_auth,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _artifact_reference(store: ArtifactStore, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "visualization.png").write_bytes(PNG_BYTES)
    imported = store.import_output(output_dir, "visualization.png", "image/png")
    return {
        "schema_version": 1,
        "id": "artifact-owned-primary",
        "role": "primary",
        "media_type": "image/png",
        "storage_key": imported["storage_key"],
        "sha256": imported["sha256"],
        "size_bytes": imported["size_bytes"],
        "producer_id": "render",
        "content_digest": "1" * 64,
        "metadata": {},
    }


def test_artifact_route_requires_owned_reference_and_verifies_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv(ARTIFACT_STORE_ROOT_ENV, str(tmp_path / "stores"))
    store = _artifact_store_for_evolution("user-1", "evolution-1", create=True)
    reference = _artifact_reference(store, tmp_path)
    evolution = {
        "history": [[{"id": "idea-1", "artifact_refs": [reference]}]],
        "population": [],
    }

    async def authenticated_user():
        return UserInfo(uid="user-1", email="user@example.com", is_admin=False)

    app.dependency_overrides[require_auth] = authenticated_user
    try:
        with patch(
            "idea.viewer.user_states.get",
            new=AsyncMock(return_value=UserEvolutionState()),
        ), patch(
            "idea.viewer.db.get_evolution",
            new=AsyncMock(return_value=evolution),
        ):
            client = TestClient(app)
            response = client.get(
                "/api/evolution/evolution-1/artifacts/artifact-owned-primary"
            )
            missing = client.get(
                "/api/evolution/evolution-1/artifacts/artifact-not-referenced"
            )
    finally:
        app.dependency_overrides.pop(require_auth, None)

    assert response.status_code == 200
    assert response.content == PNG_BYTES
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "sandbox" in response.headers["content-security-policy"]
    assert missing.status_code == 404


def test_evolution_artifact_cleanup_deletes_only_referenced_objects(tmp_path):
    store = ArtifactStore(tmp_path / "store")
    reference = _artifact_reference(store, tmp_path)
    evolution = {"history": [[{"artifact_refs": [reference]}]]}

    _delete_owned_artifact_objects(evolution, store)

    assert store.delete(reference["storage_key"]) is False


@pytest.mark.parametrize("mode,provider", [("remote", "google_generative_ai"),
    ("local", "google_generative_ai"), ("local", "openai"), ("remote", "openai")])
def test_archived_runtime_cannot_be_reenabled_by_legacy_flags(tmp_path, monkeypatch, mode, provider):
    component = LLMCodeArtifactComponentConfig(id="render", execution_profile="gemini-code-execution-v1")
    for flag in (ENABLE_GEMINI_CODE_ARTIFACTS_ENV, ENABLE_LOCAL_ARTIFACTS_ENV, ENABLE_LOCAL_ARTIFACT_REPAIR_ENV):
        monkeypatch.setenv(flag, "true")
    store_root = tmp_path / "stores"
    monkeypatch.setenv(ARTIFACT_STORE_ROOT_ENV, str(store_root))
    assert _web_evaluation_component_available(component) is False
    with pytest.raises(ArtifactStoreError, match="archived"):
        _web_artifact_runtime(user_id="user-1", evolution_id="evolution-1", api_key="synthetic-key",
                              execution_mode=mode, model_provider=provider)
    assert not store_root.exists()


def test_restored_evolution_resolves_artifact_source_provider_from_saved_model():
    with patch(
        "idea.viewer.db.get_evolution",
        new=AsyncMock(
            return_value={
                "config": {
                    "model_type": "gpt-5.6-luna",
                    "evaluation": {
                        "components": [
                            {"kind": "llm_code_artifact", "enabled": True}
                        ]
                    },
                }
            }
        ),
    ):
        provider, has_artifact_producer = asyncio.run(
            _stored_evolution_artifact_context("user-1", "evolution-1")
        )

    assert provider == "openai"
    assert has_artifact_producer is True


def test_execution_preferences_are_archived_even_with_legacy_flags(monkeypatch):
    monkeypatch.setenv(ENABLE_GEMINI_CODE_ARTIFACTS_ENV, "true")
    monkeypatch.setenv(ENABLE_LOCAL_ARTIFACTS_ENV, "true")
    async def authenticated_user():
        return UserInfo(uid="user-1", email="user@example.com", is_admin=False)
    app.dependency_overrides[require_auth] = authenticated_user
    save_preference = AsyncMock()
    settings = AsyncMock()
    try:
        with patch("idea.viewer.db.get_user_api_key", new=AsyncMock(return_value="synthetic-key")), \
             patch("idea.viewer.db.get_user_settings", new=settings), \
             patch("idea.viewer.db.save_user_execution_preference", new=save_preference):
            client = TestClient(app)
            status = client.get("/api/settings/status")
            saved = client.post("/api/settings/execution", json={"execution_mode": "local"})
    finally:
        app.dependency_overrides.pop(require_auth, None)
    assert status.status_code == 200
    assert status.json()["execution_options"]["archived"] is True
    assert status.json()["execution_options"]["local"]["available"] is False
    assert status.json()["execution_options"]["remote"]["available"] is False
    assert saved.status_code == 410
    assert saved.json()["code"] == "feature_archived"
    save_preference.assert_not_awaited()
    settings.assert_not_awaited()


def test_live_json_adapter_preserves_artifacts_and_evaluations():
    candidate = {
        "id": "idea-1",
        "idea": Idea(title="Prime map", content="Render it"),
        "artifact_ids": ["artifact-1"],
        "artifact_refs": [{"id": "artifact-1", "role": "primary"}],
        "evaluations": {"render": {"status": "ok"}},
    }

    result = idea_to_dict(candidate)

    assert result["artifact_ids"] == ["artifact-1"]
    assert result["artifact_refs"] == candidate["artifact_refs"]
    assert result["evaluations"] == candidate["evaluations"]


def test_idea_inspector_exposes_capability_driven_rich_tabs():
    viewer_html = Path("idea/static/html/viewer.html").read_text()
    viewer_js = Path("idea/static/js/viewer.js").read_text()

    assert 'data-idea-modal-tab="artifacts"' in viewer_html
    assert 'data-idea-modal-tab="code"' in viewer_html
    assert 'data-idea-modal-tab="evaluations"' in viewer_html
    assert "hydrateAuthenticatedArtifactImages" in viewer_js
    assert "getIdeaArtifactSignature" in viewer_js
    assert "output.textContent = source" in viewer_js
    assert "renderIdeaModalEvaluations" in viewer_js
    assert "Failure stage" in viewer_js
    assert "Execution attempts" in viewer_js
    assert "Repair attempts" in viewer_js
    assert "Visualization unavailable:" in viewer_js
    assert "getArtifactEvaluationFailure" in viewer_js


def test_cards_surface_failed_or_missing_primary_artifacts():
    viewer_js = Path("idea/static/js/viewer.js").read_text()
    viewer_css = Path("idea/static/css/viewer.css").read_text()

    assert "record.component_kind === 'llm_code_artifact'" in viewer_js
    assert "['ok', 'cached'].includes(record.status)" in viewer_js
    assert 'class="idea-card-artifact-failure" role="status"' in viewer_js
    assert "failure.setAttribute('role', 'status')" in viewer_js
    assert ".idea-card-artifact-failure" in viewer_css


def test_viewer_has_no_archived_execution_controls(monkeypatch):
    client = TestClient(app)
    monkeypatch.setenv(ENABLE_GEMINI_CODE_ARTIFACTS_ENV, "true")
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="enableVisualizationArtifacts"' not in response.text
    assert 'id="executionPreferenceSelect"' not in response.text
    viewer_js = Path("idea/static/js/viewer.js").read_text()
    assert "buildVisualizationArtifactEvaluation" not in viewer_js


def test_saved_evolution_identity_is_set_before_artifact_cards_render():
    viewer_js = Path("idea/static/js/viewer.js").read_text()
    history_loader = viewer_js.split("async function loadHistoryItemData", 1)[1]
    history_loader = history_loader.split("async function handleHistoryResume", 1)[0]

    identity_assignment = "currentEvolutionId = data.id || evolutionData.evolution_id || item.id;"
    assert identity_assignment in history_loader
    assert history_loader.index(identity_assignment) < history_loader.index(
        "renderGenerations(evolutionData.history);"
    )


def test_token_details_include_dynamic_artifact_usage_components():
    viewer_js = Path("idea/static/js/viewer.js").read_text()

    assert "Object.keys(tokenCounts.models || {})" in viewer_js
    assert "Artifact Render / Code Execution" in viewer_js
    assert "Tool/intermediate input" in viewer_js
