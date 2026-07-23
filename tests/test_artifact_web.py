import base64
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from idea.artifact_store import ArtifactStore
from idea.auth import UserInfo
from idea.evaluation import LLMCodeArtifactComponentConfig
from idea.models import Idea
from idea.user_state import UserEvolutionState
from idea.viewer import (
    ARTIFACT_STORE_ROOT_ENV,
    ENABLE_GEMINI_CODE_ARTIFACTS_ENV,
    _artifact_store_for_evolution,
    _delete_owned_artifact_objects,
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


def test_web_artifact_runtime_is_explicitly_opt_in(tmp_path, monkeypatch):
    component = LLMCodeArtifactComponentConfig(
        id="render",
        execution_profile="gemini-code-execution-v1",
    )
    monkeypatch.delenv(ENABLE_GEMINI_CODE_ARTIFACTS_ENV, raising=False)
    assert _web_evaluation_component_available(component) is False

    monkeypatch.setenv(ENABLE_GEMINI_CODE_ARTIFACTS_ENV, "true")
    monkeypatch.setenv(ARTIFACT_STORE_ROOT_ENV, str(tmp_path / "stores"))
    runners, store = _web_artifact_runtime(
        user_id="user-1",
        evolution_id="evolution-1",
        api_key="test-key",
    )

    assert _web_evaluation_component_available(component) is True
    assert runners["llm_code_artifact"].enabled is True
    assert isinstance(store, ArtifactStore)
    assert "user-1" not in str(store.root)


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


def test_viewer_exposes_server_gated_artifact_opt_in(monkeypatch):
    client = TestClient(app)

    monkeypatch.delenv(ENABLE_GEMINI_CODE_ARTIFACTS_ENV, raising=False)
    disabled = client.get("/")
    assert disabled.status_code == 200
    assert 'id="enableVisualizationArtifacts"' in disabled.text
    artifact_switch = disabled.text.split(
        'id="enableVisualizationArtifacts"', 1
    )[1].split(">", 1)[0]
    assert "disabled" in artifact_switch
    assert "checked" not in artifact_switch

    monkeypatch.setenv(ENABLE_GEMINI_CODE_ARTIFACTS_ENV, "true")
    enabled = client.get("/")
    artifact_switch = enabled.text.split(
        'id="enableVisualizationArtifacts"', 1
    )[1].split(">", 1)[0]
    assert "disabled" not in artifact_switch
    assert "checked" in artifact_switch

    viewer_js = Path("idea/static/js/viewer.js").read_text()
    assert "buildVisualizationArtifactEvaluation" in viewer_js
    assert "requestBody.evaluation" in viewer_js


def test_saved_evolution_identity_is_set_before_artifact_cards_render():
    viewer_js = Path("idea/static/js/viewer.js").read_text()
    history_loader = viewer_js.split("async function loadHistoryItemData", 1)[1]
    history_loader = history_loader.split("async function handleHistoryResume", 1)[0]

    identity_assignment = "currentEvolutionId = data.id || evolutionData.evolution_id || item.id;"
    assert identity_assignment in history_loader
    assert history_loader.index(identity_assignment) < history_loader.index(
        "renderGenerations(evolutionData.history);"
    )
    assert "requires: ['artifact.primary']" in viewer_js
