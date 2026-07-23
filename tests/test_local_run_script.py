import subprocess
from pathlib import Path


def test_local_run_script_uses_uv_environment():
    script = Path("run_local.sh").read_text()

    assert "command -v uv" in script
    assert "uv run uvicorn idea.viewer:app" in script
    assert "\nuvicorn idea.viewer:app" not in script


def test_local_run_script_never_uses_known_encryption_key_with_remote_firestore():
    script = Path("run_local.sh").read_text()

    remote_branch = script.split('if [ "$USE_EMULATOR" = false ]; then', 1)[1]
    remote_body, emulator_body = remote_branch.split("else", 1)

    assert 'if [ -z "${ENCRYPTION_KEY:-}" ]; then' in remote_body
    assert "ENCRYPTION_KEY must be set when using remote Firestore" in remote_body
    assert '${#ENCRYPTION_KEY}' in remote_body
    assert "local-dev-only-encryption-key" not in remote_body
    assert 'export ENCRYPTION_KEY="${ENCRYPTION_KEY:-local-dev-only-encryption-key-32-bytes-minimum}"' in emulator_body


def test_local_run_script_documents_visual_artifact_flag():
    result = subprocess.run(
        ["bash", "run_local.sh", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Usage: ./run_local.sh [--visual-artifacts]" in result.stdout
    assert "--visual-artifacts" in result.stdout


def test_local_run_script_visual_artifact_flag_sets_safe_environment():
    script = Path("run_local.sh").read_text()

    assert "ENABLE_VISUAL_ARTIFACTS=false" in script
    assert "export IDEA_ENABLE_GEMINI_CODE_ARTIFACTS=true" in script
    assert (
        'export IDEA_ARTIFACT_STORE_ROOT="${IDEA_ARTIFACT_STORE_ROOT:-${PWD}/data/artifacts}"'
        in script
    )


def test_local_run_script_rejects_unknown_flags():
    result = subprocess.run(
        ["bash", "run_local.sh", "--not-a-real-option"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unknown option: --not-a-real-option" in result.stderr
