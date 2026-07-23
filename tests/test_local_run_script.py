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
