from pathlib import Path


def test_local_run_script_uses_uv_environment():
    script = Path("run_local.sh").read_text()

    assert "command -v uv" in script
    assert "uv run uvicorn idea.viewer:app" in script
    assert "\nuvicorn idea.viewer:app" not in script
