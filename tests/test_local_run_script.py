import os
import subprocess
import shutil
from pathlib import Path
import pytest


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


def test_local_run_script_does_not_advertise_archived_visual_artifacts():
    result = subprocess.run(["bash", "run_local.sh", "--help"],
                            check=True, capture_output=True, text=True)
    assert "Usage: ./run_local.sh" in result.stdout
    assert "--visual-artifacts" not in result.stdout


def test_local_run_script_rejects_archived_visual_artifact_flag_before_startup():
    result = subprocess.run(["bash", "run_local.sh", "--visual-artifacts"],
                            check=False, capture_output=True, text=True)
    assert result.returncode == 2
    assert "Visualization execution is archived" in result.stderr
    script = Path("run_local.sh").read_text()
    assert "export IDEA_ENABLE_GEMINI_CODE_ARTIFACTS=true" not in script
    assert "docker build" not in script
    assert "ollama pull" not in script


def test_local_run_script_preflights_and_waits_for_firestore_emulator():
    script = Path("run_local.sh").read_text()

    assert "java -version" in script
    assert "JDK 21 or later" in script
    assert '--project "$GOOGLE_CLOUD_PROJECT"' in script
    assert 'kill -0 "$EMULATOR_PID"' in script
    assert '"http://${FIRESTORE_EMULATOR_HOST}/"' in script
    assert "Reusing existing Firestore emulator" in script
    assert script.index("Reusing existing Firestore emulator") < script.index(
        "firebase emulators:start"
    )


def test_local_run_script_reports_missing_java_before_starting_app(tmp_path):
    for command, exit_code in (("uv", 0), ("firebase", 0), ("java", 1)):
        executable = tmp_path / command
        executable.write_text(f"#!/bin/sh\nexit {exit_code}\n")
        executable.chmod(0o755)

    result = subprocess.run(
        ["bash", "run_local.sh"],
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{tmp_path}:/usr/bin:/bin",
            "FIRESTORE_EMULATOR_HOST": "127.0.0.1:1",
            "IDEA_LOCAL_ENV_LOADED": "1",
        },
    )

    assert result.returncode == 1
    assert "A working Java JDK is required" in result.stderr
    assert "Starting FastAPI development server" not in result.stdout


def test_local_run_script_enables_non_admin_dev_auth_for_emulator():
    script = Path("run_local.sh").read_text()

    assert 'export IDEA_DEV_AUTH="${IDEA_DEV_AUTH:-1}"' in script
    assert "IDEA_DEV_AUTH_ADMIN" not in script
    assert 'IDEA_BIND_HOST="${IDEA_BIND_HOST:-127.0.0.1}"' in script


def test_local_run_script_rejects_unknown_flags():
    result = subprocess.run(
        ["bash", "run_local.sh", "--not-a-real-option"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "Unknown option: --not-a-real-option" in result.stderr


@pytest.mark.parametrize('has_saved_export', [False, True])
def test_local_startup_honors_dotenv_and_retains_emulator_data(tmp_path, has_saved_export):
    real_uv = shutil.which('uv')
    if not real_uv:
        pytest.skip('uv is required to verify its dotenv parser')
    root = Path(__file__).resolve().parents[1]
    fake_bin = tmp_path / 'bin'
    fake_bin.mkdir()
    # All credentials are synthetic; no application/provider calls are made.
    key = 'synthetic-local-encryption-secret-12345678'
    (tmp_path / '.env').write_text(f'ENCRYPTION_KEY={key}\nENCRYPTION_SALT=synthetic-salt\n')
    (tmp_path / 'firebase.json').write_text('{}')
    saved = tmp_path / 'data/firestore-emulator'
    if has_saved_export:
        saved.mkdir(parents=True)
        (saved / 'firebase-export-metadata.json').write_text('{}')
    commands = {
        'uv': '''#!/bin/bash
if [ "$2" = --no-project ]; then exec "$IDEA_TEST_UV" "$@"; fi
printf '%s' "$ENCRYPTION_KEY:$ENCRYPTION_SALT" > "$IDEA_TEST_APP_ENV"
''',
        'firebase': '''#!/bin/bash
printf '%s\\n' "$@" > "$IDEA_TEST_FIREBASE_ARGS"
touch "$IDEA_TEST_READY"
trap 'exit 0' TERM INT
while :; do sleep 0.1; done
''',
        'java': '#!/bin/sh\nexit 0\n',
        'curl': '#!/bin/sh\ntest -f "$IDEA_TEST_READY"\n',
    }
    for name, source in commands.items():
        exe = fake_bin / name
        exe.write_text(source)
        exe.chmod(0o755)
    env = {**os.environ, 'PATH': f'{fake_bin}:/usr/bin:/bin',
        'IDEA_TEST_UV': real_uv, 'UV_CACHE_DIR': str(tmp_path / 'uv-cache'),
        'IDEA_TEST_APP_ENV': str(tmp_path / 'app-env'),
        'IDEA_TEST_FIREBASE_ARGS': str(tmp_path / 'firebase-args'),
        'IDEA_TEST_READY': str(tmp_path / 'ready')}
    for name in ('ENCRYPTION_KEY', 'ENCRYPTION_SALT', 'IDEA_LOCAL_ENV_LOADED', 'IDEA_EMULATOR_DATA_DIR'):
        env.pop(name, None)
    result = subprocess.run(['bash', str(root / 'run_local.sh')], cwd=tmp_path,
        env=env, capture_output=True, text=True, timeout=12)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / 'app-env').read_text() == f'{key}:synthetic-salt'
    args = (tmp_path / 'firebase-args').read_text().splitlines()
    assert f'--export-on-exit={saved}' in args
    assert ('--import' in args) is has_saved_export
    if has_saved_export:
        assert args[args.index('--import') + 1] == str(saved)
