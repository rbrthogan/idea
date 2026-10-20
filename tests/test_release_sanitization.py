from pathlib import Path
import subprocess

import pytest

from scripts.check_release import findings, history_files, local_only, source_files


@pytest.mark.parametrize("path", [
    ".env", ".env.production", "terraform/backend.hcl",
    "terraform/terraform.tfvars", "terraform/default.tfstate.backup",
    ".deploy/release/plan.json", "output/run/progress.json",
    "firebase-export-123/auth_export/accounts.json",
    "archive/internal-docs/Deployment.md", "service-account.json",
    "idea/static/js/auth.js",
])
def test_private_release_files_are_rejected(path):
    assert local_only(path)


@pytest.mark.parametrize("path", [
    ".env.example", "terraform/backend.hcl.example", "terraform/backend.tf",
    "terraform/.terraform.lock.hcl", "terraform/terraform.tfvars.example",
    "archive/prime-visualization/README.md", "idea/provider_credentials.py",
    "tests/test_provider_api_keys.py", "docs/LocalDevelopment.md",
])
def test_portable_source_and_examples_are_allowed(path):
    assert not local_only(path)


def test_secret_findings_never_contain_secret_material():
    secret = "AIza" + "a" * 35
    result = findings("config.py", f'KEY = "{secret}"'.encode())
    assert result == ["Google API key"]
    assert secret not in repr(result)
    assert findings("fixture.py", b'"sk-misfiled-by-stale-client"') == []


def test_index_scan_uses_staged_content_and_includes_forced_private_files(tmp_path: Path):
    def run(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init")
    (tmp_path / ".gitignore").write_text(".env\n")
    (tmp_path / ".env").write_text("private staged content")
    run("add", "-f", ".env")
    (tmp_path / ".env").write_text("different working content")
    assert list(source_files(tmp_path, index=True)) == [(".env", b"private staged content")]
    assert findings(*list(source_files(tmp_path, index=True))[0]) == ["local-only file"]


def test_history_scan_finds_content_removed_from_the_latest_commit(tmp_path: Path):
    def run(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), "-c", "user.name=Release Test",
             "-c", "user.email=release@example.test", *args],
            check=True, capture_output=True,
        )

    run("init")
    (tmp_path / "config.py").write_text('KEY = "' + "AIza" + "a" * 35 + '"')
    run("add", "config.py")
    run("commit", "-m", "fixture")
    (tmp_path / "config.py").write_text("KEY = None")
    run("add", "config.py")
    run("commit", "-m", "remove fixture")
    assert findings("config.py", (tmp_path / "config.py").read_bytes()) == []
    historical_findings = [
        (path, issue) for path, content in history_files(tmp_path)
        for issue in findings(path, content)
    ]
    assert historical_findings == [("config.py", "Google API key")]
