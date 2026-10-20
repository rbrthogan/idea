"""Offline release harness: fake every Terraform/gcloud call; never access cloud/state."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("idea_deploy", ROOT / "scripts/deploy.py")
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)
DIGEST = "sha256:" + "a" * 64
REPOSITORY = "europe-west1-docker.pkg.dev/test-project/images"
IMAGE = REPOSITORY + "/idea-evolution@" + DIGEST
CONFIG = {"projectId": "test-project", "apiKey": "public-browser-key", "authDomain": "test.example", "appId": "public-app"}


def sample_plan():
    return {
        "planned_values": {"root_module": {"resources": [
            {"address": deploy.WEB, "values": {"template": [{"containers": [{"image": IMAGE}]}]}},
            {"address": deploy.WORKER, "values": {"template": [{"template": [{"containers": [{"image": IMAGE}]}]}]}},
        ]}},
        "resource_changes": [
            {"address": deploy.WEB, "change": {"actions": ["update"], "before": {"template": [{"containers": [{"image": "old-web"}]}]}}},
            {"address": deploy.WORKER, "change": {"actions": ["create"], "before": None}},
            {"address": "google_firestore_backup_schedule.weekly[0]", "change": {"actions": ["create"]}},
            {"address": "random_password.encryption_key", "change": {"actions": ["no-op"], "before": {"result": "SENSITIVE-TEST-SENTINEL"}}},
        ],
    }


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "project with spaces"
    root.mkdir()
    for relative, value in {
        "terraform/.terraform.lock.hcl": "# synthetic provider lock",
        "terraform/terraform.tfvars": 'smtp_password = "SENSITIVE-TEST-SENTINEL"',
        "idea/viewer.py": "app = object()\n",
        "idea/worker.py": "pass\n",
        "idea/evaluation/runner.py": "pass\n",
        "idea/static/html/viewer.html": "<html></html>",
        "idea/static/css/workbench.css": "body{}",
        "idea/static/js/auth.js": "SOURCE-FIREBASE-CONFIG-MUST-NOT-CHANGE",
        "idea/static/js/viewer.js": "const app = {};",
        "idea/static/img/icon.svg": "<svg/>",
        "idea/prompts/templates/drabble.yaml": "domain: story",
        "idea/service-account.json": "SENSITIVE-TEST-SENTINEL",
        "idea/static/js/service-account.json": "SENSITIVE-TEST-SENTINEL",
        "idea/.env": "SENSITIVE-TEST-SENTINEL",
        "idea/__pycache__/code.py": "SENSITIVE-TEST-SENTINEL",
        "idea/.private/secret.py": "SENSITIVE-TEST-SENTINEL",
        "data/users.json": "SENSITIVE-TEST-SENTINEL",
        ".deploy/old/plan.tfplan": "SENSITIVE-TEST-SENTINEL",
    }.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(value)
    for name in deploy.BUILD_INPUTS:
        (root / name).write_text((ROOT / name).read_text())
    return root


class FakeCloud:
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.resources = set(deploy.EXISTING_RESOURCES)
        self.plan = sample_plan()
        self.outputs = {"project_id": "test-project", "repository_url": REPOSITORY,
                        "firebase_config_json": json.dumps(CONFIG), "service_url": "https://test.example"}
        self.build_contents = None
        self.fail_apply = False

    def run(self, args, **kwargs):
        self.calls.append(args)
        out = ""
        code = 0
        if args[0] == "terraform":
            assert args[1] == f"-chdir={self.root / 'terraform'}"
            action = args[2]
            if action == "state":
                out = "\n".join(sorted(self.resources))
            elif action == "output":
                out = json.dumps(self.outputs[args[-1]])
            elif action == "plan":
                path = Path(next(arg[5:] for arg in args if arg.startswith("-out=")))
                path.write_bytes(b"SYNTHETIC-SAVED-PLAN")
                out = "Plan: worker and backups added, web updated."
            elif action == "show":
                out = json.dumps(self.plan) if "-json" in args else "Reviewed plan text."
            elif action == "apply" and self.fail_apply:
                code = 1
                out = "Synthetic apply failure."
        elif args[:3] == ["gcloud", "builds", "submit"]:
            context = Path(args[-1])
            for path in context.rglob("*"):
                assert stat.S_IMODE(path.stat().st_mode) == (0o755 if path.is_dir() else 0o644)
            self.build_contents = {path.relative_to(context).as_posix(): path.read_bytes()
                                   for path in context.rglob("*") if path.is_file()}
        elif args[:5] == ["gcloud", "artifacts", "docker", "images", "describe"]:
            out = json.dumps({"image_summary": {"digest": DIGEST}})
        else:
            raise AssertionError(f"Unexpected external command: {args}")
        if hasattr(kwargs.get("stdout"), "write"):
            kwargs["stdout"].write(out)
            kwargs["stdout"].flush()
            return subprocess.CompletedProcess(args, code, stdout=None, stderr=None)
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="")


@pytest.fixture
def cloud(source, monkeypatch):
    fake = FakeCloud(source)
    monkeypatch.setattr(deploy.subprocess, "run", fake.run)
    monkeypatch.setattr(deploy.shutil, "which", lambda name: "/fake/" + name)
    return fake


def test_default_invocation_is_help_and_never_contacts_cloud(source, cloud):
    assert deploy.main([], root=source) == 0
    assert cloud.calls == []


def test_terraform_log_receives_progress_before_command_returns(tmp_path, monkeypatch):
    log = tmp_path / "private-progress.log"

    def running_command(args, **kwargs):
        assert kwargs["stderr"] == subprocess.STDOUT
        assert stat.S_IMODE(log.stat().st_mode) == 0o600
        kwargs["stdout"].write("Applying worker...\n")
        kwargs["stdout"].flush()
        assert log.read_text() == "Applying worker...\n"
        return subprocess.CompletedProcess(args, 0, stdout=None, stderr=None)

    monkeypatch.setattr(deploy.subprocess, "run", running_command)
    assert deploy.command(["terraform", "apply", "test-plan"], log=log) == ""


def test_check_validates_existing_state_without_build_plan_or_apply(source, cloud):
    assert deploy.main(["check"], root=source) == 0
    assert any("-lockfile=readonly" in call for call in cloud.calls)
    assert all(call[0] == "terraform" and call[2] not in {"plan", "apply"} for call in cloud.calls)


@pytest.mark.parametrize("missing", [deploy.EXISTING_RESOURCES, {"random_password.encryption_key"}, {deploy.WEB}])
def test_missing_state_stops_before_build_or_plan(source, cloud, missing, capsys):
    cloud.resources -= missing
    assert deploy.main(["plan"], root=source) == 1
    assert "Recover the correct backend/state" in capsys.readouterr().err
    assert all(call[0] == "terraform" and call[2] not in {"plan", "apply"} for call in cloud.calls)


def test_build_context_excludes_secrets_and_preserves_checkout_firebase_config(source, cloud):
    original = (source / "idea/static/js/auth.js").read_text()
    release = deploy.plan_release(source, None)
    assert cloud.build_contents
    assert all(b"SENSITIVE-TEST-SENTINEL" not in value for value in cloud.build_contents.values())
    assert "idea/service-account.json" not in cloud.build_contents
    assert "idea/static/css/workbench.css" in cloud.build_contents
    assert "idea/worker.py" in cloud.build_contents
    assert "idea/prompts/templates/drabble.yaml" in cloud.build_contents
    assert json.dumps(CONFIG, ensure_ascii=True, sort_keys=True).encode() in cloud.build_contents["idea/static/js/auth.js"]
    assert (source / "idea/static/js/auth.js").read_text() == original
    assert (source / "terraform/terraform.tfvars").read_text() == 'smtp_password = "SENSITIVE-TEST-SENTINEL"'
    manifest = json.loads((release / "manifest.json").read_text())
    assert manifest["image"] == IMAGE
    assert manifest["previous_images"][deploy.WEB] == ["old-web"]
    assert "SENSITIVE-TEST-SENTINEL" not in (release / "manifest.json").read_text()
    assert all("SENSITIVE-TEST-SENTINEL" not in path.read_text() for path in release.iterdir())
    assert not any(call[0] == "terraform" and call[2] == "apply" for call in cloud.calls)
    assert all("--project" in call and "test-project" in call for call in cloud.calls if call[0] == "gcloud")
    for path in [release, *release.iterdir()]:
        assert stat.S_IMODE(path.stat().st_mode) & 0o077 == 0


@pytest.mark.parametrize("path", ["idea/linked.py", "idea/static/js/linked.js", "idea/linked-directory"])
def test_context_refuses_symlinks_without_following_them(source, tmp_path, path):
    target = tmp_path / "external"
    target.mkdir()
    (target / "secret.py").write_text("SENSITIVE-TEST-SENTINEL")
    (source / path).symlink_to(target if path.endswith("directory") else target / "secret.py")
    with pytest.raises(deploy.ReleaseError, match="symlink|regular file"):
        deploy.stage_context(source, tmp_path / "context", CONFIG)


def test_wrong_firebase_project_stops_before_build(source, cloud):
    cloud.outputs["firebase_config_json"] = json.dumps({**CONFIG, "projectId": "different-project"})
    assert deploy.main(["plan"], root=source) == 1
    assert not any(call[0] == "gcloud" for call in cloud.calls)


@pytest.mark.parametrize("encoded", [False, True])
def test_configured_auth_domain_reaches_build_without_changing_identity_or_checkout(source, cloud, encoded):
    configured = {**CONFIG, "authDomain": "idea-evolution.web.app"}
    cloud.outputs["firebase_config_json"] = json.dumps(configured) if encoded else configured
    original = (source / "idea/static/js/auth.js").read_text()

    deploy.plan_release(source, None)

    generated = cloud.build_contents["idea/static/js/auth.js"].decode()
    config_json = generated.split("const FIREBASE_CONFIG = ", 1)[1].removesuffix(";\n")
    assert json.loads(config_json) == configured
    assert (source / "idea/static/js/auth.js").read_text() == original
    assert not any(call[0] == "terraform" and call[2] == "apply" for call in cloud.calls)


@pytest.mark.parametrize("action", [["delete"], ["delete", "create"], ["create", "delete"]])
def test_plan_refuses_any_resource_deletion_or_replacement(action):
    plan = sample_plan()
    plan["resource_changes"][0]["change"]["actions"] = action
    with pytest.raises(deploy.ReleaseError, match="deletion or replacement"):
        deploy.inspect_plan(plan, IMAGE)


@pytest.mark.parametrize("address", sorted(deploy.KEY_RESOURCES))
def test_plan_refuses_encryption_key_mutations(address):
    plan = sample_plan()
    plan["resource_changes"].append({"address": address, "change": {"actions": ["update"]}})
    with pytest.raises(deploy.ReleaseError, match="encryption-key changes"):
        deploy.inspect_plan(plan, IMAGE)


def test_plan_allows_only_retention_metadata_hardening_for_existing_key_version():
    before = {"deletion_policy": "DELETE", "secret_data": "SENSITIVE-TEST-SENTINEL", "secret": "key", "version": "1"}
    after = {**before, "deletion_policy": "ABANDON"}
    resource = {"address": "google_secret_manager_secret_version.encryption_key_version",
                "change": {"actions": ["update"], "before": before, "after": after, "after_unknown": {}}}
    plan = sample_plan()
    plan["resource_changes"].append(resource)
    summary = deploy.inspect_plan(plan, IMAGE)
    assert resource["address"] in [change["address"] for change in summary["changes"]]
    assert "SENSITIVE-TEST-SENTINEL" not in json.dumps(summary)
    for key, value in (("secret_data", "changed"), ("version", "2"), ("secret", "different-key")):
        resource["change"]["after"] = {**after, key: value}
        with pytest.raises(deploy.ReleaseError, match="encryption-key changes"):
            deploy.inspect_plan(plan, IMAGE)
    resource["change"]["after"] = after
    resource["change"]["after_unknown"] = {"secret_data": True}
    with pytest.raises(deploy.ReleaseError, match="encryption-key changes"):
        deploy.inspect_plan(plan, IMAGE)


def test_plan_requires_both_web_and_worker_using_same_digest():
    plan = sample_plan()
    plan["planned_values"]["root_module"]["resources"][1]["values"]["template"][0]["template"][0]["containers"][0]["image"] = "other-image:latest"
    with pytest.raises(deploy.ReleaseError, match="same immutable image"):
        deploy.inspect_plan(plan, IMAGE)
    plan["planned_values"]["root_module"]["resources"].pop()
    with pytest.raises(deploy.ReleaseError, match="same immutable image"):
        deploy.inspect_plan(plan, IMAGE)


def test_apply_exact_reviewed_plan_without_rebuild_replan_or_autoapprove(source, cloud):
    release = deploy.plan_release(source, DIGEST)
    cloud.calls.clear()
    deploy.apply_release(source, str(release))
    applies = [call for call in cloud.calls if call[0] == "terraform" and call[2] == "apply"]
    assert len(applies) == 1
    assert applies[0][-1] == str(release / "release.tfplan")
    assert not any("-auto-approve" in call or any(arg.startswith("-target") for arg in call) for call in cloud.calls)
    assert all(call[0] == "terraform" and call[2] != "plan" for call in cloud.calls)
    assert (release / "applied.json").is_file()
    with pytest.raises(deploy.ReleaseError, match="already been applied"):
        deploy.apply_release(source, str(release))


def test_tampered_plan_cannot_be_applied(source, cloud):
    release = deploy.plan_release(source, DIGEST)
    (release / "release.tfplan").write_bytes(b"CHANGED")
    cloud.calls.clear()
    with pytest.raises(deploy.ReleaseError, match="does not match"):
        deploy.apply_release(source, str(release))
    assert cloud.calls == []


def test_apply_rechecks_state_and_project(source, cloud):
    release = deploy.plan_release(source, DIGEST)
    cloud.outputs.update(project_id="different-project", repository_url="europe-west1-docker.pkg.dev/different-project/images")
    cloud.calls.clear()
    with pytest.raises(deploy.ReleaseError, match="different deployment"):
        deploy.apply_release(source, str(release))
    assert not any(call[0] == "terraform" and call[2] == "apply" for call in cloud.calls)


def test_apply_failure_does_not_replan_or_claim_success(source, cloud):
    release = deploy.plan_release(source, DIGEST)
    cloud.fail_apply = True
    cloud.calls.clear()
    assert deploy.main(["apply", str(release)], root=source) == 1
    assert not (release / "applied.json").exists()
    assert sum(call[0] == "terraform" and call[2] == "apply" for call in cloud.calls) == 1
    assert not any(call[0] == "terraform" and call[2] == "plan" for call in cloud.calls)


def test_reuse_digest_verifies_artifact_without_building(source, cloud):
    release = deploy.plan_release(source, DIGEST)
    assert cloud.build_contents is None
    assert any(IMAGE in call and "describe" in call for call in cloud.calls)
    assert json.loads((release / "manifest.json").read_text())["source_sha256"] is None


@pytest.mark.parametrize("digest", ["latest", "sha256:" + "f" * 63, "sha256:" + "G" * 64, "sha256:" + "f" * 64 + ";echo bad"])
def test_invalid_digest_fails_before_any_commands(source, cloud, digest):
    assert deploy.main(["plan", "--image-digest", digest], root=source) == 1
    assert cloud.calls == []


def test_shell_wrapper_is_valid_and_help_works_from_other_directory(tmp_path):
    subprocess.run(["bash", "-n", str(ROOT / "deploy.sh")], check=True)
    result = subprocess.run(["bash", str(ROOT / "deploy.sh"), "--help"], cwd=tmp_path,
                            capture_output=True, text=True, check=True)
    assert "{check,plan,apply}" in result.stdout


def test_cloud_upload_ignore_allows_runtime_only(tmp_path):
    # gcloudignore follows gitignore semantics; this catches missing negated parents.
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / ".gitignore").write_text((ROOT / ".gcloudignore").read_text())
    allowed = ["Dockerfile", "uv.lock", "pyproject.toml", "idea/viewer.py", "idea/worker.py",
               "idea/evaluation/processes.py", "idea/static/js/auth.js", "idea/static/css/workbench.css",
               "idea/prompts/templates/drabble.yaml"]
    excluded = [".env", "terraform/terraform.tfvars", ".deploy/releases/x/release.tfplan", "data/users.json",
                "idea/service-account.json", "idea/static/js/service-account.json", "tests/test_deployment_release.py"]
    result = subprocess.run(["git", "-C", str(tmp_path), "check-ignore", "--no-index", *allowed, *excluded],
                            capture_output=True, text=True, check=False)
    assert set(result.stdout.splitlines()) == set(excluded)


@pytest.mark.skipif(shutil.which("gcloud") is None, reason="Optional offline SDK verification requires gcloud")
def test_actual_gcloud_upload_selection_excludes_sensitive_files(source, tmp_path):
    context = tmp_path / "isolated-context"
    context.mkdir()
    deploy.stage_context(source, context, CONFIG)
    excluded = {".env", "terraform/terraform.tfstate", ".deploy/releases/test/release.tfplan",
                "data/keys.json", "idea/service-account.json"}
    for name in excluded:
        path = context / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("SYNTHETIC-SENSITIVE-SENTINEL")
    result = subprocess.run(["gcloud", "meta", "list-files-for-upload", "."], cwd=context,
                            capture_output=True, text=True, check=True,
                            env={**os.environ, "CLOUDSDK_CONFIG": str(tmp_path / "sdk-config"),
                                 "CLOUDSDK_CORE_DISABLE_USAGE_REPORTING": "true",
                                 "CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK": "true"})
    selected = {line.removeprefix("./") for line in result.stdout.splitlines()}
    required = {"Dockerfile", "uv.lock", "pyproject.toml", "idea/viewer.py", "idea/worker.py",
                "idea/prompts/templates/drabble.yaml", "idea/static/js/auth.js",
                "idea/static/html/viewer.html", "idea/static/css/workbench.css", "idea/static/img/icon.svg"}
    assert required <= selected
    assert not excluded.intersection(selected)


def test_build_requires_locked_dependencies_and_one_python_base():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "uv sync --locked --no-dev --no-install-project" in dockerfile
    assert "FROM python-base AS builder" in dockerfile
    assert "FROM python-base AS runtime" in dockerfile
    assert 'exec uvicorn' in dockerfile and '${PORT:-8080}' in dockerfile
    assert ".terraform.lock.hcl" not in (ROOT / ".gitignore").read_text().splitlines()
