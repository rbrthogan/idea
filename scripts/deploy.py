#!/usr/bin/env python3
"""Build an immutable release, review a saved Terraform plan, then apply it.

Uses only the standard library. No command bootstraps resources or reads secret
payloads. Terraform plans still contain sensitive state and are kept private.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import uuid

ROOT = Path(__file__).resolve().parents[1]
DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
EXISTING_RESOURCES = {
    "google_firestore_database.database",
    "random_password.encryption_key",
    "google_secret_manager_secret.encryption_key",
    "google_secret_manager_secret_version.encryption_key_version",
    "google_cloud_run_v2_service.default",
}
KEY_RESOURCES = {
    "random_password.encryption_key",
    "google_secret_manager_secret.encryption_key",
    "google_secret_manager_secret_version.encryption_key_version",
}
WEB = "google_cloud_run_v2_service.default"
WORKER = "google_cloud_run_v2_job.evolution_worker"
BUILD_INPUTS = ("Dockerfile", "pyproject.toml", "uv.lock", ".dockerignore", ".gcloudignore")


class ReleaseError(Exception):
    pass


def release_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]


def private_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=False, mode=0o700)
    path.chmod(0o700)
    return path


def write_private(path: Path, value: str) -> None:
    # New files inherit umask 077 too; chmod also protects an existing file.
    with path.open("w", encoding="utf-8") as handle:
        os.chmod(path, 0o600)
        handle.write(value)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command(args: list[str], *, log: Path | None = None) -> str:
    if log:
        with log.open("a", encoding="utf-8") as handle:
            os.chmod(log, 0o600)
            # Write directly while Terraform runs so operators can tail progress.
            result = subprocess.run(args, text=True, stdout=handle, stderr=subprocess.STDOUT)
    else:
        result = subprocess.run(args, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        operation = " ".join(args[:3])
        detail = f" See {log} for diagnostics." if log else " Check credentials and the configured Terraform backend."
        raise ReleaseError(f"{operation} failed (exit {result.returncode}).{detail}")
    return (result.stdout or "").strip()


def tf(root: Path, *args: str, log: Path | None = None) -> str:
    return command(["terraform", f"-chdir={root / 'terraform'}", *args], log=log)


def public_output(root: Path, name: str):
    try:
        return json.loads(tf(root, "output", "-json", name))
    except json.JSONDecodeError as error:
        raise ReleaseError(f"Terraform output {name} is not valid JSON.") from error


def check_existing_state(root: Path, log: Path) -> dict:
    if not shutil.which("terraform"):
        raise ReleaseError("Terraform is required.")
    if not (root / "terraform/.terraform.lock.hcl").is_file():
        raise ReleaseError("The committed terraform/.terraform.lock.hcl is missing. Restore it before releasing.")
    tf(root, "init", "-input=false", "-lockfile=readonly", log=log)
    resources = set(tf(root, "state", "list").splitlines())
    missing = sorted(EXISTING_RESOURCES - resources)
    if missing:
        raise ReleaseError(
            "Existing deployment state is incomplete; refusing to provision or rebuild. "
            "Recover the correct backend/state or import existing resources first. Missing: "
            + ", ".join(missing)
            + ". Fresh-project bootstrap is a separate reviewed infrastructure operation (DEPLOY.md)."
        )
    tf(root, "validate", "-no-color", log=log)
    project = public_output(root, "project_id")
    repository = public_output(root, "repository_url")
    if not isinstance(project, str) or not re.fullmatch(r"[a-z][a-z0-9-]{4,61}[a-z0-9]", project):
        raise ReleaseError("The existing state's project_id output is invalid.")
    if not isinstance(repository, str) or not re.fullmatch(
        rf"[a-z0-9-]+-docker\.pkg\.dev/{re.escape(project)}/[a-z0-9._-]+", repository
    ):
        raise ReleaseError("The repository_url does not belong to the state's project_id.")
    return {"project_id": project, "repository_url": repository}


def firebase_config(root: Path, project: str) -> dict:
    value = public_output(root, "firebase_config_json")
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ReleaseError("Firebase config is not valid JSON.") from error
    if not isinstance(value, dict) or value.get("projectId") != project:
        raise ReleaseError("Firebase client configuration does not match the deployment project.")
    if any(not isinstance(value.get(key), str) or not value[key].strip()
           for key in ("apiKey", "authDomain", "appId")):
        raise ReleaseError("Firebase client configuration is incomplete.")
    # Only browser-public Firebase fields are ever copied into the build context.
    return {key: value[key] for key in (
        "apiKey", "authDomain", "projectId", "storageBucket", "messagingSenderId", "appId", "measurementId"
    ) if key in value}


def source_allowed(relative: Path) -> bool:
    parts = relative.parts
    if not parts or parts[0] != "idea" or any(part.startswith(".") or part == "__pycache__" for part in parts):
        return False
    if relative.suffix == ".py":
        return True
    if parts[:3] == ("idea", "prompts", "templates") and len(parts) == 4:
        return relative.suffix == ".yaml"
    static_types = {"css": {".css"}, "html": {".html"}, "js": {".js"}, "img": {".svg", ".png", ".ico"}}
    return (len(parts) == 4 and parts[:2] == ("idea", "static")
            and relative.suffix in static_types.get(parts[2], set()))


def stage_context(root: Path, destination: Path, config: dict) -> dict[str, str]:
    """Copy application code, never local data, state, env files or credentials."""
    source_paths = [root / name for name in BUILD_INPUTS]
    # Never follow symlinked directories (including the top-level package).
    if (root / "idea").is_symlink():
        raise ReleaseError("Refusing a symlinked application source directory.")
    for directory, dirs, files in os.walk(root / "idea", followlinks=False):
        for name in dirs:
            if (Path(directory) / name).is_symlink():
                raise ReleaseError("Refusing a symlink in application source: " + str((Path(directory) / name).relative_to(root)))
        for name in files:
            path = Path(directory) / name
            relative = path.relative_to(root)
            if source_allowed(relative) and relative.as_posix() != "idea/static/js/auth.js":
                source_paths.append(path)
    hashes = {}
    for source in sorted(source_paths):
        relative = source.relative_to(root)
        if source.is_symlink() or not source.is_file():
            raise ReleaseError("Build input must be a regular file: " + str(relative))
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o644)  # Runtime image uses a non-root user.
        hashes[relative.as_posix()] = sha256(target)
    for required in ("idea/viewer.py", "idea/worker.py", "idea/static/html/viewer.html"):
        if required not in hashes:
            raise ReleaseError("Required runtime source missing from build context: " + required)
    auth = destination / "idea/static/js/auth.js"
    auth.parent.mkdir(parents=True, exist_ok=True)
    auth.write_text("// Generated from this deployment's Firebase web app.\nconst FIREBASE_CONFIG = "
                    + json.dumps(config, ensure_ascii=True, sort_keys=True) + ";\n", encoding="utf-8")
    auth.chmod(0o644)
    for directory, _, _ in os.walk(destination):
        if Path(directory) != destination:
            Path(directory).chmod(0o755)
    hashes["idea/static/js/auth.js"] = sha256(auth)
    return hashes


def planned_resources(module: dict):
    yield from module.get("resources", [])
    for child in module.get("child_modules", []):
        yield from planned_resources(child)


def container_images(value) -> list[str]:
    images = []
    if isinstance(value, dict):
        for container in value.get("containers", []):
            if isinstance(container, dict) and isinstance(container.get("image"), str):
                images.append(container["image"])
        for key, item in value.items():
            if key != "containers":
                images.extend(container_images(item))
    elif isinstance(value, list):
        for item in value:
            images.extend(container_images(item))
    return images


def has_unknown(value) -> bool:
    if isinstance(value, dict):
        return any(has_unknown(item) for item in value.values())
    if isinstance(value, list):
        return any(has_unknown(item) for item in value)
    return value is True


def only_key_retention_hardening(resource: dict) -> bool:
    """Permit DELETE -> ABANDON metadata, never replacement or changed material."""
    if resource.get("address") != "google_secret_manager_secret_version.encryption_key_version":
        return False
    change = resource.get("change", {})
    before, after = change.get("before"), change.get("after")
    if (change.get("actions") != ["update"] or not isinstance(before, dict)
            or not isinstance(after, dict) or has_unknown(change.get("after_unknown"))):
        return False
    return (before.get("deletion_policy") == "DELETE" and after.get("deletion_policy") == "ABANDON"
            and {key: value for key, value in before.items() if key != "deletion_policy"}
            == {key: value for key, value in after.items() if key != "deletion_policy"})


def inspect_plan(plan: dict, expected_image: str) -> dict:
    """Fail closed on destructive/key changes and mismatched web/worker images."""
    if plan.get("errored") or plan.get("complete") is False:
        raise ReleaseError("Terraform produced an incomplete or errored plan.")
    actions = []
    previous_images = {}
    for resource in plan.get("resource_changes", []):
        address = resource["address"]
        changes = resource.get("change", {})
        operations = changes.get("actions", [])
        if "delete" in operations:
            raise ReleaseError("Routine release refuses deletion or replacement: " + address)
        if address in KEY_RESOURCES and operations != ["no-op"] and not only_key_retention_hardening(resource):
            raise ReleaseError("Routine release refuses encryption-key changes: " + address)
        if operations not in (["no-op"], ["read"]):
            actions.append({"address": address, "actions": operations})
        if address in (WEB, WORKER):
            previous_images[address] = container_images(changes.get("before"))
    resources = {resource["address"]: resource for resource in planned_resources(
        plan.get("planned_values", {}).get("root_module", {})
    )}
    for address in (WEB, WORKER):
        images = container_images(resources.get(address, {}).get("values"))
        if images != [expected_image]:
            raise ReleaseError("Saved plan must deploy the same immutable image to web and worker: " + address)
    return {"changes": actions, "previous_images": previous_images}


def read_plan(root: Path, path: Path) -> dict:
    try:
        result = json.loads(tf(root, "show", "-json", str(path)))
    except json.JSONDecodeError as error:
        raise ReleaseError("Terraform could not read the saved plan.") from error
    if not isinstance(result, dict):
        raise ReleaseError("Terraform plan has an invalid format.")
    return result


def resolve_digest(repository: str, project: str, reference: str) -> str:
    if not shutil.which("gcloud"):
        raise ReleaseError("gcloud is required for image build/verification.")
    raw = command(["gcloud", "artifacts", "docker", "images", "describe", reference,
                   "--project", project, "--format=json"])
    try:
        digest = json.loads(raw)["image_summary"]["digest"]
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ReleaseError("Artifact Registry did not return an immutable image digest.") from error
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise ReleaseError("Artifact Registry returned an invalid image digest.")
    return digest


def plan_release(root: Path, digest: str | None) -> Path:
    if digest and not DIGEST_RE.fullmatch(digest):
        raise ReleaseError("--image-digest must be sha256 followed by 64 lowercase hex characters.")
    release = private_directory(root / ".deploy/releases" / release_id())
    log = release / "operations.log"
    deployment = check_existing_state(root, log)
    repository = deployment["repository_url"]
    project = deployment["project_id"]
    image_name = repository + "/idea-evolution"
    source_hashes = None
    if digest:
        image = image_name + "@" + digest
        resolved = resolve_digest(repository, project, image)
        if digest != resolved:
            raise ReleaseError("Artifact Registry digest does not match the requested image.")
    else:
        if not shutil.which("gcloud"):
            raise ReleaseError("gcloud is required for the build.")
        config = firebase_config(root, project)
        with tempfile.TemporaryDirectory(prefix="idea-release-") as directory:
            source_hashes = stage_context(root, Path(directory), config)
            image_tag = image_name + ":" + release.name
            print(f"Building {image_tag}. This can incur Cloud Build charges; no rollout occurs yet.", flush=True)
            # Stream build progress; context is allowlisted, with no deployment secrets.
            result = subprocess.run(["gcloud", "builds", "submit", "--project", project,
                                     "--tag", image_tag, directory], check=False)
            if result.returncode:
                raise ReleaseError("Cloud Build failed. No Terraform plan was applied.")
        digest = resolve_digest(repository, project, image_tag)
        image = image_name + "@" + digest
    plan_path = release / "release.tfplan"
    tf(root, "plan", "-input=false", "-no-color", "-lock-timeout=60s",
       "-out=" + str(plan_path), "-var=image_digest=" + digest, log=log)
    plan_path.chmod(0o600)
    summary = inspect_plan(read_plan(root, plan_path), image)
    write_private(release / "plan.txt", tf(root, "show", "-no-color", str(plan_path)))
    manifest = {"format_version": 1, **deployment, "image": image, "image_digest": digest,
                "plan_sha256": sha256(plan_path), "source_sha256": source_hashes, **summary}
    write_private(release / "manifest.json", json.dumps(manifest, indent=2) + "\n")
    print("\nPlan ready. No infrastructure changes have been applied.")
    for change in summary["changes"]:
        print("  " + "/".join(change["actions"]) + " " + change["address"])
    print(f"\nReview {release / 'plan.txt'} (sensitive: do not share or commit).")
    print("Review cost, persistence and worker changes as well as the image.")
    print(f"Then approve that exact plan with:\n  ./deploy.sh apply {release.relative_to(root)}")
    return release


def apply_release(root: Path, directory: str) -> None:
    release = Path(directory).expanduser()
    if not release.is_absolute():
        release = root / release
    release = release.resolve()
    if not release.is_relative_to((root / ".deploy/releases").resolve()):
        raise ReleaseError("Apply expects a release directory created under .deploy/releases.")
    plan_path = release / "release.tfplan"
    manifest_path = release / "manifest.json"
    if plan_path.is_symlink() or manifest_path.is_symlink():
        raise ReleaseError("Release files must not be symlinks.")
    if (release / "applied.json").exists():
        raise ReleaseError("This release has already been applied. Create a new plan for another rollout.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseError("Release manifest is missing or invalid. Create a new plan.") from error
    if manifest.get("format_version") != 1 or not plan_path.is_file() or sha256(plan_path) != manifest.get("plan_sha256"):
        raise ReleaseError("Saved plan does not match its release manifest. Create and review a new plan.")
    log = release / "apply.log"
    deployment = check_existing_state(root, log)
    if any(deployment[key] != manifest.get(key) for key in ("project_id", "repository_url")):
        raise ReleaseError("The current Terraform state belongs to a different deployment than this plan.")
    digest = manifest.get("image_digest", "")
    if not isinstance(digest, str) or not DIGEST_RE.fullmatch(digest):
        raise ReleaseError("Release manifest image digest is invalid.")
    image = deployment["repository_url"] + "/idea-evolution@" + digest
    if manifest.get("image") != image:
        raise ReleaseError("Release manifest image does not match its project and digest.")
    inspect_plan(read_plan(root, plan_path), image)
    print(f"Applying the reviewed saved plan for {image}. Diagnostics: {log}", flush=True)
    # A saved plan is already approved by this explicit apply invocation. Never replan.
    tf(root, "apply", "-input=false", "-no-color", "-lock-timeout=60s", str(plan_path), log=log)
    write_private(release / "applied.json", json.dumps({"applied_at": datetime.now(timezone.utc).isoformat()}))
    print("Terraform apply completed. Verify sign-in, saved data, keys and a tiny worker run before inviting users.")
    print("Service: " + str(public_output(root, "service_url")))


def main(argv: list[str] | None = None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action")
    commands.add_parser("check", help="validate existing state/config; no build, plan or apply")
    plan_parser = commands.add_parser("plan", help="build an image and save a plan for review; no rollout")
    plan_parser.add_argument("--image-digest", help="reuse an existing digest, e.g. for a reviewed rollback")
    apply_parser = commands.add_parser("apply", help="apply exactly one previously reviewed release plan")
    apply_parser.add_argument("release_directory")
    args = parser.parse_args(argv)
    if not args.action:
        parser.print_help()
        return 0
    os.umask(0o077)
    try:
        if args.action == "check":
            directory = private_directory(root / ".deploy/checks" / release_id())
            deployment = check_existing_state(root, directory / "check.log")
            print(f"Existing deployment state and configuration verified for {deployment['project_id']}.")
            print("No build or apply performed. Live data, backup health and running revisions still need verification.")
        elif args.action == "plan":
            plan_release(root, args.image_digest)
        else:
            apply_release(root, args.release_directory)
    except (ReleaseError, OSError) as error:
        print("Release stopped: " + str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
