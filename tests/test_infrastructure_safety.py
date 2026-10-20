"""Offline invariants for changes that could strand or destroy production data.

Terraform validate checks the provider schema separately. These tests inspect only
checked-in configuration, never local state, variable values, or cloud resources.
"""

import re
import json
from pathlib import Path
import shutil
import subprocess

import pytest


TF = Path(__file__).resolve().parents[1] / "terraform"


def block(filename, kind, *labels):
    """Read a root-level, terraform-fmt formatted block without evaluating HCL."""
    source = (TF / filename).read_text()
    start = " ".join([kind, *(f'"{label}"' for label in labels)])
    match = re.search(rf"^{re.escape(start)} \{{\n(.*?)^\}}", source, re.M | re.S)
    assert match, f"Missing {start} in {filename}"
    return match.group(1)


@pytest.mark.parametrize(
    "filename,kind,name",
    [
        ("main.tf", "google_firestore_database", "database"),
        ("main.tf", "random_password", "encryption_key"),
        ("main.tf", "google_secret_manager_secret", "encryption_key"),
        ("main.tf", "google_secret_manager_secret_version", "encryption_key_version"),
        ("main.tf", "google_firebase_web_app", "default"),
        ("main.tf", "google_artifact_registry_repository", "repo"),
        ("worker.tf", "google_storage_bucket", "artifacts"),
    ],
)
def test_durable_resources_require_deliberate_destruction(filename, kind, name):
    assert re.search(r"prevent_destroy\s*=\s*true", block(filename, "resource", kind, name))


def test_database_recovery_defaults_do_not_replace_existing_database():
    database = block("main.tf", "resource", "google_firestore_database", "database")
    assert re.search(r'name\s*=\s*"\(default\)"', database)
    assert re.search(r'delete_protection_state\s*=\s*"DELETE_PROTECTION_ENABLED"', database)
    assert re.search(r'deletion_policy\s*=\s*"ABANDON"', database)
    assert 'var.firestore_pitr_enabled ? "POINT_IN_TIME_RECOVERY_ENABLED"' in database
    for variable in ("firestore_pitr_enabled", "firestore_weekly_backups_enabled"):
        assert re.search(r"default\s*=\s*true", block("variables.tf", "variable", variable))
    backup = block("main.tf", "resource", "google_firestore_backup_schedule", "weekly")
    assert "database.database.name" in backup
    assert re.search(r'retention\s*=\s*"2419200s"', backup)
    assert re.search(r'day\s*=\s*"SUNDAY"', backup)


def test_web_and_worker_share_image_database_and_exact_encryption_version():
    workloads = [
        block("main.tf", "resource", "google_cloud_run_v2_service", "default"),
        block("worker.tf", "resource", "google_cloud_run_v2_job", "evolution_worker"),
    ]
    for workload in workloads:
        assert re.search(r"image\s*=\s*local.deployment_image", workload)
        assert re.search(
            r'name\s*=\s*"IDEA_FIRESTORE_DATABASE"\s+value\s*=\s*var.runtime_firestore_database_id',
            workload,
        )
        assert re.search(
            r"version\s*=\s*google_secret_manager_secret_version.encryption_key_version.version",
            workload,
        )
        assert re.search(
            r'name\s*=\s*"ENCRYPTION_SALT"\s+value\s*=\s*local.encryption_salt',
            workload,
        )
        assert not re.search(r'version\s*=\s*"latest"', workload)
    assert re.search(
        r'encryption_salt\s*=\s*"idea-evolution-salt"',
        (TF / "main.tf").read_text(),
    )
    key_version = block("main.tf", "resource", "google_secret_manager_secret_version", "encryption_key_version")
    assert re.search(r"secret_data\s*=\s*random_password.encryption_key.result", key_version)
    assert re.search(r'deletion_policy\s*=\s*"ABANDON"', key_version)


def test_artifacts_are_private_recoverable_and_not_automatically_expired():
    bucket = block("worker.tf", "resource", "google_storage_bucket", "artifacts")
    assert re.search(r'public_access_prevention\s*=\s*"enforced"', bucket)
    assert re.search(r"uniform_bucket_level_access\s*=\s*true", bucket)
    assert re.search(r"force_destroy\s*=\s*false", bucket)
    assert re.search(r"retention_duration_seconds\s*=\s*604800", bucket)
    assert "lifecycle_rule" not in bucket


def test_jobs_do_not_automatically_repeat_provider_work():
    worker = block("worker.tf", "resource", "google_cloud_run_v2_job", "evolution_worker")
    assert re.search(r"max_retries\s*=\s*0", worker)
    assert re.search(r"task_count\s*=\s*1", worker)
    web = block("main.tf", "resource", "google_cloud_run_v2_service", "default")
    assert re.search(r"cpu_idle\s*=\s*false", web), "Web auto-rating still needs background CPU"


@pytest.mark.skipif(shutil.which("terraform") is None, reason="Terraform CLI unavailable")
@pytest.mark.parametrize(
    "domain,valid",
    [
        (None, True),
        ("idea-evolution.web.app", True),
        ("auth.example.com", True),
        ("", False),
        ("https://idea-evolution.web.app", False),
        ("idea-evolution.web.app/__/auth/handler", False),
        ("idea-evolution.web.app:443", False),
        ("*.example.com", False),
        (" idea-evolution.web.app", False),
        ("a" * 64 + ".example.com", False),
    ],
)
def test_auth_domain_contract_in_isolated_provider_free_terraform(tmp_path, domain, valid):
    """Evaluate the actual HCL without touching the deployment backend or providers."""
    write_auth_domain_fixture(tmp_path)
    args = ["terraform", f"-chdir={tmp_path}", "plan", "-refresh=false", "-lock=false", "-input=false", "-no-color"]
    if domain is not None:
        (tmp_path / "terraform.tfvars.json").write_text(json.dumps({"firebase_auth_domain": domain}))
    result = subprocess.run(args, capture_output=True, text=True, timeout=15)
    assert (result.returncode == 0) is valid, result.stderr
    if valid:
        assert json.dumps(domain or "existing-project.firebaseapp.com") in result.stdout
    else:
        assert "firebase_auth_domain must be null or a lowercase DNS hostname" in result.stderr


def write_auth_domain_fixture(tmp_path):
    """An isolated local-only configuration: one variable/output, zero resources."""
    variable = block("variables.tf", "variable", "firebase_auth_domain")
    config = block("outputs.tf", "output", "firebase_config_json")
    selected = re.search(r"authDomain\s*=\s*(.*)", config).group(1)
    selected = selected.replace(
        "data.google_firebase_web_app_config.default.auth_domain",
        '"existing-project.firebaseapp.com"',
    )
    (tmp_path / "main.tf").write_text(
        'variable "firebase_auth_domain" {\n' + variable + '}\n'
        + 'output "selected_domain" {\nvalue = ' + selected + '\n}\n'
    )


@pytest.mark.skipif(shutil.which("terraform") is None, reason="Terraform CLI unavailable")
def test_auth_domain_output_publication_requires_apply_before_build(tmp_path):
    """Local-only saved plans verify the documented ordering; no cloud/provider exists."""
    write_auth_domain_fixture(tmp_path)

    def terraform(*args):
        result = subprocess.run(
            ["terraform", f"-chdir={tmp_path}", *args],
            capture_output=True, text=True, timeout=15,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    terraform("plan", "-input=false", "-out=baseline.tfplan")
    terraform("apply", "-input=false", "baseline.tfplan")
    (tmp_path / "terraform.tfvars.json").write_text(
        json.dumps({"firebase_auth_domain": "idea-evolution.web.app"})
    )
    assert terraform("output", "-raw", "selected_domain") == "existing-project.firebaseapp.com"
    terraform("plan", "-refresh-only", "-input=false", "-out=auth-domain.tfplan")
    plan = json.loads(terraform("show", "-json", "auth-domain.tfplan"))
    assert not plan.get("resource_changes")
    assert plan["output_changes"]["selected_domain"]["after"] == "idea-evolution.web.app"
    terraform("apply", "-input=false", "auth-domain.tfplan")
    assert terraform("output", "-raw", "selected_domain") == "idea-evolution.web.app"
