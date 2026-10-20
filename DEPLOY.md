# Deploy and roll back

The release helper updates an **existing** Google Cloud installation. It keeps
the web service and evolution worker on the same immutable image, preserves
stored data and encryption material, and separates planning from applying.

## Prepare

Install Python 3.10+, uv, Terraform and gcloud. Configure gcloud and application
default credentials for the intended project. Obtain the installation's private
`terraform/terraform.tfvars` and `terraform/backend.hcl` from its operator.
Never commit them, state files, credentials or release plans.

On a fresh checkout, connect to the **existing** backend:

```bash
terraform -chdir=terraform init -backend-config=backend.hcl -lockfile=readonly
./deploy.sh check
uv sync --locked
uv run --no-sync pytest -q --strict-markers
```

Keep the committed provider lockfile. Do not add `-migrate-state`, delete state
or bootstrap resources to resolve an ordinary release error. `check` validates
the configured state and required resource identities; it does not verify live
application health.

Before building, confirm the deployment target, a usable recovery point, and
the existing encryption secret version and salt. Check active work and drain
anything incompatible with the release. See [recovery](docs/InfrastructureRecovery.md).

## Plan, review, apply

```bash
./deploy.sh plan
```

This builds an allowlisted source context in Cloud Build, resolves an immutable
image digest, and saves a Terraform plan under `.deploy/releases/`. It excludes
local data and credentials and generates public Firebase browser configuration
inside the staged context. Building can incur cloud charges.

Review the printed directory's `plan.txt` and `manifest.json`:

- The intended project, web service and worker use the same image digest.
- Database, Firebase identity, buckets and encryption material remain intact.
- IAM, resource additions, configuration and ongoing costs are understood.

The helper rejects deletion/replacement and encryption changes, except a
verified metadata-only change from secret-version deletion policy `DELETE` to
`ABANDON`. These guards supplement review; they do not replace it.

Apply the exact reviewed release:

```bash
./deploy.sh apply .deploy/releases/RELEASE_ID
```

This verifies and applies the saved plan without rebuilding or replanning.
Terraform rejects stale plans. **Keep the whole release directory private**:
saved plans may contain secret values even when displayed output is redacted.

## Verify

Successful apply is only the infrastructure step. Check:

1. The intended image is running in both the web service and worker job.
2. The public URL loads; Google sign-in and sign-out work.
3. An existing user's history, templates and settings remain available.
4. A small, explicitly budgeted worker run persists its result after refresh,
   and cancellation/failure states remain visible.

Retain the prior release digest and manifest for rollback. Keep Firebase
authorized domains and any custom domain mapping consistent with the public URL.
The helper does not change those external settings or run paid model checks.

The configured worker has a one-hour execution timeout and no automatic job
retries. A longer or interrupted run may need an explicit resume from its last
durable checkpoint. Updating the web service does not extend a running job's
deadline.

## Rollback and failures

A build or plan failure applies nothing. An apply failure can leave partial
changes: inspect the private logs, actual workload definitions and state, then
prepare a fresh reviewed plan.

To reuse a known compatible image:

```bash
./deploy.sh plan --image-digest sha256:EXACT_64_CHARACTER_DIGEST
# Review the plan, then apply its printed release directory.
```

Use a worker-compatible image and check saved-data compatibility. Updating a
job definition does not replace already-running executions. Moving only web
traffic leaves the worker unchanged and is not a complete rollback.

An image rollback does not undo data writes. Never rotate the encryption key,
delete state or overwrite the only database copy as a repair shortcut.
Use the [recovery procedure](docs/InfrastructureRecovery.md) for data incidents.

## New installations

Bootstrap is a separate reviewed infrastructure operation: establish the
private remote state backend, Firebase project/authentication, Firestore,
encryption secret, image repository and initial application image before
enabling the web service and worker. The helper intentionally refuses an
incomplete existing state.

Python dependencies use `uv.lock`; Terraform uses its provider lockfile.
Base-image tags and package repositories can change between builds. Reusing
the recorded application digest is the supported reproducible release path.
