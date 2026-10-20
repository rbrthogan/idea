# Recovery

Keep deployment-specific resource IDs, recovery timestamps and release
manifests in private operator records. Verify the live configuration before
using a recovery source. [Ordinary releases](../DEPLOY.md) must preserve the
existing state, Firebase project, database and encryption material.

## What must survive

| Data | Location |
| --- | --- |
| Accounts and user IDs | Firebase Authentication |
| Settings, encrypted provider keys, templates, runs and ratings | Firestore |
| Artifact bytes | Private Cloud Storage, or local artifact storage in development |
| Encryption key and salt | Secret Manager / deployment configuration |
| Infrastructure identities and sensitive values | Terraform remote state |

Firestore backups do not include Firebase Authentication accounts, encryption
secrets or artifact bytes. Keep those independently recoverable. The container
filesystem is temporary. Same-project backups do not protect against losing
the whole project or administrative access.

## Before a release

- Verify deletion protection, PITR and backup configuration on the actual
  database. A schedule does not prove that a backup has completed; PITR only
  covers its reported recovery window.
- Identify a completed, usable backup/export or PITR timestamp. Use a consistent
  snapshot export, or pause writes for an export that can include concurrent
  writes. Verify the export's metadata and data objects.
- Preserve the encryption secret version **and salt**. Existing ciphertext has
  no key ring: changing either requires a separate verified re-encryption
  migration. Retain old key material for recovery.
- Keep remote state private, versioned and access-controlled. Saved Terraform
  plans and state can contain full secrets. The runtime service account should
  not have access to the state or recovery bucket.

## Restore data

1. Pause new runs, auto-rating and other writes; drain relevant workers. Record
   the incident time and preserve the current database and recovery source.
2. Restore a completed backup/export or usable PITR point into a **new named
   database in the same Firebase project**. Preserve Authentication UIDs,
   Firebase configuration, key and salt.
3. Verify user ownership, templates, histories, checkpoints, ratings, artifact
   references and credential decryption without exposing secret values.
   Recovered active leases describe old work: reconcile them as interrupted,
   never automatically replay them into paid jobs.
4. Configure deletion protection, PITR, backup schedule, IAM and any required
   rules/indexes for the new database before cutover. Existing Terraform
   protections for `(default)` do not automatically protect a restored database.
5. With writes paused, set `runtime_firestore_database_id` and deploy the
   configuration to **both** web and worker. This variable only selects the
   target; it does not create or protect the restored database. Ensure old
   revisions and jobs cannot continue writing to the old target.
6. Verify reads, then a small budgeted run before reopening work. Keep the
   original database and recovery source through the rollback window. Account
   for new writes before switching back.

An application-image rollback does not restore data. A database cutover can
strand writes made after it; record and reconcile them deliberately.

## Terraform state recovery

For an ordinary checkout, connect to the existing backend with
`terraform init -backend-config=backend.hcl -lockfile=readonly` from `terraform/`.
A missing local state file is expected with a remote backend.

A backend move is a separate maintenance operation: stop concurrent releases,
privately back up authoritative state, configure a protected destination bucket,
then use `init -migrate-state`. Preserve the provider lockfile and verify lineage,
resource addresses, locking and a fresh plan afterward. `-reconfigure` does not
copy state. Never initialize empty state to repair an existing deployment or
apply a stale state version without reconciliation.

See the official procedures for [Firestore backup/restore](https://docs.cloud.google.com/firestore/native/docs/backups),
[PITR](https://docs.cloud.google.com/firestore/native/docs/pitr),
[export/import](https://docs.cloud.google.com/firestore/native/docs/manage-data/export-import)
and the [Terraform GCS backend](https://developer.hashicorp.com/terraform/language/backend/gcs).
Recovery operations and retained storage can incur charges.
