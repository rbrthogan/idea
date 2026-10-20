# Evolution workers outlive individual web requests. These resources are only
# provisioned by an explicit deployment; local development uses RunService.
resource "google_storage_bucket" "artifacts" {
  name                        = "${var.project_id}-idea-artifacts"
  location                    = var.region
  uniform_bucket_level_access = true
  public_access_prevention    = "enforced"
  force_destroy               = false
  soft_delete_policy {
    retention_duration_seconds = 604800 # Seven-day recovery after object deletion.
  }
  lifecycle {
    prevent_destroy = true
  }
  depends_on = [google_project_service.apis]
}

resource "google_storage_bucket_iam_member" "artifact_writer" {
  bucket = google_storage_bucket.artifacts.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.idea_sa.email}"
}

resource "google_cloud_run_v2_job" "evolution_worker" {
  name     = "idea-evolution-worker"
  location = var.region
  template {
    task_count  = 1
    parallelism = 1
    template {
      service_account = google_service_account.idea_sa.email
      max_retries     = 0
      timeout         = "3600s"
      containers {
        image   = local.deployment_image
        command = ["python", "-m", "idea.worker"]
        resources {
          limits = { cpu = "1000m", memory = "1Gi" }
        }
        env {
          name  = "IDEA_ARTIFACT_BUCKET"
          value = google_storage_bucket.artifacts.name
        }
        env {
          name  = "IDEA_FIRESTORE_DATABASE"
          value = var.runtime_firestore_database_id
        }
        env {
          name  = "ENCRYPTION_SALT"
          value = local.encryption_salt
        }
        env {
          name = "ENCRYPTION_KEY"
          value_source {
            secret_key_ref {
              secret  = google_secret_manager_secret.encryption_key.secret_id
              version = google_secret_manager_secret_version.encryption_key_version.version
            }
          }
        }
      }
    }
  }
  depends_on = [google_project_service.apis, google_project_iam_member.firestore_access,
    google_secret_manager_secret_iam_member.secret_access,
    google_secret_manager_secret_version.encryption_key_version,
  google_storage_bucket_iam_member.artifact_writer]
}

resource "google_project_iam_custom_role" "worker_dispatch" {
  role_id     = "ideaWorkerDispatch"
  title       = "Idea evolution worker dispatch"
  permissions = ["run.jobs.run", "run.jobs.runWithOverrides"]
}

resource "google_cloud_run_v2_job_iam_member" "worker_dispatch" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_job.evolution_worker.name
  role     = google_project_iam_custom_role.worker_dispatch.name
  member   = "serviceAccount:${google_service_account.idea_sa.email}"
}
