terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.0"
    }
  }
}

locals {
  image_repository = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}/idea-evolution"
  deployment_image = var.image_digest != null ? "${local.image_repository}@${var.image_digest}" : "${local.image_repository}:${var.image_tag}"
  # Existing encrypted provider keys require this exact historical salt.
  encryption_salt = "idea-evolution-salt"
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project               = var.project_id
  region                = var.region
  user_project_override = true
  billing_project       = var.project_id
}

# Enable required APIs
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "firestore.googleapis.com",
    "firebase.googleapis.com",
    "artifactregistry.googleapis.com",
    "secretmanager.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "serviceusage.googleapis.com"
  ])
  service            = each.key
  disable_on_destroy = false
}

# Artifact Registry to store Docker images
resource "google_artifact_registry_repository" "repo" {
  location      = var.region
  repository_id = "idea-evolution"
  description   = "Docker repository for Idea Evolution"
  format        = "DOCKER"
  lifecycle {
    prevent_destroy = true
  }
  depends_on = [google_project_service.apis]
}

# Service Account for Cloud Run
resource "google_service_account" "idea_sa" {
  account_id   = "idea-evolution-sa"
  display_name = "Idea Evolution Service Account"
  depends_on   = [google_project_service.apis]
}

# Firestore Database (Native mode)
resource "google_firestore_database" "database" {
  name                              = "(default)"
  location_id                       = var.region
  type                              = "FIRESTORE_NATIVE"
  concurrency_mode                  = "OPTIMISTIC"
  app_engine_integration_mode       = "DISABLED"
  point_in_time_recovery_enablement = var.firestore_pitr_enabled ? "POINT_IN_TIME_RECOVERY_ENABLED" : "POINT_IN_TIME_RECOVERY_DISABLED"
  delete_protection_state           = "DELETE_PROTECTION_ENABLED"
  deletion_policy                   = "ABANDON"

  # Database location/name changes require a separate migration, never replacement.
  lifecycle {
    prevent_destroy = true
  }

  depends_on = [google_project_service.apis]
}

resource "google_firestore_backup_schedule" "weekly" {
  count     = var.firestore_weekly_backups_enabled ? 1 : 0
  project   = var.project_id
  database  = google_firestore_database.database.name
  retention = "2419200s" # Four weeks; native backups, without another scheduler.
  weekly_recurrence {
    day = "SUNDAY"
  }
}

# Secret for Encryption Key
resource "random_password" "encryption_key" {
  length  = 32
  special = true

  lifecycle {
    prevent_destroy = true
  }
}

resource "google_secret_manager_secret" "encryption_key" {
  secret_id = "idea-encryption-key"
  replication {
    auto {}
  }
  lifecycle {
    prevent_destroy = true
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "encryption_key_version" {
  secret          = google_secret_manager_secret.encryption_key.id
  secret_data     = random_password.encryption_key.result
  deletion_policy = "ABANDON"

  # Existing encrypted provider keys cannot survive a casual key rotation.
  lifecycle {
    prevent_destroy = true
  }
}

# Grant Cloud Run SA access to the secret
resource "google_secret_manager_secret_iam_member" "secret_access" {
  secret_id = google_secret_manager_secret.encryption_key.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.idea_sa.email}"
}

# SMTP Password Secret
resource "google_secret_manager_secret" "smtp_password" {
  secret_id = "idea-smtp-password"
  replication {
    auto {}
  }
  depends_on = [google_project_service.apis]
}

resource "google_secret_manager_secret_version" "smtp_password_version" {
  secret          = google_secret_manager_secret.smtp_password.id
  secret_data     = var.smtp_password
  deletion_policy = "ABANDON" # Old revisions retain their pinned credential.
}

resource "google_secret_manager_secret_iam_member" "smtp_password_access" {
  secret_id = google_secret_manager_secret.smtp_password.id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${google_service_account.idea_sa.email}"
}

# Grant Cloud Run SA access to Firestore
resource "google_project_iam_member" "firestore_access" {
  project = var.project_id
  role    = "roles/datastore.user"
  member  = "serviceAccount:${google_service_account.idea_sa.email}"
}

# Cloud Run Service
resource "google_cloud_run_v2_service" "default" {
  name     = "idea-evolution"
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  # Build before planning; web and worker deploy the same immutable digest.
  template {
    service_account = google_service_account.idea_sa.email
    # Template authoring is synchronous; allow margin beyond Argon's 900s
    # provider deadline and file processing. Evolution runs use the worker job.
    timeout = "1200s"
    containers {
      image = local.deployment_image

      env {
        name  = "IDEA_RUN_JOB"
        value = google_cloud_run_v2_job.evolution_worker.id
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

      resources {
        # Auto-rating still runs in the web process between requests.
        cpu_idle = false
        limits = {
          cpu    = "1000m"
          memory = "512Mi"
        }
      }

      env {
        name  = "ADMIN_EMAILS"
        value = var.admin_emails
      }

      env {
        name  = "ADMIN_EMAIL"
        value = split(",", var.admin_emails)[0]
      }

      env {
        name  = "SMTP_SERVER"
        value = var.smtp_server
      }

      env {
        name  = "SMTP_PORT"
        value = var.smtp_port
      }

      env {
        name  = "SMTP_USERNAME"
        value = var.smtp_username
      }

      env {
        name = "SMTP_PASSWORD"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.smtp_password.secret_id
            version = google_secret_manager_secret_version.smtp_password_version.version
          }
        }
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
    scaling {
      min_instance_count = 0
      max_instance_count = 3
    }
  }

  depends_on = [
    google_project_service.apis,
    google_secret_manager_secret_version.encryption_key_version,
    google_secret_manager_secret_iam_member.secret_access,
    google_secret_manager_secret_version.smtp_password_version,
    google_secret_manager_secret_iam_member.smtp_password_access,
    google_project_iam_member.firestore_access,
    google_cloud_run_v2_job_iam_member.worker_dispatch,
    google_storage_bucket_iam_member.artifact_writer,
  ]

  # Terraform owns the image; only gcloud client metadata is ignored.
  lifecycle {
    ignore_changes = [
      client,
      client_version
    ]
  }
}

# Allow unauthenticated access
resource "google_cloud_run_service_iam_member" "public_access" {
  service  = google_cloud_run_v2_service.default.name
  location = google_cloud_run_v2_service.default.location
  role     = "roles/run.invoker"
  member   = "allUsers"
}


# Firebase Web App
resource "google_firebase_web_app" "default" {
  provider     = google-beta
  project      = var.project_id
  display_name = "Idea Evolution Web"
  lifecycle {
    prevent_destroy = true
  }
  depends_on = [google_project_service.apis]
}

# Enable Authentication (Note: Terraform support for configuring Auth providers is limited/beta,
# we might still need some manual step or gcloud command for Auth provider,
# but we can get the config at least)
