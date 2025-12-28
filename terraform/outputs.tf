output "service_url" {
  value = google_cloud_run_v2_service.default.uri
}

output "repository_url" {
  value = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.repo.repository_id}"
}

output "firebase_app_id" {
  value = google_firebase_web_app.default.app_id
}

output "project_id" {
  value = var.project_id
}

# We can query the SDK/web config via data source or just output needed IDs
data "google_firebase_web_app_config" "default" {
  provider   = google-beta
  web_app_id = google_firebase_web_app.default.app_id
  project    = var.project_id
}

output "firebase_config_json" {
  value = jsonencode({
    apiKey        = data.google_firebase_web_app_config.default.api_key
    authDomain    = data.google_firebase_web_app_config.default.auth_domain
    projectId     = var.project_id
    storageBucket = data.google_firebase_web_app_config.default.storage_bucket
    messagingSenderId = data.google_firebase_web_app_config.default.messaging_sender_id
    appId         = google_firebase_web_app.default.app_id
  })
}
