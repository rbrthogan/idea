# This bucket is managed independently of the application state.
# Recovery and migration procedure: docs/InfrastructureRecovery.md.
terraform {
  backend "gcs" {}
}
