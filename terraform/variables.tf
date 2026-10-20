variable "project_id" {
  description = "The Google Cloud Project ID"
  type        = string
}

variable "region" {
  description = "The Google Cloud region to deploy to"
  type        = string
  default     = "us-central1"
}

variable "admin_emails" {
  description = "Comma-separated list of admin emails"
  type        = string
}

variable "firebase_auth_domain" {
  description = "Verified Firebase sign-in helper hostname. Null preserves the project default; authorize its OAuth callback before setting."
  type        = string
  default     = null
  validation {
    condition = var.firebase_auth_domain == null ? true : (
      length(var.firebase_auth_domain) <= 253 &&
      can(regex("^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$", var.firebase_auth_domain)) &&
      alltrue([for label in split(".", var.firebase_auth_domain) : length(label) <= 63])
    )
    error_message = "firebase_auth_domain must be null or a lowercase DNS hostname, without a scheme, path, port, wildcard, or whitespace."
  }
}

variable "image_tag" {
  description = "Legacy/bootstrap image tag; ordinary releases should supply image_digest"
  type        = string
  default     = "latest"
}

variable "image_digest" {
  description = "Immutable built image digest, shared by the web service and evolution worker"
  type        = string
  default     = null
  validation {
    condition     = var.image_digest == null ? true : can(regex("^sha256:[0-9a-f]{64}$", var.image_digest))
    error_message = "image_digest must be null or sha256 followed by 64 lowercase hex characters."
  }
}

variable "firestore_pitr_enabled" {
  description = "Retain up to seven days of document versions. Billable storage begins only when applied."
  type        = bool
  default     = true
}

variable "firestore_weekly_backups_enabled" {
  description = "Create a weekly native backup retained for four weeks. Backup storage and restores are billable."
  type        = bool
  default     = true
}

variable "runtime_firestore_database_id" {
  description = "Database used by web and worker. A restored named database can be selected without replacing the protected default database."
  type        = string
  default     = "(default)"
  validation {
    condition     = var.runtime_firestore_database_id == "(default)" || can(regex("^[a-z][a-z0-9-]{2,61}[a-z0-9]$", var.runtime_firestore_database_id))
    error_message = "Use (default) or a valid Firestore database ID (4-63 lowercase letters, numbers, and hyphens, beginning with a letter)."
  }
}

variable "smtp_server" {
  description = "SMTP Server Address"
  type        = string
  default     = "smtp.gmail.com"
}

variable "smtp_port" {
  description = "SMTP Server Port"
  type        = string
  default     = "587"
}

variable "smtp_username" {
  description = "SMTP Username"
  type        = string
}

variable "smtp_password" {
  description = "SMTP Password"
  type        = string
  sensitive   = true
}
