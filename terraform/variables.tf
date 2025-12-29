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

variable "image_tag" {
  description = "The tag of the container image to deploy"
  type        = string
  default     = "latest"
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
