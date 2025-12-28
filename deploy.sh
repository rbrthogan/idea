#!/bin/bash
set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Idea Evolution Deployment ===${NC}"

# Check prerequisites
if ! command -v terraform &> /dev/null; then
    echo "Error: terraform is not installed."
    exit 1
fi

if ! command -v gcloud &> /dev/null; then
    echo "Error: gcloud is not installed."
    exit 1
fi

# Ensure user is logged in
echo "Checking gcloud auth..."
gcloud auth application-default print-access-token > /dev/null 2>&1 || {
    echo "Please login first:"
    echo "gcloud auth application-default login"
    exit 1
}

# Go to terraform directory
cd terraform

# Initialize Terraform
echo -e "${BLUE}Initializing Terraform...${NC}"
terraform init

# Check key configs
if [ ! -f "terraform.tfvars" ]; then
    echo -e "${BLUE}Configuration not found. Let's set it up.${NC}"
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    read -p "Enter Admin Email(s) (comma separated): " ADMIN_EMAILS

    echo "project_id = \"$PROJECT_ID\"" > terraform.tfvars
    echo "admin_emails = \"$ADMIN_EMAILS\"" >> terraform.tfvars
    echo "region = \"us-central1\"" >> terraform.tfvars
else
    # Extract project ID from tfvars
    PROJECT_ID=$(grep project_id terraform.tfvars | cut -d'"' -f2)
fi

REGION="us-central1" # defaulting for simplicity, could parse from info

# 0. Manual Prerequisite Check removed (see DEPLOY.md)

# 1. Provision Infrastructure
# First, enable APIs and wait for propagation to avoid 403 errors
echo -e "${BLUE}Enabling APIs...${NC}"
terraform apply -target=google_project_service.apis -auto-approve

echo -e "${BLUE}Waiting 30 seconds for API propagation...${NC}"
sleep 30

# Ensure ADC has the quota project set (fixes 403 on Firebase resources)
echo -e "${BLUE}Setting ADC Quota Project...${NC}"
gcloud auth application-default set-quota-project $PROJECT_ID

echo -e "${BLUE}Provisioning Remaining Resources...${NC}"
terraform apply -target=google_artifact_registry_repository.repo \
                -target=google_firestore_database.database \
                -target=google_secret_manager_secret_version.encryption_key_version \
                -target=google_firebase_web_app.default \
                -target=data.google_firebase_web_app_config.default \
                -auto-approve

# 2. Configure Firebase in Frontend
echo -e "${BLUE}Configuring Frontend...${NC}"
FIREBASE_CONFIG=$(terraform output -json firebase_config_json)

# Go back to root
cd ..

# Write the config to the JS file
echo "const FIREBASE_CONFIG = $FIREBASE_CONFIG;" > idea/static/js/auth.js
echo "Updated idea/static/js/auth.js with new config."

# 3. Build and Push Image
echo -e "${BLUE}Building and Pushing Container Image...${NC}"
REPO_URL="$REGION-docker.pkg.dev/$PROJECT_ID/idea-evolution"
# Generate a timestamp based tag
TAG=$(date +%Y%m%d-%H%M%S)

echo "Building $REPO_URL/idea-evolution:$TAG..."
gcloud builds submit --project "$PROJECT_ID" --tag "$REPO_URL/idea-evolution:$TAG" .

# 4. Deploy Cloud Run Service
echo -e "${BLUE}Deploying Service...${NC}"
cd terraform

# Pass the new tag to Terraform
terraform apply -var="image_tag=$TAG" -auto-approve

echo -e "${GREEN}Deployment Complete!${NC}"
terraform output service_url
echo -e "${BLUE}IMPORTANT: Ensure you have enabled Google Sign-in in the Firebase Console:${NC}"
echo "https://console.firebase.google.com/project/$PROJECT_ID/authentication/providers"
