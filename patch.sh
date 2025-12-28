#!/bin/bash
set -e

# Quick deploy script - rebuilds and pushes the container image without full Terraform
# Use this when you've only changed application code (not infrastructure)

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Quick Deploy (Patch) ===${NC}"

# Get config from terraform
cd terraform

if [ ! -f "terraform.tfvars" ]; then
    echo "Error: terraform.tfvars not found. Run deploy.sh first."
    exit 1
fi

PROJECT_ID=$(grep project_id terraform.tfvars | cut -d'"' -f2)
REGION="us-central1"

# Update auth.js with current Firebase config
echo -e "${BLUE}Updating Firebase config...${NC}"
FIREBASE_CONFIG=$(terraform output -raw firebase_config_json)
cd ..
echo "const FIREBASE_CONFIG = $FIREBASE_CONFIG;" > idea/static/js/auth.js

# Build and push
echo -e "${BLUE}Building and pushing container...${NC}"
REPO_URL="$REGION-docker.pkg.dev/$PROJECT_ID/idea-evolution"
TAG=$(date +%Y%m%d-%H%M%S)

gcloud builds submit --project "$PROJECT_ID" --tag "$REPO_URL/idea-evolution:$TAG" .

# Deploy to Cloud Run
echo -e "${BLUE}Deploying to Cloud Run...${NC}"
cd terraform
terraform apply -var="image_tag=$TAG" -auto-approve

echo -e "${GREEN}Quick deploy complete!${NC}"
terraform output service_url
