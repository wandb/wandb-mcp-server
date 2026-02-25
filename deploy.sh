#!/bin/bash

# W&B MCP Server - Google Cloud Run Deployment Script
# 
# This script deploys the wandb-mcp-server to Google Cloud Run with complete
# security configuration including HMAC-SHA256 session management.
#
# DEPLOYMENT CONFIGURATION:
# ========================
# Service:         wandb-mcp-server
# Project:         wandb-mcp-production
# Region:          us-central1
# Service Account: wandb-mcp-server@wandb-mcp-production.iam.gserviceaccount.com
# Port:            8080
#
# Resources:
# - Memory:        2Gi
# - CPU:           2 vCPU
# - Timeout:       300s
# - Concurrency:   1000 requests per instance
# - Auto-scaling:  0-10 instances
#
# Security Features:
# - HMAC-SHA256 API key hashing (enabled)
# - Multi-tenant session isolation (enabled)
# - Google Secret Manager integration (for HMAC keys)
# - Dedicated service account with minimal permissions
# - Client Bearer token authentication
#
# Environment Variables Set:
# - DEPLOY_TIMESTAMP:                        Deployment tracking (auto-generated)
# - MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS:  Enable HMAC sessions (true)
# - MCP_SERVER_SECRETS_PROVIDER:             Secrets provider (gcp)
# - MCP_SERVER_SECRETS_PROJECT:              GCP project for secrets (wandb-mcp-production)
#
# Prerequisites:
# - gcloud CLI installed and authenticated
# - Access to wandb-mcp-production project
# - Service account must exist with Secret Manager access
# - Secret 'mcp-server-secret-hmac-key' must exist in Secret Manager
#
# Usage:
#   ./deploy.sh
#
# The script will:
# 1. Verify all prerequisites
# 2. Prompt for confirmation
# 3. Deploy with complete configuration
# 4. Log deployment to deployments/deployment_history.log
# 5. Verify deployment health

set -e  # Exit on any error

# Configuration
PROJECT_ID="wandb-mcp-production"
SERVICE_NAME="wandb-mcp-server"
REGION="us-central1"
SERVICE_ACCOUNT="wandb-mcp-server@wandb-mcp-production.iam.gserviceaccount.com"
PORT="8080"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if gcloud is installed
check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        print_error "gcloud CLI is not installed. Please install it from: https://cloud.google.com/sdk/docs/install"
        exit 1
    fi
    print_info "gcloud CLI found"
}

# Function to check if user is authenticated
check_auth() {
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
        print_error "Not authenticated with gcloud. Please run: gcloud auth login"
        exit 1
    fi
    local account=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
    print_info "Authenticated as: $account"
}

# Function to set the correct project
set_project() {
    print_info "Setting project to: $PROJECT_ID"
    gcloud config set project "$PROJECT_ID"
}

# Function to get current git branch and commit
get_git_info() {
    local branch=$(git rev-parse --abbrev-ref HEAD)
    local commit=$(git rev-parse --short HEAD)
    local timestamp=$(date +%s)
    
    print_info "Current branch: $branch"
    print_info "Current commit: $commit"
    
    echo "${branch}-${commit}-${timestamp}"
}

# Function to check if service account exists
check_service_account() {
    print_info "Checking if service account exists: $SERVICE_ACCOUNT"
    if gcloud iam service-accounts describe "$SERVICE_ACCOUNT" --project="$PROJECT_ID" &> /dev/null; then
        print_info "Service account verified"
        return 0
    else
        print_error "Service account not found: $SERVICE_ACCOUNT"
        print_error "Please create the service account or update the SERVICE_ACCOUNT variable in this script"
        exit 1
    fi
}

# Function to deploy to Cloud Run
deploy() {
    local tag=$(get_git_info)
    local git_sha=$(git rev-parse HEAD)
    
    print_info "Starting deployment with tag: $tag"
    print_info "Service: $SERVICE_NAME"
    print_info "Region: $REGION"
    print_info "Service Account: $SERVICE_ACCOUNT"
    print_info "Labeling deployment with git_sha: $git_sha"
    
    # Use .venv Python if available for gcloud
    if [ -f ".venv/bin/python" ]; then
        export CLOUDSDK_PYTHON="$(pwd)/.venv/bin/python"
        print_info "Using Python from .venv: $CLOUDSDK_PYTHON"
    fi
    
    # Deploy with all security settings and HMAC configuration
    gcloud run deploy "$SERVICE_NAME" \
        --source . \
        --region "$REGION" \
        --port "$PORT" \
        --platform managed \
        --allow-unauthenticated \
        --service-account "$SERVICE_ACCOUNT" \
        --labels "git_sha=${git_sha}" \
        --update-env-vars "\
DEPLOY_TIMESTAMP=$tag,\
MCP_SERVER_ENABLE_HMAC_SHA256_SESSIONS=true,\
MCP_SERVER_SECRETS_PROVIDER=gcp,\
MCP_SERVER_SECRETS_PROJECT=wandb-mcp-production" \
        --memory 2Gi \
        --cpu 2 \
        --timeout 300 \
        --concurrency 1000 \
        --max-instances 10 \
        --min-instances 0 \
        --project "$PROJECT_ID"
    
    if [ $? -eq 0 ]; then
        print_info "Deployment successful!"
        
        # Get the service URL
        local service_url=$(gcloud run services describe "$SERVICE_NAME" \
            --region "$REGION" \
            --project "$PROJECT_ID" \
            --format="value(status.url)")
        
        print_info "Service URL: $service_url"
        print_info "Health check: $service_url/health"
        
        # Log deployment info
        log_deployment "$tag" "$service_url"
    else
        print_error "Deployment failed!"
        exit 1
    fi
}

# Function to log deployment information
log_deployment() {
    local tag=$1
    local url=$2
    local timestamp=$(date -u +"%Y-%m-%d %H:%M:%S UTC")
    
    # Create deployments directory if it doesn't exist
    mkdir -p deployments
    
    # Create deployment log entry
    cat >> deployments/deployment_history.log << EOF
---
Deployment: $timestamp
Tag: $tag
Service: $SERVICE_NAME
Region: $REGION
Service Account: $SERVICE_ACCOUNT
URL: $url
Deployed by: $(gcloud auth list --filter=status:ACTIVE --format="value(account)")
---

EOF
    
    print_info "Deployment logged to: deployments/deployment_history.log"
}

# Function to verify deployment
verify_deployment() {
    print_info "Verifying deployment..."
    
    local service_url=$(gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --format="value(status.url)")
    
    print_info "Testing health endpoint..."
    
    # Wait a moment for the service to be ready
    sleep 5
    
    # Test health endpoint
    if curl -f -s "${service_url}/health" > /dev/null; then
        print_info "Health check passed!"
    else
        print_warning "Health check failed or endpoint not responding"
        print_warning "This might be normal if the service is still starting up"
    fi
    
    # Show service details
    print_info "Service details:"
    gcloud run services describe "$SERVICE_NAME" \
        --region "$REGION" \
        --project "$PROJECT_ID" \
        --format="table(status.url,status.conditions.type,status.conditions.status)"
}

# Main execution
main() {
    print_info "W&B MCP Server - Cloud Run Deployment"
    print_info "======================================"
    
    # Pre-flight checks
    check_gcloud
    check_auth
    set_project
    check_service_account
    
    # Confirm deployment
    print_warning "This will deploy to production: $PROJECT_ID"
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_info "Deployment cancelled"
        exit 0
    fi
    
    # Deploy
    deploy
    
    # Verify
    verify_deployment
    
    print_info "Deployment complete!"
}

# Run main function
main

