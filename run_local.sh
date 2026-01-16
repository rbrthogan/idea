#!/bin/bash
# run_local.sh - Run the Idea Evolution application locally with Firestore
#
# This script sets up the local development environment with Firebase emulator
# or falls back to using the remote Firestore database.
#
# Prerequisites:
# - Firebase CLI installed: npm install -g firebase-tools
# - Firebase emulator initialized: firebase init emulators
# - Or: GOOGLE_APPLICATION_CREDENTIALS set for remote Firestore access

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Idea Evolution Local Development ===${NC}"

# Check if using emulator
USE_EMULATOR=true

if ! command -v firebase &> /dev/null; then
    echo -e "${YELLOW}Firebase CLI not found. Install with: npm install -g firebase-tools${NC}"
    echo -e "${YELLOW}Falling back to remote Firestore...${NC}"
    USE_EMULATOR=false
fi

# Set default environment variables
export ENCRYPTION_KEY="${ENCRYPTION_KEY:-local-dev-encryption-key}"
export ENCRYPTION_SALT="${ENCRYPTION_SALT:-idea-evolution-salt}"
export ADMIN_EMAILS="${ADMIN_EMAILS:-}"

# If using emulator, start it
if [ "$USE_EMULATOR" = true ]; then
    # Check if firebase.json exists
    if [ ! -f "firebase.json" ]; then
        echo -e "${RED}firebase.json not found. Run 'firebase init emulators' first.${NC}"
        echo -e "${YELLOW}Falling back to remote Firestore...${NC}"
        USE_EMULATOR=false
    else
        echo -e "${GREEN}Starting Firebase emulators...${NC}"

        # Start emulator in background
        firebase emulators:start --only firestore &
        EMULATOR_PID=$!

        # Wait for emulator to start
        sleep 5

        # Set emulator host
        export FIRESTORE_EMULATOR_HOST="localhost:8080"

        echo -e "${GREEN}Firestore emulator running at localhost:8080${NC}"
        echo -e "${GREEN}Emulator UI available at http://localhost:4000${NC}"

        # Trap to kill emulator on exit
        trap "echo 'Stopping emulator...'; kill $EMULATOR_PID 2>/dev/null" EXIT
    fi
fi

if [ "$USE_EMULATOR" = false ]; then
    echo -e "${YELLOW}Using remote Firestore.${NC}"
    echo -e "${YELLOW}Make sure GOOGLE_APPLICATION_CREDENTIALS is set to your service account key.${NC}"
fi

echo ""
echo -e "${GREEN}Starting FastAPI development server...${NC}"
echo -e "${GREEN}Application will be available at http://localhost:8000${NC}"
echo ""

# Run the application
uvicorn idea.viewer:app --reload --host 0.0.0.0 --port 8000
