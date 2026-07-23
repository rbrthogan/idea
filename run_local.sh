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

ENABLE_VISUAL_ARTIFACTS=false

show_help() {
    echo "Usage: ./run_local.sh [--visual-artifacts]"
    echo ""
    echo "Options:"
    echo "  --visual-artifacts  Enable the opt-in Gemini visualization-artifact workflow."
    echo "                      Uses IDEA_ARTIFACT_STORE_ROOT when set, otherwise"
    echo "                      stores artifacts under \$PWD/data/artifacts."
    echo "  -h, --help          Show this help message."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --visual-artifacts)
            ENABLE_VISUAL_ARTIFACTS=true
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}" >&2
            show_help >&2
            exit 2
            ;;
    esac
    shift
done

echo -e "${GREEN}=== Idea Evolution Local Development ===${NC}"

if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv is required for local development. Install it from https://docs.astral.sh/uv/getting-started/installation/${NC}"
    exit 1
fi

# Check if using emulator
USE_EMULATOR=true

if ! command -v firebase &> /dev/null; then
    echo -e "${YELLOW}Firebase CLI not found. Install with: npm install -g firebase-tools${NC}"
    echo -e "${YELLOW}Falling back to remote Firestore...${NC}"
    USE_EMULATOR=false
fi

# Set non-secret default environment variables. The encryption key is resolved
# only after we know whether storage is the disposable local emulator.
export ENCRYPTION_SALT="${ENCRYPTION_SALT:-idea-evolution-salt}"
export ADMIN_EMAILS="${ADMIN_EMAILS:-}"

if [ "$ENABLE_VISUAL_ARTIFACTS" = true ]; then
    export IDEA_ENABLE_GEMINI_CODE_ARTIFACTS=true
    export IDEA_ARTIFACT_STORE_ROOT="${IDEA_ARTIFACT_STORE_ROOT:-${PWD}/data/artifacts}"
    echo -e "${GREEN}Visualization artifacts enabled.${NC}"
    echo -e "${GREEN}Artifact store: ${IDEA_ARTIFACT_STORE_ROOT}${NC}"
fi

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
    if [ -z "${ENCRYPTION_KEY:-}" ]; then
        echo -e "${RED}ENCRYPTION_KEY must be set when using remote Firestore.${NC}"
        exit 1
    fi
    if [ "${#ENCRYPTION_KEY}" -lt 32 ]; then
        echo -e "${RED}ENCRYPTION_KEY must contain at least 32 bytes.${NC}"
        exit 1
    fi
    echo -e "${YELLOW}Using remote Firestore.${NC}"
    echo -e "${YELLOW}Make sure GOOGLE_APPLICATION_CREDENTIALS is set to your service account key.${NC}"
else
    export ENCRYPTION_KEY="${ENCRYPTION_KEY:-local-dev-only-encryption-key-32-bytes-minimum}"
fi

echo ""
echo -e "${GREEN}Starting FastAPI development server...${NC}"
echo -e "${GREEN}Application will be available at http://localhost:8000${NC}"
echo ""

# Run the application
IDEA_BIND_HOST="${IDEA_BIND_HOST:-127.0.0.1}"
uv run uvicorn idea.viewer:app --reload --host "$IDEA_BIND_HOST" --port 8000
