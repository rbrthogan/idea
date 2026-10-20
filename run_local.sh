#!/bin/bash
# run_local.sh - Run the Idea Evolution application locally with Firestore
#
# This script sets up the local development environment with Firebase emulator
# or falls back to using the remote Firestore database.
#
# Prerequisites:
# - Firebase CLI installed: npm install -g firebase-tools
# - Java JDK 21+ installed for the Firestore emulator
# - Firebase emulator initialized: firebase init emulators
# - Or: GOOGLE_APPLICATION_CREDENTIALS set for remote Firestore access

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

LOCAL_LAUNCH_ARGS=("$@")

show_help() {
    echo "Usage: ./run_local.sh"
    echo ""
    echo "Options:"
    echo "  -h, --help          Show this help message."
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --visual-artifacts)
            echo "Visualization execution is archived; see archive/prime-visualization/README.md." >&2
            exit 2
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

# Parse dotenv through uv before setting defaults or validating encryption.
# Re-exec preserves arguments and never treats .env contents as shell code.
if [ -f .env ] && [ "${IDEA_LOCAL_ENV_LOADED:-0}" != 1 ]; then
    export IDEA_LOCAL_ENV_LOADED=1
    exec uv run --no-project --no-sync --env-file .env bash "$0" "${LOCAL_LAUNCH_ARGS[@]}"
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

# If using emulator, start it
if [ "$USE_EMULATOR" = true ]; then
    # Check if firebase.json exists
    if [ ! -f "firebase.json" ]; then
        echo -e "${RED}firebase.json not found. Run 'firebase init emulators' first.${NC}"
        echo -e "${YELLOW}Falling back to remote Firestore...${NC}"
        USE_EMULATOR=false
    else
        FIREBASE_EMULATOR_PROJECT_ID="${FIREBASE_EMULATOR_PROJECT_ID:-demo-idea-evolution}"
        export GOOGLE_CLOUD_PROJECT="${GOOGLE_CLOUD_PROJECT:-$FIREBASE_EMULATOR_PROJECT_ID}"
        export GCLOUD_PROJECT="${GCLOUD_PROJECT:-$GOOGLE_CLOUD_PROJECT}"
        export FIRESTORE_EMULATOR_HOST="${FIRESTORE_EMULATOR_HOST:-127.0.0.1:8080}"

        # Local emulator runs use the loopback-only, non-admin auth shim. It
        # issues an ephemeral bearer token to the same-origin browser.
        export IDEA_DEV_AUTH="${IDEA_DEV_AUTH:-1}"

        echo -e "${GREEN}Starting Firebase emulators...${NC}"

        if curl --silent --output /dev/null --max-time 1 "http://${FIRESTORE_EMULATOR_HOST}/"; then
            echo -e "${YELLOW}Reusing existing Firestore emulator at ${FIRESTORE_EMULATOR_HOST}.${NC}"
        else
            # macOS ships a /usr/bin/java launcher even when no Java runtime is
            # installed, so checking command -v alone is insufficient.
            if ! java -version >/dev/null 2>&1; then
                echo -e "${RED}A working Java JDK is required by the Firestore emulator.${NC}" >&2
                echo -e "${YELLOW}Install JDK 21 or later, then confirm that 'java -version' succeeds.${NC}" >&2
                echo -e "${YELLOW}Download: https://adoptium.net/temurin/releases/?version=21${NC}" >&2
                exit 1
            fi

            # Start emulator in background
            LOCAL_EMULATOR_DATA="${IDEA_EMULATOR_DATA_DIR:-${PWD}/data/firestore-emulator}"
            LOCAL_EMULATOR_IMPORT_ARGS=()
            if [ -f "$LOCAL_EMULATOR_DATA/firebase-export-metadata.json" ]; then
                LOCAL_EMULATOR_IMPORT_ARGS=(--import "$LOCAL_EMULATOR_DATA")
            fi
            firebase emulators:start --only firestore --project "$GOOGLE_CLOUD_PROJECT" \
                --export-on-exit="$LOCAL_EMULATOR_DATA" "${LOCAL_EMULATOR_IMPORT_ARGS[@]}" &
            EMULATOR_PID=$!

            cleanup_emulator() {
                if kill -0 "$EMULATOR_PID" 2>/dev/null; then
                    echo "Stopping emulator..."
                    kill "$EMULATOR_PID" 2>/dev/null || true
                    wait "$EMULATOR_PID" 2>/dev/null || true
                fi
            }
            trap cleanup_emulator EXIT INT TERM

            # Fail before starting the app if Firebase exits during startup.
            STARTUP_ATTEMPTS=0
            while [ "$STARTUP_ATTEMPTS" -lt 30 ]; do
                if ! kill -0 "$EMULATOR_PID" 2>/dev/null; then
                    wait "$EMULATOR_PID" 2>/dev/null || true
                    echo -e "${RED}Firestore emulator failed to start; see the Firebase output above.${NC}" >&2
                    exit 1
                fi
                if curl --silent --output /dev/null --max-time 1 "http://${FIRESTORE_EMULATOR_HOST}/"; then
                    break
                fi
                STARTUP_ATTEMPTS=$((STARTUP_ATTEMPTS + 1))
                sleep 1
            done

            if [ "$STARTUP_ATTEMPTS" -ge 30 ]; then
                echo -e "${RED}Timed out waiting for Firestore at ${FIRESTORE_EMULATOR_HOST}.${NC}" >&2
                exit 1
            fi
        fi

        echo -e "${GREEN}Firestore emulator running at ${FIRESTORE_EMULATOR_HOST}${NC}"
        echo -e "${GREEN}Emulator UI available at http://localhost:4000${NC}"
        echo -e "${GREEN}Loopback development authentication enabled (non-admin).${NC}"
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
LOCAL_ENV_FILE_ARGS=()
if [ -f .env ]; then
    LOCAL_ENV_FILE_ARGS=(--env-file .env)
fi
uv run uvicorn idea.viewer:app --reload --host "$IDEA_BIND_HOST" --port 8000 "${LOCAL_ENV_FILE_ARGS[@]}"
