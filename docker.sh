#!/bin/bash

# Sentinel Docker Environment Setup
# Usage: ./docker.sh [flags]
# Flags:
#   --build    Rebuild the Docker image before starting
#   --clean    Remove the application image before starting
#   --wipe     Wipe all data (Database & Volumes)

if ! docker info > /dev/null 2>&1; then echo "❌ Error: Docker is not running."
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Determine if we should use 'docker-compose' or 'docker compose'
if docker compose version > /dev/null 2>&1; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

if [ ! -f .env ]; then
    echo "❌ Error: .env file not found."
    echo "Please create one with: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST=db"
    exit 1
fi

# Handle Flags
BUILD_FLAG="false"

for arg in "$@"; do
    if [[ "$arg" == "--wipe" ]]; then
        echo "🧨 Wiping all data (Database & Volumes)..."
        $DOCKER_COMPOSE down -v
        echo "✅ System wiped. Next run will start fresh."
        exit 0
    fi

    if [[ "$arg" == "--clean" ]]; then
        echo "🧹 Removing Sentinel application image..."
        $DOCKER_COMPOSE down --rmi local
        echo "✅ Application image removed. Next run will rebuild code layers."
        exit 0
    fi

    if [[ "$arg" == "--build" ]]; then
        BUILD_FLAG="true"
    fi
done

# Ensure Database is Running
# Checks if the 'db' service is up. If not, starts it.
if [ -z "$($DOCKER_COMPOSE ps -q db 2>/dev/null)" ] || [ "$($DOCKER_COMPOSE ps -q db 2>/dev/null | xargs docker inspect -f '{{.State.Running}}' 2>/dev/null)" != "true" ]; then
    echo "🔄 Starting Database container..."
    $DOCKER_COMPOSE up -d db
    echo "⏳ Waiting for Database to initialize..."
    sleep 2
fi

# Detect GPU capability
COMPOSE_FILES="-f docker-compose.yml"
if docker info 2>/dev/null | grep -i "runtimes.*nvidia" > /dev/null; then
    echo "🚀 NVIDIA GPU detected. Enabling GPU acceleration."
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
else
    echo "💻 Running in CPU-only mode."
fi

# Check Docker Memory Limit
MEM_BYTES=$(docker info --format '{{.MemTotal}}')
if [ "$MEM_BYTES" -lt 4294967296 ]; then
    echo "⚠️  WARNING: Docker has less than 4GB of RAM allocated."
    echo "   Sentinel's AI models are memory intensive."
    echo "   Recommendation: Increase Docker memory to 4GB+."
fi

# Rebuild if requested
if [ "$BUILD_FLAG" == "true" ]; then
    echo "🔨 Rebuilding Docker image..."
    DOCKER_BUILDKIT=1 $DOCKER_COMPOSE $COMPOSE_FILES build --build-arg CACHEBUST=$(date +%s) app || exit 1
fi

# Start Interactive Session
echo "🐳 Starting Sentinel Shell..."
echo "   - Current directory mounted to: /data"
echo "   - Run 'sentinel --help' to get started."

# We use --entrypoint bash to override the default 'sentinel' entrypoint
# This drops the user into a shell where they can run the binary manually.
# We pass '-l' to override the default command (["--help"]) defined in docker-compose.yml
$DOCKER_COMPOSE $COMPOSE_FILES run --rm -w /data --entrypoint bash app -l

# Cleanup (Only if we built)
if [ "$BUILD_FLAG" == "true" ]; then
    echo "🧹 Pruning old image layers..."
    docker image prune -f > /dev/null
fi

echo "👋 Session ended."