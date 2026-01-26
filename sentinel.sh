#!/bin/bash

# Sentinel Docker Wrapper
# Usage: ./sentinel <command> <flags>
# Example: ./sentinel scan -i video.mp4

if ! docker info > /dev/null 2>&1; then
    echo "❌ Error: Docker is not running."
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

# Lifecycle Commands
if [[ "$1" == "stop" ]]; then
    echo "🛑 Gracefully stopping Sentinel services..."
    $DOCKER_COMPOSE down
    exit 0
fi

if [[ "$1" == "wipe" ]]; then
    echo "🧨 Wiping all data (Database & Volumes)..."
    $DOCKER_COMPOSE down -v
    echo "✅ System wiped. Next run will start fresh."
    exit 0
fi

if [[ "$1" == "prune" ]]; then
    echo "🧹 Pruning Docker stopped containers and dangling images..."
    docker system prune -f
    # NOTE: We do NOT run 'docker builder prune' here anymore.
    # That deletes the cached pip/apt layers, forcing slow re-downloads.
    echo "✅ Docker system cleaned (Build cache preserved)."
    exit 0
fi

if [[ "$1" == "clean" ]]; then
    echo "🧹 Removing Sentinel application image..."
    $DOCKER_COMPOSE down --rmi local
    echo "✅ Application image removed. Next run will rebuild code layers."
    exit 0
fi

# Ensure Database is Running
# Checks if the 'db' service is up. If not, starts it.
if [ -z "$($DOCKER_COMPOSE ps -q db 2>/dev/null)" ] || [ "$($DOCKER_COMPOSE ps -q db 2>/dev/null | xargs docker inspect -f '{{.State.Running}}' 2>/dev/null)" != "true" ]; then
    echo "🔄 Starting Database container..."
    $DOCKER_COMPOSE up -d db
    echo "⏳ Waiting for Database to initialize..."
    sleep 2
fi

# Detect GPU capability
# We check if the Docker daemon reports an 'nvidia' runtime.
COMPOSE_FILES="-f docker-compose.yml"
if docker info 2>/dev/null | grep -i "runtimes.*nvidia" > /dev/null; then
    echo "🚀 NVIDIA GPU detected. Enabling GPU acceleration."
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.gpu.yml"
else
    echo "💻 Running in CPU-only mode."
fi

# Check Docker Memory Limit
# Warn if less than 4GB (approx 4 * 1024^3 bytes)
MEM_BYTES=$(docker info --format '{{.MemTotal}}')
if [ "$MEM_BYTES" -lt 4294967296 ]; then
    echo "⚠️  WARNING: Docker has less than 4GB of RAM allocated."
    echo "   Sentinel's AI models are memory intensive."
    echo "   Recommendation: Increase Docker memory to 4GB+ or run with '-e 1'."
fi

# Execute Command
# Helper: Auto-convert local relative paths (./file) to container paths (/data/file)
declare -a docker_args
BUILD_FLAG=""

for arg in "$@"; do
    if [[ "$arg" == "--build" ]]; then
        BUILD_FLAG="true"
        continue
    fi

    if [[ "$arg" == ./* ]]; then
        # Replace leading "./" with "/data/"
        docker_args+=("/data/${arg#./}")
    elif [[ "$arg" == /* ]]; then
        # Handle absolute paths: Check if file is inside current directory
        if [[ "$arg" == "$PWD"* ]]; then
            # Convert absolute path to relative container path
            docker_args+=("/data/${arg#$PWD/}")
        else
            docker_args+=("$arg")
        fi
    elif [[ -f "$arg" ]] && [[ "$arg" != -* ]] && [[ "$arg" == *.* ]] && [[ "$arg" != /* ]]; then
        # Auto-map local files with extensions (e.g. video.mp4 -> /data/video.mp4)
        docker_args+=("/data/$arg")
    else
        docker_args+=("$arg")
    fi
done

if [ "$BUILD_FLAG" == "true" ]; then
    echo "🔨 Rebuilding Docker image..."
    # Pass CACHEBUST to invalidate the COPY layer for Python code, but keep pip/apt cache
    DOCKER_BUILDKIT=1 $DOCKER_COMPOSE $COMPOSE_FILES build --build-arg CACHEBUST=$(date +%s) app || exit 1
fi

$DOCKER_COMPOSE $COMPOSE_FILES run --rm app "${docker_args[@]}"

# Cleanup (Only if we built)
if [ "$BUILD_FLAG" == "true" ]; then
    echo "🧹 Pruning old image layers..."
    docker image prune -f > /dev/null
fi