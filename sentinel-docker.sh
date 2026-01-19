#!/bin/bash

# Sentinel Docker Wrapper
# Usage: ./sentinel-docker.sh <command> <flags>
# Example: ./sentinel-docker.sh scan -i /data/video.mp4

# 1. Verify Environment
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found."
    echo "Please create one with: POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST=db"
    exit 1
fi

# 2. Ensure Database is Running
# Checks if the 'db' service is up. If not, starts it.
if [ -z "$(docker-compose ps -q db 2>/dev/null)" ] || [ "$(docker-compose ps -q db 2>/dev/null | xargs docker inspect -f '{{.State.Running}}' 2>/dev/null)" != "true" ]; then
    echo "🔄 Starting Database container..."
    docker-compose up -d db
    echo "⏳ Waiting for Database to initialize..."
    sleep 2
fi

# 3. Execute Command
# We use "$@" to pass all arguments (e.g., "scan -i ...") to the container.
# Note: The current directory is mounted to /data inside the container.
# You must reference files as /data/filename.mp4
echo "🐳 Running Sentinel..."
docker-compose run --rm app "$@"