# Stage 1: Build the Go Binary
FROM golang:1.25-bookworm AS builder

WORKDIR /app

# Copy dependency definitions
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary. 
# We assume the entry point (package main) is in the root directory.
RUN go build -o /bin/sentinel .

# Stage 2: Runtime Environment (Python + FFmpeg)
FROM python:3.11-slim-bookworm

# Install system dependencies
# - ffmpeg: Required for video processing
# - cmake, build-essential: Required to compile dlib
RUN apt-get update && apt-get install -y \
    ffmpeg \
    cmake \
    build-essential \
    libopenblas-dev \
    liblapack-dev \
    libx11-dev \
    libgtk-3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# Note: dlib compilation can take several minutes
RUN pip install --no-cache-dir numpy && \
    pip install --no-cache-dir dlib face_recognition opencv-python-headless Pillow

# Copy the compiled Go binary from the builder stage
COPY --from=builder /bin/sentinel /usr/local/bin/sentinel

# Copy the Python worker scripts
# The Go app expects "python/worker.py" to exist relative to the working directory
COPY python/ ./python/

ENTRYPOINT ["sentinel"]
CMD ["--help"]