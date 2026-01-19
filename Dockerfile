# Stage 1: Build the Go Binary
FROM golang:1.25-bookworm AS builder

WORKDIR /app

# Copy dependency definitions
COPY go.mod go.sum ./
RUN go mod download

# Copy source code
COPY . .

# Build the binary. 
# The entry point is in cmd/sentinel
RUN go build -o /bin/sentinel ./cmd/sentinel

# Stage 2: Runtime Environment (Python + FFmpeg)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Install system dependencies
# - python3.11, pip: For our application runtime
# - ffmpeg: Required for video processing
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
# - onnxruntime-gpu: For GPU-accelerated inference
# - insightface: For state-of-the-art face detection & recognition
RUN pip3 install --no-cache-dir \
    onnxruntime-gpu \
    insightface \
    opencv-python-headless Pillow numpy

# Copy the compiled Go binary from the builder stage
COPY --from=builder /bin/sentinel /usr/local/bin/sentinel

# Copy the Python worker scripts
# The Go app expects "python/worker.py" to exist relative to the working directory
COPY python/ ./python/

ENTRYPOINT ["sentinel"]
CMD ["--help"]