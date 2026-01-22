# Stage 1: Build the Go Binary
FROM golang:1.25-bookworm AS builder

WORKDIR /app

# Disable CGO for faster builds (pure Go binary)
ENV CGO_ENABLED=0

# Copy dependency definitions
COPY go.mod go.sum ./
RUN --mount=type=cache,target=/go/pkg/mod go mod download

# Copy source code
COPY . .

# Build the binary. 
# The entry point is in cmd/sentinel
# Use BuildKit cache to persist build artifacts across Docker runs
RUN --mount=type=cache,target=/root/.cache/go-build \
    --mount=type=cache,target=/go/pkg/mod \
    go build -v -o /bin/sentinel ./cmd/sentinel

# Stage 2: Runtime Environment (Python + FFmpeg)
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

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

# Pre-download the InsightFace models during the build to prevent race conditions at runtime.
# We use CPU provider here as the build environment may not have a GPU.
RUN python3 -c "import insightface; insightface.app.FaceAnalysis(providers=['CPUExecutionProvider'], allowed_modules=['detection', 'recognition']).prepare(ctx_id=0, det_size=(640, 640))"

# Copy the compiled Go binary from the builder stage
COPY --from=builder /bin/sentinel /usr/local/bin/sentinel

# Copy the Python worker scripts
# The Go app expects "python/worker.py" to exist relative to the working directory
COPY python/ ./python/

ENTRYPOINT ["sentinel"]
CMD ["--help"]