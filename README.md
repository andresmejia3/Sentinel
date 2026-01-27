# Sentinel: Enterprise Biometric Security & Redaction Engine

[![CI Pipeline](https://github.com/andresmejia3/sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/andresmejia3/sentinel/actions)
[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go)](https://go.dev/)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Sentinel** is a production-grade, hybrid-architecture system designed to bridge the gap between high-performance systems engineering and state-of-the-art computer vision. It transforms unstructured **"Dark Data"** (raw video) into searchable mathematical vectors.

Traditional search engines cannot "see" inside video files. Sentinel addresses this by indexing faces into a **PostgreSQL (pgvector)** database, enabling sub-second identity retrieval across thousands of hours of footage. It features a unique hybrid AI-Tracking pipeline to ensure 100% redaction continuity without sacrificing performance.

---

## 🚀 Engineering Philosophy & Architecture

Sentinel was built to demonstrate **Systems Programming** proficiency, specifically focusing on the interoperability between **Go** (for high-concurrency orchestration) and **Python** (for tensor-heavy inference).

### 1. The Hybrid Architecture (Go + Python)
Instead of a monolithic Python application (which suffers from the GIL) or a pure C++ application (which lacks ML flexibility), Sentinel utilizes a **Centralized Worker-Pool Pattern**:

*   **The Orchestrator (Go):** Manages the lifecycle of the application, handles OS signals, manages database connections pool, and spawns worker processes.
*   **The Inference Engine (Python):** Runs as persistent subprocesses. We avoid the overhead of Python startup time by keeping workers "warm" and communicating via **Standard Streams (Stdin/Stdout)**.
*   **IPC Protocol:** A custom binary protocol (4-byte Big-Endian headers) ensures strict message framing between Go and Python, preventing stream desynchronization.

### 2. Zero-Disk I/O Pipeline
Traditional video processing often writes frames to disk (`.jpg`) before reading them back for inference. Sentinel utilizes **Unix Pipes** to stream raw frame bytes directly from **FFmpeg** $\rightarrow$ **Go** $\rightarrow$ **Python**, keeping data entirely in RAM. This **Zero-Disk I/O** approach maximizes throughput and prevents **disk burnout** (NVMe wear) associated with high-frequency frame writes.

### 3. Vector Search at Scale
Identities are stored as 512-dimensional vectors in **PostgreSQL** using the `pgvector` extension. We utilize **HNSW (Hierarchical Navigable Small World)** indexing to enable sub-millisecond similarity searches across millions of vectors.

---

## 🛠 Tech Stack

The system is built on a "Right Tool for the Job" philosophy. We use **Go 1.25** to leverage the latest scheduler improvements for high-concurrency orchestration, while sticking to **Python 3.11** for the inference engine to ensure maximum compatibility with the stable ML ecosystem.

| Component | Technology | Justification |
| :--- | :--- | :--- |
| **Core Logic** | **Go 1.25** | Goroutines for parallel processing; static typing for reliability. |
| **AI Inference** | **Python 3.11** | Access to `InsightFace` and `ONNX Runtime`. |
| **Acceleration** | **CUDA / ONNX** | GPU-accelerated inference for real-time performance. |
| **Database** | **PostgreSQL 16** | ACID compliance combined with vector search capabilities. |
| **Containerization** | **Docker** | Multi-stage builds for small, secure production images. |
| **CI/CD** | **GitHub Actions** | Automated testing, linting, and build verification. |

---

## ⚡ Key Features

*   **Temporal Redaction (Linger):** Implements a temporal buffer for redaction. If a targeted face is momentarily lost by the tracker, the system continues to redact its last known position to prevent privacy leaks (flicker).
*   **Paranoid Redaction:** An optional safety net that blurs *all* detected faces if a targeted identity is temporarily lost, ensuring maximum privacy at the cost of broader redaction.
*   **Tracking by Detection:** Utilizes high-frequency re-identification (every $N$ frames) combined with temporal smoothing to maintain identity continuity without the drift associated with pure visual trackers.
*   **Interval Debouncing:** Optimizes storage by merging contiguous detections into time intervals (e.g., "Person A: 00:01:05 - 00:01:10") rather than storing per-frame rows.

---

## 📚 Command Reference

Sentinel exposes a robust CLI interface. Below are the available commands and their flags.

### `scan`
Ingests a video file, detects faces, and indexes them into the vector database.

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | (Required) | Path to the video file. |
| `--engines` | `-e` | `1` | Number of parallel AI worker processes. |
| `--nth-frame` | `-n` | `10` | Process every Nth frame (lower = more accuracy, slower). |
| `--threshold` | `-t` | `0.6` | Face matching cosine distance threshold (lower is stricter). |
| `--detection-threshold` | `-D` | `0.5` | Minimum confidence for face detection. |
| `--grace-period` | `-g` | `2s` | Time a face can be missing before the track is closed. |
| `--blip-duration` | `-b` | `100ms` | Minimum duration for a track to be saved (filters noise). |
| `--buffer-size` | `-B` | `200` | Max frames to buffer in memory. |
| `--quality-strategy` | | `clarity` | Strategy for face quality (`clarity`, `portrait`, `confidence`). |
| `--worker-timeout` | | `30s` | Timeout for AI worker processing per frame. |
| `--debug-screenshots` | `-d` | `false` | Save frames with bounding boxes to `/data/debug_frames/`. |

### `redact`
Redacts faces in a video based on detection or specific identities.

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | (Required) | Path to input video. |
| `--output` | `-o` | `output/redacted.mp4` | Path to output video. |
| `--mode` | `-m` | `blur-all` | Redaction mode: `blur-all` or `targeted`. |
| `--target` | | | Comma-separated list of Identity IDs (required for `targeted`). |
| `--style` | | `black` | Redaction style: `pixel`, `black`, `gauss`, `secure`. |
| `--strength` | `-s` | `15` | Intensity of the blur/pixelation. |
| `--linger` | | `1s` | Continue redacting area for X time after face is lost. |
| `--paranoid` | | `false` | If a target is lost, switch to blurring ALL faces temporarily. |
| `--engines` | `-e` | `1` | Number of parallel AI worker processes. |
| `--threshold` | `-t` | `0.6` | Face matching threshold (for targeted mode). |
| `--detection-threshold` | `-D` | `0.5` | Minimum confidence for face detection. |
| `--buffer-size` | `-B` | `35` | Max frames to buffer (lower than scan to prevent OOM). |
| `--worker-timeout` | | `30s` | Timeout for AI worker processing per frame. |

### `find`
Search for a person in the database using a reference image.

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--threshold` | `-t` | `0.6` | Face matching cosine distance threshold. |
| `--detection-threshold` | `-D` | `0.5` | Minimum confidence for face detection. |
| `--debug` | `-d` | `false` | Save debug screenshots. |

```bash
sentinel find -t 0.5 /data/suspect.jpg
```

### `label`
Assign a name to a discovered identity ID.
```bash
sentinel label 12 "Jane Doe"
```

### `list`
List all stored identities and their metadata.
```bash
sentinel list
```

### `reset`
Wipe system data. By default, clears everything.

| Flag | Description |
| :--- | :--- |
| `--db` | Drop all database tables. |
| `--files` | Delete generated thumbnails and output videos. |
| `--debug` | Delete debug frames. |

---

## 🏗 Production Readiness

This repository demonstrates a "Day 1 Ready" codebase structure:

### 1. CI/CD Pipeline
A robust GitHub Actions workflow (`.github/workflows/ci.yml`) enforces quality gates on every push:
*   **Static Analysis:** Runs `go vet` and `flake8` to catch logical and stylistic errors.
*   **Unit Testing:** Runs Go and Python test suites in parallel.
*   **Build Verification:** Verifies that the Docker image builds successfully with all dependencies.

### 2. Testing Strategy
We employ a **Mock-Heavy** testing approach to ensure CI speed and reliability:
*   **Protocol Isolation:** Go tests verify the binary IPC protocol using `io.Reader` mocks, ensuring data integrity without spawning real processes.
*   **Dependency Injection:** Python tests mock `sys.modules` to simulate heavy libraries (`insightface`, `numpy`) allowing tests to run in milliseconds without GPU requirements.

### 3. Docker & Deployment
*   **Multi-Stage Builds:** The `Dockerfile` separates the build environment (Go compiler) from the runtime environment (Python + FFmpeg), resulting in a minimal final image.
*   **Infrastructure as Code:** `docker-compose.yml` defines the entire stack (App + DB + GPU config) for one-command deployment.
*   **Security:** Secrets are managed via `.env` files and never committed to version control.

---

## 🚦 Installation & Usage

Sentinel can be run in a containerized environment (recommended) or compiled locally.

### Option A: Docker (Recommended)
No dependencies required other than Docker Desktop.

1.  **Start the Environment:**
    ```bash
    ./docker.sh
    ```
    This will start the database, build the image, and drop you into a **Sentinel Shell**.

2.  **Run Commands:**
    Inside the shell, the `sentinel` binary is already added to your `$PATH`.
    ```bash
    # Scan a video
    sentinel scan -i /data/video.mp4

    # List identified people
    sentinel list
    ```

### Option B: Local Installation
If you prefer to run Sentinel natively, you must install the following dependencies:

1.  **System Requirements:**
    *   **Go 1.25+**
    *   **Python 3.11**
    *   **FFmpeg** (Must be in `$PATH`)
    *   **PostgreSQL 16** with `pgvector` extension enabled.

2.  **Python Dependencies:**
    ```bash
    pip install insightface onnxruntime numpy opencv-python-headless
    ```
    *(Note: Use `onnxruntime-gpu` if you have an NVIDIA GPU)*

3.  **Compile:**
    ```bash
    go build -o sentinel
    ```

4.  **Run:**
    Ensure your `.env` file connects to your local Postgres instance.
    ```bash
    ./sentinel scan -i video.mp4
    ```

---

## 🔮 Future Roadmap

*   **Live RTSP Ingestion:** Ring-buffer implementation for processing live IP camera feeds.
*   **Multimodal Audio Redaction:** Visual redaction is insufficient if a subject speaks their name. We plan to integrate **OpenAI Whisper** to generate timestamped transcripts, aligning them with the video track to silence sensitive keywords (PII) via FFmpeg re-encoding.
*   **Unsupervised Clustering:** Using DBSCAN on vector embeddings to discover unique identities without prior knowledge.

---

### Contact

**Andres Mejia**  
*Systems Engineer | Go & Python Specialist*
GitHub Profile