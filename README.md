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

*   **Hysteresis Redaction:** Implements a "safety net" temporal buffer. If a face is momentarily lost by the tracker, the system maintains the redaction mask to prevent privacy leaks (flicker).
*   **Hybrid Tracking:** Combines heavy Neural Network inference (every $N$ frames) with lightweight Optical Flow tracking (inter-frame) to achieve high FPS.
*   **Interval Debouncing:** Optimizes storage by merging contiguous detections into time intervals (e.g., "Person A: 00:01:05 - 00:01:10") rather than storing per-frame rows.

---

## ⚙️ CLI Configuration

Sentinel exposes a robust CLI interface for tuning performance and privacy parameters.

### Performance & Accuracy
| Flag | Name | Description | Default |
| :--- | :--- | :--- | :--- |
| `-n` | `--nth-frame` | AI keyframe interval (e.g., scan every 10th frame). | `10` |
| `-e` | `--engines` | Number of parallel engine worker processes. | `1` |
| `-t` | `--threshold` | Face matching threshold (lower is stricter). | `0.6` |

### Privacy & Redaction
| Flag | Name | Description | Default |
| :--- | :--- | :--- | :--- |
| `-m` | `--mode` | `blur-all`, `selective`, `targeted`, or `none`. | `none` |
| `-g` | `--grace-period` | The longest period where a face can be missing before Sentinel declares they are out of frame and logs it to the database. | `2.0s` |
| `-l` | `--linger` | How long to keep blurring after a face is lost. | `2.0s` |
| | `--disable-safety-net`| Disable the \"blur all\" safety net (blur everything if we lost the targets face). | `false` |
| `-s` | `--strength`| Strength of the Gaussian blur kernel. | `99` |

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

## 🚦 Getting Started

### Prerequisites
*   Docker & Docker Compose
*   NVIDIA GPU (Optional, defaults to CPU if drivers missing)

### Installation
1.  **Clone the repository:**
    ```bash
    git clone https://github.com/andresmejia3/sentinel.git
    cd sentinel
    ```

2.  **Configure Environment:**
    Create a `.env` file:
    ```bash
    POSTGRES_USER=sentinel
    POSTGRES_PASSWORD=secret
    POSTGRES_DB=sentinel
    POSTGRES_HOST=db
    ```

3.  **Run with Docker:**
    Use the provided wrapper script for easy execution:
    ```bash
    chmod +x sentinel-docker.sh
    ./sentinel-docker.sh --help
    ```

### Usage Examples

**Index a Video:**
```bash
# Note: The current directory is mounted to /data inside the container
./sentinel-docker.sh scan -i /data/security_footage.mp4
```

**Search for a Person:**
```bash
./sentinel-docker.sh find /data/suspect_photo.jpg
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