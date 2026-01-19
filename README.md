# Sentinel

**A High-Performance, Privacy-First Biometric Video Indexing & Redaction Engine.**

[![Version](https://img.shields.io/badge/version-0.0.1-blue.svg)](https://github.com/andresmejia3/sentinel)
[![Status](https://img.shields.io/badge/status-Architecture%20Finalized-green.svg)](https://github.com/andresmejia3/sentinel)

Sentinel is an enterprise-grade biometric tool designed to transform unstructured "Dark Data" (raw video) into searchable mathematical vectors. By bridging high-concurrency systems engineering in **Go** with state-of-the-art neural inference in **Python**, Sentinel solves the tension between security monitoring and data privacy compliance (GDPR/CCPA).

---

## 1. Executive Summary

Traditional search engines cannot "see" inside video files. Sentinel addresses this by indexing faces into a **PostgreSQL (pgvector)** database, enabling sub-second identity retrieval across thousands of hours of footage. It features a unique hybrid AI-Tracking pipeline to ensure 100% redaction continuity without sacrificing performance.

---

## 2. System Architecture & Orchestration

Sentinel utilizes a **Centralized Worker-Pool Pattern** to maintain a "warm" environment and maximize throughput.

### A. The Mother Process (Go)
The Go binary serves as the master orchestrator, managing the entire pipeline:
* **Process Supervision:** Spawns and monitors $N$ persistent Python daemons via `os/exec`.
* **Resource Pooling:** Implements a buffered channel-based worker pool to load-balance frames across idle engines.
* **Stream Management:** Orchestrates **FFmpeg** to extract raw frame bytes, piping data directly to workers via STDIN/OUT to achieve **Zero-Disk I/O**.

### B. The Inference Engine (Python)
Stripped-down, high-speed daemons that stay resident in memory:
* **Biometric Extraction:** Uses **ArcFace** (via `InsightFace` and `ONNX Runtime`) to generate 512-dimensional embeddings.
* **Redaction Logic:** Performs frame-level manipulation using **OpenCV** based on signals from the Go orchestrator.
* **Bypassing the GIL:** Running $N$ independent processes allows for true multi-core utilization, bypassing Python's Global Interpreter Lock.

---

## 3. Key Engineering Breakthroughs

### A. Hysteresis Redaction (The Safety Net)
To solve the "Security Gap" where a target might momentarily turn their head, Sentinel implements **Hysteresis Logic**:
* **Mechanism:** If the tracker loses a target, the system enters a "Default Blur" state.
* **Persistence:** The blur remains active for a user-defined period ($G$ seconds) until the next AI keyframe re-establishes identity.

### B. Hybrid Temporal Tracking
* **AI Keyframes:** Heavy neural inference is performed every $n$ frames (user-configurable).
* **Centroid Tracking:** In the gaps between keyframes, a lightweight **OpenCV Tracker** follows the geometry of the face, allowing for 60FPS+ processing speeds.

### C. Interval Debouncing & Storage Optimization
Sentinel does not store a row for every frame. 
* **Logic:** Consecutive detections are merged into **Temporal Intervals** (`00:01:05` -> `00:04:20`).
* **Storage Reduction:** By only persisting the "Entry" and "Exit" vectors for stable appearances, database size is reduced by **>90%**.

---

## 4. Technical Stack & Frameworks

| Component | Technology | Specific Framework/Library |
| :--- | :--- | :--- |
| **Orchestration** | **Go 1.22+** | `Cobra` (CLI), `pgx` (Postgres), `zerolog` |
| **Inference** | **Python 3.11** | `InsightFace`, `ONNX Runtime (GPU)`, `NumPy` |
| **Database** | **PostgreSQL 16**| `pgvector` extension (HNSW Indexing) |
| **Processing** | **FFmpeg** | Headless binary via Unix Pipes |

---

## 5. CLI Configuration


---

## 6. Vector Search Logic

Sentinel uses **Cosine Distance** (`<=>` operator in `pgvector`) to find matches. This measures the angle between vectors, ensuring that lighting changes do not break identity recognition.

### HNSW Indexing
Scanning millions of faces is made instantaneous using a **Hierarchical Navigable Small World (HNSW)** index. It allows the system to navigate high-dimensional space through a multi-layered graph, finding nearest neighbors in milliseconds.

**Database Schema:**
```sql
CREATE TABLE face_intervals (
    id UUID PRIMARY KEY,
    video_id UUID REFERENCES video_metadata(id),
    start_offset FLOAT,
    end_offset FLOAT,
    face_embedding VECTOR(512),
    confidence FLOAT
);

CREATE INDEX ON face_intervals 
USING hnsw (face_embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

## 7. Operational Workflows

Sentinel's power lies in its automated pipelines. Below are the three primary execution paths:

### Workflow A: The "Scan" (Indexing)
1. **Initiation:** User executes `sentinel scan /path/to/media`.
2. **Streaming:** Go spawns **FFmpeg** and the **Python Engine Pool**.
3. **Inference:** Raw frames are load-balanced across engines; Python returns face vectors.
4. **Persistence:** Go debounces detections into logical intervals and saves them to **Postgres** via `pgvector`.

### Workflow B: The "Find" (Retrieval)
1. **Query:** User executes `sentinel find person.jpg`.
2. **Encoding:** The persistent Python engine converts `person.jpg` into a 128-d **Query Vector**.
3. **Search:** Go executes an **HNSW Cosine Similarity** search in the database.
4. **Output:** Sentinel returns a human-readable list of video filenames and precise time-ranges.

### Workflow C: The "Redact" (Compliance)
1. **Targeting:** User executes `sentinel redact video.mp4 --mode selective`.
2. **Authorization:** Go checks every detected face against the "Authorized" database in real-time.
3. **Manipulation:** Go signals Python to blur unauthorized regions using the **Hysteresis Safety Net**.
4. **Encoding:** **FFmpeg** re-encodes the redacted stream into a privacy-compliant output file.

```
---

## 8. Testing Strategy & CI/CD

Sentinel employs a **Mock-Heavy Testing Strategy** to ensure rapid CI feedback without the overhead of compiling heavy AI dependencies (dlib, numpy) in every build.

*   **Scope of Testing:** We explicitly do **not** test "Does the AI recognize a face?"—that is the responsibility of the underlying `InsightFace` library.
*   **Contract Verification:** Instead, tests focus on **Glue Code Verification**: ensuring the Python worker correctly formats AI outputs into the strict JSON schema expected by the Go orchestrator.
*   **Protocol Isolation:** Go unit tests verify the binary IPC protocol (4-byte headers) using `io.Reader` mocks, proving the system handles stream fragmentation and endianness correctly without spawning real processes.

---

## Future Features

### Explainable Search Results
Enhance `sentinel find` with transparency and interpretability:

- Display similarity scores for each match
- Show representative frames for matched identities
- Visualize temporal density of appearances across videos
- Optional heatmap timeline indicating frequency and duration of matches

These features help users understand *why* a match occurred and assess confidence.

---

### Live Stream Indexing
Support real-time ingestion and indexing of live video sources:

- Accept RTSP / webcam / live file streams
- Perform incremental face detection and identity assignment
- Make identities searchable while streams are still processing
- Enable graceful interruption and resumption of long-running streams

---

### Live Stream Ingestion (RTSP)
Move beyond static file analysis to support real-time security feeds such as CCTV.

**Architecture**
- Implement a circular ring buffer in Go to absorb network jitter and dropped frames from RTSP streams
- Decouple frame ingestion from inference to prevent blocking Python workers
- Apply backpressure and frame-dropping strategies under load

**Use Case**
- Continuous monitoring of secure facilities with sub-second alert latency

---

### Multimodal Audio Redaction
Achieve true GDPR-style compliance by sanitizing identity across *all* data modalities, not just video.

**Problem**
- Visual redaction alone is insufficient if the audio track contains PII (e.g., a subject stating their full name)

**Solution**
- Integrate OpenAI Whisper into the Python worker pool to perform faster-than-realtime speech-to-text (STT)

**Workflow**
- Users provide a configurable deny-list of keywords or phrases
- Sentinel aligns transcript timestamps with the original audio track
- Sensitive segments are replaced with a 1000Hz sine wave (beep) or silence during the FFmpeg re-encoding pass

---

### Unsupervised Identity Clustering
Transition from *active search* (finding a specific person) to *passive discovery* (understanding the dataset).

**Algorithm**
- Implement DBSCAN (Density-Based Spatial Clustering of Applications with Noise) directly on the PostgreSQL vector space

**Capabilities**
- **Group Discovery**: Automatically cluster unknown faces into unique identities based on embedding proximity
- **Frequency Analysis**: Identify patterns such as “Unknown Person A appears in 15 videos between 2:00 PM and 4:00 PM”
- **Anomaly Detection**: Flag faces that appear only once or extremely rarely (outliers)

This enables higher-level analytics and insight generation without requiring prior identity labels.
