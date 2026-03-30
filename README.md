# Sentinel: Biometric Video Indexing and Redaction

[![CI Pipeline](https://github.com/andresmejia3/sentinel/actions/workflows/ci.yml/badge.svg)](https://github.com/andresmejia3/sentinel/actions)
[![Go Report Card](https://goreportcard.com/badge/github.com/andresmejia3/sentinel)](https://goreportcard.com/report/github.com/andresmejia3/sentinel)
[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go)](https://go.dev/)
[![Python Version](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Sentinel is a high-performance biometric video indexing and redaction engine built with a hybrid Go + Python architecture. It scans video, extracts face embeddings with InsightFace, stores them in PostgreSQL with `pgvector`, and supports human-reviewed identity indexing, similarity search, and targeted redaction.

Under the hood, Sentinel is designed like a real systems project: Go orchestrates concurrency, worker lifecycles, FFmpeg streaming, and database coordination, while persistent Python workers handle ML inference over a custom binary IPC pipeline. The result is a zero-disk frame-processing workflow built for long-form video, human-in-the-loop accuracy, and production-style data integrity.

## Highlights

- Zero-Disk Video Pipeline: Streams frames from FFmpeg through Go into warm Python workers without writing temporary frame images to disk.
- Hybrid Go + Python Architecture: Uses Go for concurrency, orchestration, cancellation, and database coordination, while Python stays focused on InsightFace inference.
- Warm Worker Pool + Binary IPC: Keeps inference workers alive and communicates over a custom binary protocol so the system avoids per-frame Python startup and model warmup overhead.
- Human-in-the-Loop Identity Review: `scan` stages review YAML by default so ambiguous matches can be reviewed before they affect stored identities or intervals.
- Representative Thumbnail Selection via Embedding Centroid: Sentinel does not just keep the sharpest face crop. For each track, it retains high-quality candidate frames, maintains a running mean embedding, and then selects the candidate whose embedding is closest to that centroid. That makes the chosen thumbnail the most representative view of that track or appearance cluster, not just the loudest frame. Because the selection happens in embedding space instead of using hand-tuned pose-specific rules, the same logic works across different people, looks, poses, lighting conditions, and variants.
- Transaction-Safe Commit / Rollback: Reviewed identity updates are applied with a ledgered vector-delta model so changes can be committed atomically and rolled back cleanly.
- Targeted Redaction with Temporal Safety Net: Supports linger and paranoid modes so temporary detection loss is less likely to cause privacy leaks.

## What Sentinel Actually Does

- Go handles orchestration, worker lifecycle, FFmpeg streaming, concurrency, and database access.
- Python handles face detection and recognition through InsightFace.
- FFmpeg streams frames through pipes instead of writing them to disk first.
- PostgreSQL stores 512-dimensional face embeddings plus appearance intervals.

The worker IPC is a custom binary protocol. Go sends frames over stdin and reads structured responses from a dedicated side pipe, not from normal stdout.

## Default Workflow

Sentinel is review-first by default.

1. Run `sentinel scan`.
2. Sentinel writes a review YAML file and thumbnails.
3. You review or edit the file.
4. Run `sentinel commit <review.yaml>` to apply it to Postgres.

This is the safest workflow because it keeps ambiguous matches out of the database until a human confirms them.

### Important: `--no-staging` Is Unsafe

`sentinel scan --no-staging` skips review and writes identities and intervals directly to Postgres.

That mode is intentionally faster and riskier. In plain terms, it can save the wrong person under a new identity or variant when a scene is ambiguous.

Examples of when that can happen:

- crowded scenes with many similar-looking people
- occlusion or fast motion
- difficult lighting
- scene cuts and unstable detections
- one person getting split into multiple unknown identities

If you care about correctness, use the default staged workflow. Treat `--no-staging` as a convenience mode for quick experiments, not the safe production path.

## Architecture

### Hybrid Design

- Go: orchestration, cancellation, worker pools, DB writes, CLI
- Python: InsightFace inference
- FFmpeg: decode and encode video streams
- PostgreSQL + pgvector: embedding storage and nearest-neighbor search

### Zero-Disk Frame Flow

Sentinel does not dump frames to disk as part of the main processing path. Video flows from FFmpeg into Go, then into warm Python workers over pipes. This keeps the hot path in memory.

### Review and Commit Model

- `scan` in default mode creates a review file and leaves Postgres untouched for identities and intervals
- `commit` applies reviewed actions atomically and records a rollback ledger
- `rollback` reverts a specific commit, with guards to prevent rolling back older video commits after newer ones touched the same video

## Command Reference

Global flag:

- `--db <postgres-connection-string>` overrides `.env` / default DB connection resolution

### `scan`

Scans a video and produces a review file by default.

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | required | Path to video |
| `--engines` | `-e` | `1` | Number of Python workers |
| `--nth-frame` | `-n` | `10` | Process every Nth frame |
| `--threshold` | `-t` | `0.6` | Face matching cosine distance threshold |
| `--detection-threshold` | `-D` | `0.5` | Detection confidence threshold |
| `--grace-period` | `-g` | `2s` | How long a track can disappear before closing |
| `--blip-duration` | `-b` | `100ms` | Minimum saved track length |
| `--buffer-size` | `-B` | `200` | Max in-flight frames in memory |
| `--worker-timeout` |  | `30s` | Per-frame worker timeout |
| `--debug-screenshots` | `-d` | `false` | Save debug frames |
| `--review-file` |  | auto | Custom output path for the review YAML |
| `--no-staging` |  | `false` | Bypass review and write directly to Postgres. Unsafe. |

Behavior:

- default: writes `data/reviews/<video>.review.yaml` and thumbnails
- `--review-file`: same staged behavior, custom review file path
- `--no-staging`: skips review and writes directly to the DB

### `commit`

Applies a reviewed scan file to Postgres.

```bash
sentinel commit data/reviews/video.review.yaml
```

### `rollback`

Rolls back a previously committed batch by commit ID.

```bash
sentinel rollback <commit_id>
```

### `redact`

Redacts faces from a video.

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input` | `-i` | required | Path to input video |
| `--output` | `-o` | `output/redacted.mp4` | Output path |
| `--mode` | `-m` | `blur-all` | `blur-all` or `targeted` |
| `--target` |  |  | Comma-separated identity IDs for targeted mode |
| `--style` |  | `black` | `pixel`, `black`, `gauss`, `secure` |
| `--strength` | `-s` | `15` | Pixel/block strength |
| `--linger` |  | `1s` | Keep redacting briefly after loss |
| `--paranoid` |  | `false` | Blur all faces if a tracked target is lost |
| `--paranoid-strict` |  | `false` | In paranoid mode, trigger even before a target has appeared |
| `--engines` | `-e` | `1` | Number of Python workers |
| `--threshold` | `-t` | `0.6` | Targeted re-ID threshold |
| `--detection-threshold` | `-D` | `0.5` | Detection threshold |
| `--buffer-size` | `-B` | `35` | Max in-flight frames |
| `--worker-timeout` |  | `30s` | Per-frame worker timeout |

### `find`

Searches the database using a reference image.

```bash
sentinel find suspect.jpg
sentinel find suspect.jpg -t 0.5
```

Flags:

- `-t, --threshold`
- `-D, --detection-threshold`
- `-d, --debug`
- `--worker-timeout`

### `label`

Administrative relabeling commands.

Rename an identity:

```bash
sentinel label identity <identity_id> <new_name>
```

Link a variant to an identity and rename the variant:

```bash
sentinel label variant <variant_id> <identity_name> <variant_name>
```

### `delete`

Destructive admin commands for removing identities or variants.

Delete an identity, all of its variants, and linked intervals:

```bash
sentinel delete identity <identity_id>
sentinel delete identity <identity_id> --yes
```

Delete a single variant and the intervals linked to that variant:

```bash
sentinel delete variant <variant_id>
sentinel delete variant <variant_id> --yes
```

Notes:

- `delete identity` cascades through all variants under that identity
- `delete variant` removes only that variant
- if you delete the last variant, Sentinel will ask whether you also want to delete the now-empty identity
- both commands delete linked face intervals through foreign keys
- both commands ask for confirmation unless you pass `--yes`
- `--yes` only skips prompts for the explicit command you ran. It will not auto-delete the parent identity after a last-variant delete.

### `list`

List stored data.

```bash
sentinel list
sentinel list --name Monica
sentinel list variants <identity_id>
sentinel list commits
```

### `reset`

Dangerous admin command. By default it clears everything.

Flags:

- `--db` drops application tables
- `--files` removes generated thumbnails and output videos
- `--debug` removes debug frames

## Delete Behavior

Sentinel now supports targeted deletion.

- `sentinel delete identity <id>` removes that identity, its variants, and linked intervals
- `sentinel delete variant <id>` removes that single variant and its linked intervals
- if the deleted variant was the last one, Sentinel can also delete the now-empty identity after asking you
- these delete commands are destructive admin actions
- `sentinel rollback` is for reviewed commit batches. It is not the undo path for manual deletes.
- `sentinel reset --db` still wipes the entire application schema

## Installation and Usage

### Docker Recommended

`./launch` is the easiest way to run Sentinel locally.

Useful launcher commands:

- `./launch`
- `./launch --build`
- `./launch stop`
- `./launch wipe`
- `./launch clean`
- `./launch prune`

Inside the launched shell:

```bash
# Safe default: create a review file
sentinel scan -i /data/video.mp4

# Apply the reviewed scan to Postgres
sentinel commit /data/reviews/video.review.yaml

# Fast but unsafe direct-write mode
sentinel scan -i /data/video.mp4 --no-staging

# List identities
sentinel list

# Search for someone with an image
sentinel find /data/suspect.jpg

# Targeted redaction
sentinel redact -i /data/video.mp4 -m targeted --target 3 -o /data/output/redacted.mp4
```

### Native Local Installation

Requirements:

- Go 1.25+
- Python 3.11
- FFmpeg in `$PATH`
- PostgreSQL 16 with `pgvector`

Python packages:

```bash
pip install insightface onnxruntime numpy opencv-python-headless
```

Use `onnxruntime-gpu` instead of `onnxruntime` if you want GPU inference and your environment supports it.

Build:

```bash
go build -o sentinel ./cmd/sentinel
```

Run:

```bash
./sentinel scan -i video.mp4
./sentinel commit data/reviews/video.review.yaml
```

Important local DB note:

- the repo `.env` is Docker-oriented by default and commonly uses `POSTGRES_HOST=db`
- for native local runs, use a local Postgres host in `.env` or pass `--db`

Example:

```bash
./sentinel --db "postgres://user:password@localhost:5432/sentinel" list
```

## Data Model Notes

- identities are the top-level people
- variants are appearance clusters under an identity, such as different looks
- intervals point to variants, not directly to identities
- `find` searches nearest variants and then shows intervals for the parent identity

## Current Tradeoffs

- staged review is the safe path
- `--no-staging` is faster but can save wrong identities in ambiguous scenes
- manual delete commands are destructive admin actions and separate from the review/commit workflow
- startup expects a Postgres environment that already works for Sentinel's schema needs

## Development

Useful local sanity checks:

```bash
env GOCACHE=/tmp/sentinel-gocache go test ./...
env GOCACHE=/tmp/sentinel-gocache go build ./cmd/sentinel
```

## Roadmap

- live RTSP ingestion
- audio-aware redaction
- improved clustering and review tooling

### Contact

Andres Mejia  
Systems Engineer | Go & Python
