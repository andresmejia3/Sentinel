import sys
import os

# --- Suppress ONNX Runtime Logging (Must be before importing insightface/onnxruntime) ---
os.environ["ORT_LOGGING_LEVEL"] = "3" # 3 = Error only
# ----------------------------------------------------------------------------------------

import io
import json
import cv2
import insightface
import numpy as np
import struct
import uuid
import time
import base64
from PIL import Image, ImageDraw

# --- Suppress InsightFace Logging ---
import logging
logging.getLogger('insightface').setLevel(logging.ERROR)

# --- Global InsightFace App Initialization ---
# This is done once when the worker starts to load the models into memory (and GPU VRAM).
# We only load detection and recognition models to save VRAM (skipping gender/age/landmarks)
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640))
# Warm up the engine with a dummy inference to force CUDA initialization
# This prevents the "first frame lag" and ensures workers are truly ready.
app.get(np.zeros((640, 640, 3), dtype=np.uint8))
# ---

# read_exactly reads n bytes from the stream, handling partial reads from OS pipes
def read_exactly(stream, n):
    chunks = []
    bytes_read = 0
    while bytes_read < n:
        chunk = stream.read(n - bytes_read)
        if not chunk:
            break
        chunks.append(chunk)
        bytes_read += len(chunk)
    return b''.join(chunks)

def process_frame(image_bytes, debug=False):
    """
    Decodes an image and returns a JSON string of face vectors.
    Isolated for testing purposes.
    """
    try:
        # Decode directly to BGR using OpenCV (faster, and native to InsightFace)
        # This avoids the expensive Full-Frame RGB->BGR conversion.
        frame_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_array is None:
            return json.dumps({"error": "Image decode failed"})

        # InsightFace expects BGR, so we pass it directly
        faces = app.get(frame_array)

        if debug and len(faces) > 0:
            try:
                # Convert BGR to RGB for PIL debug drawing
                debug_img = Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(debug_img)
                for face in faces:
                    bbox = face.bbox.astype(int)
                    draw.rectangle(bbox.tolist(), outline="red", width=3)
                
                # Save to /data/debug_frames (mounted volume)
                os.makedirs("/data/debug_frames", exist_ok=True)
                debug_img.save(f"/data/debug_frames/debug_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.jpg")
            except Exception as e:
                sys.stderr.write(f"Debug save failed: {e}\n")

        results = []
        for face in faces:
            # Normalize embedding to unit length for Cosine Similarity
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm

            # --- Calculate Face Quality Score (ISO/IEC 29794-5 inspired) ---
            box = face.bbox.astype(int)
            h, w = frame_array.shape[:2]
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])
            area = (x2 - x1) * (y2 - y1)
            
            # Filter small faces (noise/ghosts)
            # If a face is smaller than 40x40 pixels, it's likely a false positive or too blurry to be useful.
            # if (x2 - x1) < 40 or (y2 - y1) < 40:
            #     continue

            # 1. Alignment Score (0.0 - 1.0)
            # Penalize faces that are looking away (side profile)
            alignment = 0.5 # Default fallback
            if face.kps is not None:
                # Calculate horizontal offset of the nose (idx 2) relative to the eyes (idx 0, 1)
                eye_center_x = (face.kps[0][0] + face.kps[1][0]) / 2
                eye_dist = np.linalg.norm(face.kps[1] - face.kps[0])
                if eye_dist > 0:
                    nose_offset = abs(face.kps[2][0] - eye_center_x)
                    # Ideally nose is in the center. We penalize deviation relative to eye distance.
                    alignment = max(0.0, 1.0 - (nose_offset / eye_dist))

            # 2. Sharpness Score (Laplacian Variance)
            # Standard metric for blur detection. Higher is sharper.
            face_roi = frame_array[y1:y2, x1:x2]
            sharpness = 0.0
            if face_roi.size > 0:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            # 3. Composite Quality Score
            # We use log(sharpness) to dampen the effect of extremely textured backgrounds
            # We use sqrt(area) to prevent massive blurry faces from winning just by size
            quality = face.det_score * alignment * np.log(max(1.0, sharpness)) * np.sqrt(max(1.0, float(area)))

            # Encode thumbnail to Base64 (In-Memory)
            # We now use the FULL FRAME with a bounding box drawn on it.
            thumb_img = frame_array.copy()
            # Draw Green Box (0, 255, 0) with thickness 2
            cv2.rectangle(thumb_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Convert BGR to RGB for PIL/JPEG encoding
            thumb_rgb = cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)
            face_img = Image.fromarray(thumb_rgb)
            mem_file = io.BytesIO()
            face_img.save(mem_file, format="JPEG", quality=85)
            thumb_b64 = base64.b64encode(mem_file.getvalue()).decode('utf-8')

            results.append({
                "loc": face.bbox.astype(int).tolist(), # Bounding box
                "vec": embedding.tolist(),        # 512-d vector
                "thumb_b64": thumb_b64,
                "quality": float(quality)
            })
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    """Main loop for the Python Inference Worker."""
    # Stdout is now free for human-readable logs.

    debug_mode = "--debug" in sys.argv

    # Open File Descriptor 3 (passed from Go) for clean data output
    with os.fdopen(3, 'wb') as out_pipe:
        # Handshake: Signal readiness to Go via the data pipe.
        out_pipe.write(b'READY')
        out_pipe.flush()

        while True:
            # 1. Read 4-byte header
            header = read_exactly(sys.stdin.buffer, 4)
            if len(header) < 4:
                break
            
            frame_size = struct.unpack('>I', header)[0]
            
            # 2. Read image bytes
            image_bytes = read_exactly(sys.stdin.buffer, frame_size)
            if len(image_bytes) != frame_size:
                break

            response_data = process_frame(image_bytes, debug=debug_mode)

            # 6. Strict Output Protocol
            resp_bytes = response_data.encode('utf-8')
            # Write 4-byte BigEndian length
            out_pipe.write(struct.pack('>I', len(resp_bytes)))
            # Write payload
            out_pipe.write(resp_bytes)
            out_pipe.flush()

if __name__ == "__main__":
    main()