import sys
import os
import argparse

# --- Suppress ONNX Runtime Logging (Must be before importing insightface/onnxruntime) ---
os.environ["ORT_LOGGING_LEVEL"] = "3" # 3 = Error only
# ----------------------------------------------------------------------------------------

import cv2
import insightface
import numpy as np
import struct
import uuid
import time

# --- Suppress InsightFace Logging ---
import logging
logging.getLogger('insightface').setLevel(logging.ERROR)

# --- Parse CLI Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--detection-threshold', type=float, default=0.5, help='Face detection confidence threshold')
args, _ = parser.parse_known_args()

DETECTION_THRESHOLD = args.detection_threshold

# --- Global InsightFace App Initialization ---
# This is done once when the worker starts to load the models into memory (and GPU VRAM).
# We only load detection and recognition models to save VRAM (skipping gender/age/landmarks)
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=DETECTION_THRESHOLD)
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

def process_frame(image_bytes, debug=False) -> bytes:
    """
    Decodes an image and returns a BINARY payload of face vectors.

    Protocol Definition (Big Endian):
    -----------------------------------------------------------------------
    [Envelope Header] (4 Bytes) : Total Length of the following payload

    Payload Structure:
    [Status] (1 Byte)           : 0x00 = Success, 0x01 = Error

    If Status == 0x00 (Success):
      [Count] (4 Bytes)         : Number of faces detected (N)
      
      For each face (Repeat N times):
        [Box]     (16 Bytes)    : 4x int32 (x1, y1, x2, y2)
        [Vector]  (2048 Bytes)  : 512x float32 embeddings
        [Quality] (4 Bytes)     : 1x float32 quality score
        [ImgLen]  (4 Bytes)     : Length of the JPEG thumbnail (M)
        [ImgData] (M Bytes)     : Raw JPEG bytes

    If Status == 0x01 (Error):
      [MsgLen]  (4 Bytes)       : Length of error message
      [Message] (N Bytes)       : UTF-8 Error string
    -----------------------------------------------------------------------
    """
    try:
        # Decode directly to BGR using OpenCV (faster, and native to InsightFace)
        # This avoids the expensive Full-Frame RGB->BGR conversion.
        frame_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_array is None:
            return b'\x01' + struct.pack('>I', 19) + b"Image decode failed" # Status 1 = Error

        # InsightFace expects BGR, so we pass it directly
        faces = app.get(frame_array)

        if debug and len(faces) > 0:
            try:
                # Draw directly on the BGR frame using OpenCV
                debug_img = frame_array.copy()
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                
                # Save to /data/debug_frames (mounted volume)
                os.makedirs("/data/debug_frames", exist_ok=True)
                cv2.imwrite(f"/data/debug_frames/debug_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.jpg", debug_img)
            except Exception as e:
                sys.stderr.write(f"Debug save failed: {e}\n")

        # --- PRE-FILTER FACES ---
        # CRITICAL: We must filter faces BEFORE writing the header count.
        # If we filter inside the loop, we send fewer faces than promised, causing Go to hang.
        valid_faces = []
        h_frame, w_frame = frame_array.shape[:2]
        
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w_frame, box[2]), min(h_frame, box[3])
            
            # 1. Filter invalid embeddings (Zero Norm)
            # This prevents Division-By-Zero and DB corruption (pgvector cosine ops fail on zero vectors)
            norm = np.linalg.norm(face.embedding)
            if norm > 1e-6:
                valid_faces.append((face, x1, y1, x2, y2, norm))

        # OPTIMIZATION: Resize massive frames ONCE, not per-face.
        # This prevents resizing a 4K/8K image N times if N faces are found.
        max_dim = 1920
        scale = 1.0
        resized_frame = frame_array
        if max(h_frame, w_frame) > max_dim:
            scale = max_dim / float(max(h_frame, w_frame))
            resized_frame = cv2.resize(frame_array, (0, 0), fx=scale, fy=scale)

        face_payloads = []

        for face, x1, y1, x2, y2, norm in valid_faces:
            # Normalize embedding to unit length for Cosine Similarity
            # We already calculated norm in the filter loop
            # OPTIMIZATION: Convert to Big-Endian Float32 (>f4) and get raw bytes directly
            # This avoids the overhead of struct.pack unpacking 512 arguments
            embedding_bytes = (face.embedding / norm).astype('>f4').tobytes()

            # --- Calculate Face Quality Score (ISO/IEC 29794-5 inspired) ---
            area = (x2 - x1) * (y2 - y1)
            
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

            # Prepare thumbnail (Raw JPEG bytes)
            # We now use the FULL FRAME with a bounding box drawn on it.
            thumb_img = resized_frame.copy()
            
            # Calculate scaled coordinates for drawing
            draw_x1, draw_y1, draw_x2, draw_y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)

            # Draw Green Box (0, 255, 0) with thickness 2
            cv2.rectangle(thumb_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
            
            # Encode directly using OpenCV (Faster, no PIL overhead)
            success, encoded_img = cv2.imencode('.jpg', thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                sys.stderr.write("Warning: Failed to encode thumbnail for a face, skipping.\n")
                continue # Skip this face if encoding fails
            thumb_bytes = encoded_img.tobytes()

            face_payloads.append(b''.join([
                struct.pack('>4i', x1, y1, x2, y2),
                embedding_bytes,
                struct.pack('>f', float(quality)),
                struct.pack('>I', len(thumb_bytes)),
                thumb_bytes
            ]))

        # Now, construct the final response with the correct count
        response = [
            b'\x00', # Status OK
            struct.pack('>I', len(face_payloads)) # Correct number of faces
        ] + face_payloads

        return b''.join(response)

    except Exception as e:
        err_msg = str(e).encode('utf-8')
        return b'\x01' + struct.pack('>I', len(err_msg)) + err_msg

def main():
    """Main loop for the Python Inference Worker."""
    # Stdout is now free for human-readable logs.

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

            response_data = process_frame(image_bytes, debug=args.debug)

            # 6. Strict Output Protocol
            # Write 4-byte BigEndian length
            out_pipe.write(struct.pack('>I', len(response_data)))
            # Write payload
            out_pipe.write(response_data)
            out_pipe.flush()

if __name__ == "__main__":
    main()