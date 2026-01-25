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
parser.add_argument('--quality-strategy', type=str, default='clarity', choices=['clarity', 'portrait', 'confidence', 'legacy'], help='Quality scoring strategy')
args, _ = parser.parse_known_args()

DETECTION_THRESHOLD = args.detection_threshold

# --- Global InsightFace App Initialization ---
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=DETECTION_THRESHOLD)
# Warm up
app.get(np.zeros((640, 640, 3), dtype=np.uint8))
# ---

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
    try:
        frame_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        if frame_array is None:
            return b'\x01' + struct.pack('>I', 19) + b"Image decode failed"

        faces = app.get(frame_array)

        if debug and len(faces) > 0:
            try:
                debug_img = frame_array.copy()
                for face in faces:
                    bbox = face.bbox.astype(int)
                    cv2.rectangle(debug_img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 3)
                
                base_dir = "data"
                if os.path.exists("/data"):
                    base_dir = "/data"
                debug_dir = os.path.join(base_dir, "debug_frames")
                os.makedirs(debug_dir, exist_ok=True)
                cv2.imwrite(f"{debug_dir}/debug_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}.jpg", debug_img)
            except Exception as e:
                sys.stderr.write(f"Debug save failed: {e}\n")

        valid_faces = []
        h_frame, w_frame = frame_array.shape[:2]
        
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w_frame, box[2]), min(h_frame, box[3])
            
            norm = np.linalg.norm(face.embedding)
            if norm > 1e-6:
                valid_faces.append((face, x1, y1, x2, y2, norm))

        max_dim = 1920
        scale = 1.0
        resized_frame = frame_array
        if max(h_frame, w_frame) > max_dim:
            scale = max_dim / float(max(h_frame, w_frame))
            resized_frame = cv2.resize(frame_array, (0, 0), fx=scale, fy=scale)

        face_payloads = []

        for face, x1, y1, x2, y2, norm in valid_faces:
            embedding_bytes = (face.embedding / norm).astype('>f4').tobytes()

            area = (x2 - x1) * (y2 - y1)
            alignment = 0.5
            if face.kps is not None:
                eye_center_x = (face.kps[0][0] + face.kps[1][0]) / 2
                eye_dist = np.linalg.norm(face.kps[1] - face.kps[0])
                if eye_dist > 0:
                    nose_offset = abs(face.kps[2][0] - eye_center_x)
                    alignment = max(0.0, 1.0 - (nose_offset / eye_dist))

            face_roi = frame_array[y1:y2, x1:x2]
            sharpness = 0.0
            if face_roi.size > 0:
                gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
                sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

            quality_strategy = args.quality_strategy
            if quality_strategy == 'portrait':
                quality = face.det_score * (alignment ** 2) * np.log(max(1.0, sharpness)) * np.log(max(1.0, float(area)))
            elif quality_strategy == 'clarity':
                size_score = 1.0 - np.exp(-float(area) / 40000.0)
                quality = face.det_score * alignment * np.log(max(1.0, sharpness)) * size_score
            elif quality_strategy == 'confidence':
                quality = (face.det_score ** 2) * alignment * np.log(max(1.0, sharpness)) * np.log(max(1.0, float(area)))
            else:
                quality = face.det_score * alignment * np.log(max(1.0, sharpness)) * np.sqrt(max(1.0, float(area)))

            if debug:
                sys.stderr.write(f"Face: Score={face.det_score:.2f} Align={alignment:.2f} Sharp={sharpness:.1f} Area={area:.0f} -> Quality ({quality_strategy})={quality:.4f}\n")

            thumb_img = resized_frame.copy()
            draw_x1, draw_y1, draw_x2, draw_y2 = int(x1*scale), int(y1*scale), int(x2*scale), int(y2*scale)
            cv2.rectangle(thumb_img, (draw_x1, draw_y1), (draw_x2, draw_y2), (0, 255, 0), 2)
            
            success, encoded_img = cv2.imencode('.jpg', thumb_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not success:
                sys.stderr.write("Warning: Failed to encode thumbnail for a face, skipping.\n")
                continue
            thumb_bytes = encoded_img.tobytes()

            face_payloads.append(b''.join([
                struct.pack('>4i', x1, y1, x2, y2),
                embedding_bytes,
                struct.pack('>f', float(quality)),
                struct.pack('>I', len(thumb_bytes)),
                thumb_bytes
            ]))

        response = [
            b'\x00', # Status OK
            struct.pack('>I', len(face_payloads))
        ] + face_payloads

        return b''.join(response)

    except Exception as e:
        err_msg = str(e).encode('utf-8')
        return b'\x01' + struct.pack('>I', len(err_msg)) + err_msg

def main():
    with os.fdopen(3, 'wb') as out_pipe:
        sys.stderr.write(f"🐍 Scan Worker Started. Strategy: {args.quality_strategy}\n")
        out_pipe.write(b'READY')
        out_pipe.flush()

        while True:
            header = read_exactly(sys.stdin.buffer, 4)
            if len(header) < 4:
                break
            
            frame_size = struct.unpack('>I', header)[0]
            image_bytes = read_exactly(sys.stdin.buffer, frame_size)
            if len(image_bytes) != frame_size:
                break

            response_data = process_frame(image_bytes, debug=args.debug)

            out_pipe.write(struct.pack('>I', len(response_data)))
            out_pipe.write(response_data)
            out_pipe.flush()

if __name__ == "__main__":
    main()