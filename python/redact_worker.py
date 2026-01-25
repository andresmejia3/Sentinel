import sys
import os
import argparse

os.environ["ORT_LOGGING_LEVEL"] = "3"

import cv2
import insightface
import numpy as np
import struct
import time

import logging
logging.getLogger('insightface').setLevel(logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true')
parser.add_argument('--detection-threshold', type=float, default=0.5)
parser.add_argument('--inference-mode', type=str, default='full', choices=['full', 'detection-only'])
parser.add_argument('--raw-width', type=int, default=0, help='Raw image width (if using rawvideo)')
parser.add_argument('--raw-height', type=int, default=0, help='Raw image height (if using rawvideo)')
args, _ = parser.parse_known_args()

DETECTION_THRESHOLD = args.detection_threshold

# Optimization: Load only detection model if we don't need embeddings (blur-all mode)
allowed_modules = ['detection', 'recognition']
if args.inference_mode == 'detection-only':
    allowed_modules = ['detection']

app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=allowed_modules)
app.prepare(ctx_id=0, det_size=(640, 640), det_thresh=DETECTION_THRESHOLD)
app.get(np.zeros((640, 640, 3), dtype=np.uint8))

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
        if args.raw_width > 0 and args.raw_height > 0:
            # Zero-Copy: Interpret raw bytes as RGBA numpy array
            frame_array = np.frombuffer(image_bytes, dtype=np.uint8).reshape((args.raw_height, args.raw_width, 4))
            # InsightFace expects BGR, so we convert RGBA -> BGR
            frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGBA2BGR)
        else:
            frame_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)

        if frame_array is None:
            return b'\x01' + struct.pack('>I', 19) + b"Image decode failed"

        faces = app.get(frame_array)

        valid_faces = []
        h_frame, w_frame = frame_array.shape[:2]
        
        for face in faces:
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, box[0]), max(0, box[1]), min(w_frame, box[2]), min(h_frame, box[3])
            
            # If we are in full mode, check embedding validity
            norm = 1.0
            if args.inference_mode == 'full':
                if face.embedding is None:
                    continue
                norm = np.linalg.norm(face.embedding)
                if norm <= 1e-6:
                    continue
            
            valid_faces.append((face, x1, y1, x2, y2, norm))

        face_payloads = []

        for face, x1, y1, x2, y2, norm in valid_faces:
            # Protocol: [Box 16B] [Vec 2048B]
            # Total 2064 bytes per face.
            
            if args.inference_mode == 'full':
                embedding_bytes = (face.embedding / norm).astype('>f4').tobytes()
            else:
                # Send zero-filled vector for detection-only mode to maintain protocol structure
                embedding_bytes = b'\x00' * 2048

            face_payloads.append(b''.join([
                struct.pack('>4i', x1, y1, x2, y2),
                embedding_bytes
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
        sys.stderr.write(f"🐍 Redact Worker Started. Mode: {args.inference_mode}\n")
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