import sys
import os
import io
import json
import insightface
import numpy as np
import struct
from PIL import Image

# --- Global InsightFace App Initialization ---
# This is done once when the worker starts to load the models into memory (and GPU VRAM).
app = insightface.app.FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
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

def process_frame(image_bytes):
    """
    Decodes an image and returns a JSON string of face vectors.
    Isolated for testing purposes.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        frame_array = np.array(image)

        # Use InsightFace to get all face data in one call
        faces = app.get(frame_array)

        results = []
        for face in faces:
            results.append({
                "loc": face.bbox.astype(int).tolist(), # Bounding box
                "vec": face.embedding.tolist()         # 512-d vector
            })
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"error": str(e)})

def main():
    """Main loop for the Python Inference Worker."""
    # Logging to stderr is SAFE (Go ignores it or logs it separately)
    # printing to stdout will BREAK the program.
    sys.stderr.write("Worker Engine Warm. Awaiting frames...\n")
    sys.stderr.flush()

    # Open File Descriptor 3 (passed from Go) for clean data output
    with os.fdopen(3, 'wb') as out_pipe:
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

            response_data = process_frame(image_bytes)

            # 6. Strict Output Protocol
            resp_bytes = response_data.encode('utf-8')
            # Write 4-byte BigEndian length
            out_pipe.write(struct.pack('>I', len(resp_bytes)))
            # Write payload
            out_pipe.write(resp_bytes)
            out_pipe.flush()
            sys.stderr.flush()

if __name__ == "__main__":
    main()