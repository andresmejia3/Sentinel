import sys
import os
import io
import json
import face_recognition
import numpy as np
import struct
from PIL import Image

# read_exactly reads n bytes from the stream, handling partial reads from OS pipes.
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

        face_locations = face_recognition.face_locations(frame_array)
        face_encodings = face_recognition.face_encodings(frame_array, face_locations)

        results = []
        for loc, enc in zip(face_locations, face_encodings):
            results.append({
                "loc": loc,
                "vec": enc.tolist()
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

if __name__ == "__main__":
    main()