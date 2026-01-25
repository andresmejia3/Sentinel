import sys
import os
import unittest
import struct
import importlib
from unittest.mock import MagicMock, patch

class TestWorkerLogic(unittest.TestCase):
    
    def setUp(self):
        # Ensure the directory containing this test file is in sys.path
        # This allows 'import worker' to work regardless of CWD
        test_dir = os.path.dirname(os.path.abspath(__file__))
        if test_dir not in sys.path:
            sys.path.insert(0, test_dir)

        # 1. Create fresh mocks for every single test run
        self.mock_insightface = MagicMock()
        self.mock_np = MagicMock()
        self.mock_cv2 = MagicMock()
        
        # 2. Patch sys.modules so 'import worker' sees our mocks instead of real libraries
        self.modules_patcher = patch.dict(sys.modules, {
            "insightface": self.mock_insightface,
            "numpy": self.mock_np,
            "cv2": self.mock_cv2,
        })
        self.modules_patcher.start()

        # 3. Import the worker module
        # We need to mock the app initialization that happens at module level
        self.mock_app_instance = MagicMock()
        self.mock_insightface.app.FaceAnalysis.return_value = self.mock_app_instance

        # FIX: Import 'scan_worker' instead of the unused 'worker.py'
        self.worker = importlib.import_module('scan_worker')

    def tearDown(self):
        # Stop patching to clean up
        self.modules_patcher.stop()
        # Remove worker from sys.modules so it doesn't pollute other tests
        if 'scan_worker' in sys.modules:
            del sys.modules['scan_worker']

    def test_process_frame_success(self):
        """Test that valid image data returns correct JSON structure."""
        
        # 1. Setup Mock AI responses
        # cv2.imdecode returns a mock array
        # FIX: The frame comes from imdecode, so IT needs the shape, not np.array
        mock_frame = MagicMock()
        mock_frame.shape = (100, 100, 3) # Height, Width, Channels

        # FIX: Mock the ROI slice to have a numeric size for sharpness check (face_roi.size > 0)
        mock_roi = MagicMock()
        mock_roi.size = 100
        mock_frame.__getitem__.return_value = mock_roi

        self.mock_cv2.imdecode.return_value = mock_frame

        # Simulate InsightFace returning one face object
        mock_face = MagicMock()
        mock_face.det_score = 0.99 # Must be a float for quality calc
        # FIX: Set kps to None to skip complex alignment math for this test
        mock_face.kps = None

        # Mock bbox: Needs to support indexing (box[0]) AND .tolist()
        # box = [10, 20, 30, 40]
        mock_box = MagicMock()
        coords = [10, 20, 30, 40]
        mock_box.__getitem__.side_effect = lambda i: coords[i]
        mock_box.tolist.return_value = coords
        mock_face.bbox.astype.return_value = mock_box

        # Mock embedding: Needs to support division and .tolist()
        fake_vec = [0.1] * 512
        mock_embedding = MagicMock()
        mock_face.embedding = mock_embedding
        
        # Mock numpy linalg norm
        self.mock_np.linalg.norm.return_value = 1.0
        
        # FIX: Mock math functions used in quality score calculation
        self.mock_np.log.return_value = 1.0
        self.mock_np.sqrt.return_value = 1.0
        self.mock_np.exp.return_value = 0.5 # Mock for 'clarity' strategy
        self.mock_cv2.Laplacian.return_value.var.return_value = 100.0
        
        # Mock the result of division: (embedding / norm)
        mock_div_result = MagicMock()
        mock_embedding.__truediv__.return_value = mock_div_result
        
        # Mock the result of astype: (embedding / norm).astype('>f4')
        mock_astype_result = MagicMock()
        mock_div_result.astype.return_value = mock_astype_result
        mock_astype_result.tobytes.return_value = struct.pack('>512f', *fake_vec)

        self.mock_app_instance.get.return_value = [mock_face]

        # Mock cv2.imencode for thumbnail generation
        mock_encoded = MagicMock()
        mock_encoded.tobytes.return_value = b'fake_jpeg_bytes'
        self.mock_cv2.imencode.return_value = (True, mock_encoded)
        
        # 2. Run the function under test
        result_bytes = self.worker.process_frame(b"fake_image_bytes")
        
        # 3. Assertions (Binary Protocol)
        # Structure: [Status:1] [Count:4] [Box:16] [Vec:2048] [Qual:4] [ImgLen:4] [ImgData:N]
        
        # Check Status (0 = Success)
        self.assertEqual(result_bytes[0], 0x00)
        
        # Check Count (4 bytes, Big Endian)
        count = struct.unpack('>I', result_bytes[1:5])[0]
        self.assertEqual(count, 1)
        
        # Check Box (4 ints * 4 bytes = 16 bytes)
        offset = 5
        box = struct.unpack('>4i', result_bytes[offset:offset+16])
        self.assertEqual(list(box), [10, 20, 30, 40])
        
        # Check Vector (First float of 512)
        offset += 16
        vec_start = struct.unpack('>f', result_bytes[offset:offset+4])[0]
        self.assertAlmostEqual(vec_start, 0.1, places=5)

    def test_process_frame_error_handling(self):
        """Test that invalid data returns a Binary error packet."""
        # Force cv2.imdecode to raise an exception (or return None, but here we test exception catch)
        self.mock_cv2.imdecode.side_effect = Exception("Corrupt Data")

        result_bytes = self.worker.process_frame(b"garbage")
        
        # Structure: [Status:1] [MsgLen:4] [Msg:N]
        self.assertEqual(result_bytes[0], 0x01) # Status 1 = Error
        
        msg_len = struct.unpack('>I', result_bytes[1:5])[0]
        msg = result_bytes[5:5+msg_len].decode('utf-8')
        
        self.assertEqual(msg, "Corrupt Data")

    def test_process_frame_imdecode_returns_none(self):
        """Test that a None from imdecode returns the correct binary error."""
        self.mock_cv2.imdecode.return_value = None

        result_bytes = self.worker.process_frame(b"garbage")

        # Structure: [Status:1] [MsgLen:4] [Msg:N]
        self.assertEqual(result_bytes[0], 0x01) # Status 1 = Error

        msg_len = struct.unpack('>I', result_bytes[1:5])[0]
        msg = result_bytes[5:5+msg_len].decode('utf-8')

        self.assertEqual(msg_len, 19)
        self.assertEqual(msg, "Image decode failed")

    def test_process_frame_imencode_failure(self):
        """Test that a face is skipped if thumbnail encoding fails, and the header is correct."""
        # Setup mocks similar to success test, but for two faces
        mock_frame = MagicMock()
        mock_frame.shape = (100, 100, 3)

        # FIX: Mock the ROI slice to have a numeric size
        mock_roi = MagicMock()
        mock_roi.size = 100
        mock_frame.__getitem__.return_value = mock_roi
        self.mock_cv2.imdecode.return_value = mock_frame

        # Create two mock faces
        face1 = MagicMock(det_score=0.99, kps=None, bbox=MagicMock(), embedding=MagicMock())
        face1.bbox.astype.return_value.__getitem__.side_effect = lambda i: [10,10,20,20][i]
        
        # Mock embedding division and astype for face1
        mock_div_result1 = MagicMock()
        face1.embedding.__truediv__.return_value = mock_div_result1
        mock_astype_result1 = MagicMock()
        mock_div_result1.astype.return_value = mock_astype_result1
        mock_astype_result1.tobytes.return_value = struct.pack('>512f', *([0.1]*512))

        face2 = MagicMock(det_score=0.98, kps=None, bbox=MagicMock(), embedding=MagicMock())
        face2.bbox.astype.return_value.__getitem__.side_effect = lambda i: [30,30,40,40][i]
        
        # Mock embedding division and astype for face2
        mock_div_result2 = MagicMock()
        face2.embedding.__truediv__.return_value = mock_div_result2
        mock_astype_result2 = MagicMock()
        mock_div_result2.astype.return_value = mock_astype_result2
        mock_astype_result2.tobytes.return_value = struct.pack('>512f', *([0.2]*512))

        self.mock_app_instance.get.return_value = [face1, face2]

        # Mock np.linalg.norm to always return 1.0
        self.mock_np.linalg.norm.return_value = 1.0
        self.mock_np.log.return_value = 1.0
        self.mock_np.sqrt.return_value = 1.0
        self.mock_np.exp.return_value = 0.5
        self.mock_cv2.Laplacian.return_value.var.return_value = 100.0

        # Mock cv2.imencode: Succeed for the first face, fail for the second
        mock_encoded_success = MagicMock()
        mock_encoded_success.tobytes.return_value = b'jpeg1'
        self.mock_cv2.imencode.side_effect = [
            (True, mock_encoded_success), # Success for face 1
            (False, None)                 # Failure for face 2
        ]

        result_bytes = self.worker.process_frame(b"fake_image_bytes")

        # Assertions
        self.assertEqual(result_bytes[0], 0x00) # Status should be success

        # Count should be 1, not 2, because the second face failed to encode
        count = struct.unpack('>I', result_bytes[1:5])[0]
        self.assertEqual(count, 1)

        # Verify the total length is correct for one face
        expected_len = 1 + 4 + 16 + 2048 + 4 + 4 + len(b'jpeg1')
        self.assertEqual(len(result_bytes), expected_len)

    def test_read_exactly(self):
        """Test that read_exactly handles partial reads correctly."""
        # Mock a stream that returns data in chunks
        mock_stream = MagicMock()
        # Side effect returns chunks: b'12', b'34', b'5', empty (EOF)
        mock_stream.read.side_effect = [b'12', b'34', b'5', b'']
        
        # We want 5 bytes. 
        data = self.worker.read_exactly(mock_stream, 5)
        self.assertEqual(data, b'12345')
        
        # Verify calls: 5-0=5, 5-2=3, 5-4=1
        self.assertEqual(mock_stream.read.call_count, 3)

    def test_read_exactly_short(self):
        """Test that read_exactly handles EOF before N bytes."""
        mock_stream = MagicMock()
        mock_stream.read.side_effect = [b'12', b''] # EOF after 2 bytes
        data = self.worker.read_exactly(mock_stream, 5)
        self.assertEqual(data, b'12')

if __name__ == '__main__':
    unittest.main()