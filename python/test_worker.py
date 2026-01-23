import sys
import unittest
import struct
import importlib
from unittest.mock import MagicMock, patch

class TestWorkerLogic(unittest.TestCase):
    
    def setUp(self):
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

        # Since we clean up in tearDown, we can simply import fresh every time.
        self.worker = importlib.import_module('worker')

    def tearDown(self):
        # Stop patching to clean up
        self.modules_patcher.stop()
        # Remove worker from sys.modules so it doesn't pollute other tests
        if 'worker' in sys.modules:
            del sys.modules['worker']

    def test_process_frame_success(self):
        """Test that valid image data returns correct JSON structure."""
        
        # 1. Setup Mock AI responses
        # cv2.imdecode returns a mock array
        self.mock_cv2.imdecode.return_value = MagicMock()

        # Mock numpy array shape for thumbnail cropping
        mock_frame_array = MagicMock()
        mock_frame_array.shape = (100, 100, 3) # Height, Width, Channels
        self.mock_np.array.return_value = mock_frame_array

        # Simulate InsightFace returning one face object
        mock_face = MagicMock()
        mock_face.det_score = 0.99 # Must be a float for quality calc

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

if __name__ == '__main__':
    unittest.main()