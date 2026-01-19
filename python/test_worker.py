import sys
import unittest
import json
from unittest.mock import MagicMock, patch

# --- MOCKING HEAVY DEPENDENCIES ---
# We mock ALL heavy libraries before importing worker.
# This allows tests to run in CI with ONLY `pip install pytest`.
mock_fr = MagicMock()
mock_np = MagicMock()
mock_pil = MagicMock()

sys.modules["face_recognition"] = mock_fr
sys.modules["numpy"] = mock_np
sys.modules["PIL"] = mock_pil
sys.modules["PIL.Image"] = mock_pil.Image

# Now we can safely import our worker
from worker import process_frame

class TestWorkerLogic(unittest.TestCase):
    
    def setUp(self):
        # Reset mocks before each test
        mock_fr.reset_mock()
        mock_np.reset_mock()
        mock_pil.reset_mock()

    def test_process_frame_success(self):
        """Test that valid image data returns correct JSON structure."""
        
        # 1. Setup the Mock AI responses
        # When worker calls Image.open().convert("RGB")...
        mock_image_obj = MagicMock()
        mock_pil.Image.open.return_value.convert.return_value = mock_image_obj

        # Simulate finding one face
        mock_fr.face_locations.return_value = [(10, 20, 30, 40)]
        # Simulate a 128-d vector
        fake_vec = [0.1] * 128
        mock_fr.face_encodings.return_value = [MagicMock(tolist=lambda: fake_vec)]
        
        # 2. Run the function under test
        result_json = process_frame(b"fake_image_bytes")
        
        # 3. Assertions
        result = json.loads(result_json)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['loc'], [10, 20, 30, 40])
        self.assertEqual(result[0]['vec'], fake_vec)

    def test_process_frame_error_handling(self):
        """Test that invalid data returns a JSON error object."""
        # Force Image.open to raise an exception
        mock_pil.Image.open.side_effect = Exception("Corrupt Data")

        result_json = process_frame(b"garbage")
        result = json.loads(result_json)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Corrupt Data")

if __name__ == '__main__':
    unittest.main()