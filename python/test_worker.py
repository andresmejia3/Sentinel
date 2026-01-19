import sys
import unittest
import json
import importlib
from unittest.mock import MagicMock, patch

class TestWorkerLogic(unittest.TestCase):
    
    def setUp(self):
        # 1. Create fresh mocks for every single test run
        self.mock_insightface = MagicMock()
        self.mock_np = MagicMock()
        self.mock_pil = MagicMock()
        
        # 2. Patch sys.modules so 'import worker' sees our mocks instead of real libraries
        self.modules_patcher = patch.dict(sys.modules, {
            "insightface": self.mock_insightface,
            "numpy": self.mock_np,
            "PIL": self.mock_pil,
            "PIL.Image": self.mock_pil.Image,
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
        
        # 1. Setup the Mock AI responses
        # When worker calls Image.open().convert("RGB")...
        mock_image_obj = MagicMock()
        self.mock_pil.Image.open.return_value.convert.return_value = mock_image_obj

        # Simulate InsightFace returning one face object
        mock_face = MagicMock()
        # InsightFace returns a bbox as a numpy array, we mock the astype().tolist() chain
        mock_face.bbox.astype.return_value.tolist.return_value = [10, 20, 30, 40]
        # InsightFace returns a 512-d embedding
        fake_vec = [0.1] * 512
        mock_face.embedding.tolist.return_value = fake_vec
        
        self.mock_app_instance.get.return_value = [mock_face]
        
        # 2. Run the function under test
        result_json = self.worker.process_frame(b"fake_image_bytes")
        
        # 3. Assertions
        result = json.loads(result_json)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['loc'], [10, 20, 30, 40])
        self.assertEqual(result[0]['vec'], fake_vec)

    def test_process_frame_error_handling(self):
        """Test that invalid data returns a JSON error object."""
        # Force Image.open to raise an exception
        self.mock_pil.Image.open.side_effect = Exception("Corrupt Data")

        result_json = self.worker.process_frame(b"garbage")
        result = json.loads(result_json)
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Corrupt Data")

if __name__ == '__main__':
    unittest.main()