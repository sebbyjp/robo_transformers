# import unittest
# from unittest.mock import patch, Mock
# import numpy as np
# from PIL import Image
# from robo_transformers.inference_server import InferenceServer

# class TestInferenceServer(unittest.TestCase):
#     def setUp(self):
#         self.inference_server = InferenceServer()

#     @patch('numpy.array')
#     @patch.object(Image, 'fromarray')
#     def test_call_with_image(self, mock_fromarray, mock_np_array):
#         mock_image = Mock()
#         mock_np_array.return_value = mock_image
#         self.inference_server(save=True, image=np.ones((480, 640, 3), dtype=np.uint8), instruction="test")
#         mock_np_array.assert_called_once_with(np.ones((480, 640, 3), dtype=np.uint8))
#         mock_fromarray.assert_called_once_with(mock_image)

#     def test_call_without_image(self):
#         result = self.inference_server()
#         self.assertIsNone(result)