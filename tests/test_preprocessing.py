import os
import unittest
import numpy as np
import cv2
from src.preprocessing import MRIPreprocessor

class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        # Mock configuration dictionary
        self.config = {
            "data": {
                "image_size": [224, 224]
            },
            "preprocessing": {
                "clahe_clip_limit": 2.0,
                "clahe_tile_grid_size": [8, 8],
                "gaussian_kernel_size": [5, 5],
                "gaussian_sigma": 1.0,
                "enable_skull_strip": True
            }
        }
        self.preprocessor = MRIPreprocessor(self.config)
        
        # Create a synthetic image for test runs
        self.test_img = np.zeros((100, 100), dtype=np.uint8)
        # Draw a central shape (simulating a brain)
        cv2.circle(self.test_img, (50, 50), 30, 120, -1)
        # Draw an outer skull ring
        cv2.circle(self.test_img, (50, 50), 45, 240, 2)
        
        # Save temp test image to disk
        self.temp_img_path = "tests_temp_mri_slice.png"
        cv2.imwrite(self.temp_img_path, self.test_img)

    def tearDown(self):
        if os.path.exists(self.temp_img_path):
            os.remove(self.temp_img_path)

    def test_gaussian_blur(self):
        blurred = self.preprocessor.gaussian_blur(self.test_img)
        self.assertEqual(blurred.shape, self.test_img.shape)
        self.assertEqual(blurred.dtype, np.uint8)

    def test_apply_clahe(self):
        clahe_img = self.preprocessor.apply_clahe(self.test_img)
        self.assertEqual(clahe_img.shape, self.test_img.shape)

    def test_skull_strip(self):
        stripped = self.preprocessor.skull_strip(self.test_img)
        self.assertEqual(stripped.shape, self.test_img.shape)
        # Verify that skull pixels (value 240) are successfully stripped to 0
        skull_indices = np.where(self.test_img == 240)
        stripped_skull_values = stripped[skull_indices]
        num_stripped = np.sum(stripped_skull_values == 0)
        self.assertTrue(num_stripped > 0, "Failed to strip any skull pixels.")

    def test_full_preprocessing_pipeline(self):
        preprocessed = self.preprocessor.preprocess_image(self.temp_img_path)
        # Size must match config (224, 224, 3)
        self.assertEqual(preprocessed.shape, (224, 224, 3))
        self.assertEqual(preprocessed.dtype, np.uint8)

if __name__ == "__main__":
    unittest.main()
