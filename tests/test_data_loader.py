import unittest
import os
import numpy as np
import cv2
from src.data_loader import DataLoader

class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.data_dir = "data/test"
        self.data_loader = DataLoader(self.data_dir)

    def test_load_images(self):
        images = self.data_loader.load_images()
        self.assertTrue(len(images) > 0)
        self.assertTrue(images.shape[3] == 3)
        self.assertTrue(np.max(images) <= 255)

    def test_load_steering_angles(self):
        steering_angles = self.data_loader.load_steering_angles()
        self.assertTrue(len(steering_angles) > 0)
        self.assertTrue(np.min(steering_angles) >= -1)
        self.assertTrue(np.max(steering_angles) <= 1)

    def test_load_data(self):
        images, steering_angles = self.data_loader.load_data()
        self.assertTrue(len(images) == len(steering_angles))

    def test_load_data_with_flipping(self):
        images, steering_angles = self.data_loader.load_data(flip_images=True)
        self.assertTrue(len(images) == len(steering_angles))
        self.assertTrue(np.any(images[:, :, :, 0] != images[0, :, :, 0]))

    def test_load_data_with_brightness_augmentation(self):
        images, steering_angles = self.data_loader.load_data(brightness_range=(0.5, 1.5))
        self.assertTrue(len(images) == len(steering_angles))
        self.assertTrue(np.any(images[:, :, :, 0] != images[0, :, :, 0]))

    def test_load_data_with_flipping_and_brightness_augmentation(self):
        images, steering_angles = self.data_loader.load_data(flip_images=True, brightness_range=(0.5, 1.5))
        self.assertTrue(len(images) == len(steering_angles))
        self.assertTrue(np.any(images[:, :, :, 0] != images[0, :, :, 0]))

if __name__ == '__main__':
    unittest.main()
