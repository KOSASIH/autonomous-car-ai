import unittest
import numpy as np
from src.data_preprocessor import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.images = np.random.rand(100, 64, 64, 3)
        self.steering_angles = np.random.rand(100)
        self.data_preprocessor= DataPreprocessor(self.images, self.steering_angles)

    def test_preprocess_data(self):
        X, y = self.data_preprocessor.preprocess_data()
        self.assertTrue(X.shape[0] == 100)
        self.assertTrue(X.shape[1] == 64)
        self.assertTrue(X.shape[2] == 64)
        self.assertTrue(X.shape[3] == 3)
        self.assertTrue(np.min(X) >= 0)
        self.assertTrue(np.max(X) <= 1)
        self.assertTrue(y.shape[0] == 100)
        self.assertTrue(np.min(y) >= -1)
        self.assertTrue(np.max(y) <= 1)

    def test_split_data(self):
        X, y = self.data_preprocessor.preprocess_data()
        X_train, X_val, y_train, y_val = self.data_preprocessor.split_data(X, y, test_size=0.2, random_state=42)
        self.assertTrue(X_train.shape[0] == int(0.8 * 100))
        self.assertTrue(X_val.shape[0] == int(0.2 * 100))
        self.assertTrue(np.all(y_train == y[0:int(0.8 * 100)]))
        self.assertTrue(np.all(y_val == y[int(0.8 * 100):]))
