import unittest
import tensorflow as tf
from src.model import Model

class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model((64, 64, 3))

    def test_build_model(self):
        self.model.build_model()
        self.assertTrue(len(self.model.model.layers) == 9)

    def test_compile_model(self):
        self.model.compile_model(self.model.model)
        self.assertTrue(self.model.model.optimizer.name == "Adam")
        self.assertTrue(self.model.model.loss == "mse")

    def test_train_model(self):
        X_train = np.random.rand(100, 64, 64, 3)
        y_train = np.random.rand(100)
        X_val = np.random.rand(50, 64, 64, 3)
        y_val = np.random.rand(50)
        history = self.model.train_model(self.model.model, X_train, y_train, X_val, y_val)
        self.assertTrue(len(history.history.keys()) == 2)
        self.assertTrue("loss" in history.history.keys())
        self.assertTrue("val_loss" in history.history.keys())
