import unittest
import matplotlib.pyplot as plt
from src.model import Model

class TestVisualizer(unittest.TestCase):
    def setUp(self):
        self.model = Model((64, 64, 3))
        self.model.compile_model(self.model.model)
        self.history = self.model.train_model(self.model.model, X_train, y_train, X_val, y_val, epochs=10)

    def test_plot_history(self):
        Visualizer().plot_history(self.history)
        plt.show()
