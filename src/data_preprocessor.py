import numpy as np

class DataPreprocessor:
    def __init__(self, images, steering_angles):
        self.images = images
        self.steering_angles = steering_angles

    def preprocess_data(self):
        X = self.images / 255.0
        y = self.steering_angles
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_val, y_train, y_val
