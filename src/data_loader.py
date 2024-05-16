import os
import cv2
import numpy as np

class DataLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_images(self):
        images = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                img = cv2.imread(os.path.join(self.data_dir, filename))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (64, 64))
                images.append(img)
        return np.array(images)

    def load_steering_angles(self):
        steering_angles = []
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(self.data_dir, filename), "r") as f:
                    steering_angle = float(f.read().strip())
                    steering_angles.append(steering_angle)
        return np.array(steering_angles)
