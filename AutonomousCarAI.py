import os
import numpy as np
import cv2
import tensorflow as tf
from images.image import load_images
from steering_angles.steering_angle import load_steering_angle
from trained_model.training_model import build_model, train_model

def main():
    # Set the paths for the data directories
    data_dir = "data"
    images_dir = os.path.join(data_dir, "IMG")
    steering_angles_dir = os.path.join(data_dir, "steering_angles")

    # Load the images and steering angles
    images = load_images(images_dir)
    steering_angles = load_steering_angle(steering_angles_dir)

    # Preprocess the data
    X_train, y_train, X_val, y_val = preprocess_data(images, steering_angles)

    # Build and train the deep learning model
    input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3])
    model = build_model(input_shape)
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Save the trained model
    model.save("trained_model/model.h5")

def preprocess_data(images, steering_angles):
    """
    Preprocesses the data for training.

    Args:
        images (list): The list of images.
        steering_angles (list): The list of steering angles.

    Returns:
        X_train (numpy.ndarray): The training input data.
        y_train (numpy.ndarray): The training target data.
        X_val (numpy.ndarray): The validation input data.
        y_val (numpy.ndarray): The validation target data.
    """
    # Preprocess the images and steering angles
    X = np.array([cv2.resize(image, (64, 64)) for image in images])
    y = np.array(steering_angles)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize the input data
    X_train = X_train / 255.0
    X_val = X_val / 255.0

    # Reshape the input data for the deep learning model
    X_train = X_train.reshape(-1, 64, 64, 3)
    X_val = X_val.reshape(-1, 64, 64, 3)

    return X_train, y_train, X_val, y_val

if __name__ == "__main__":
    main()
