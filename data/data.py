import os
import numpy as np
import cv2

def load_data(data_dir):
    """
    Loads the driving data from the specified directory.

    Args:
        data_dir (str): The directory path containing the driving data.

    Returns:
        images (list): A list of image file paths.
        angles (list): A list of corresponding steering angles.
    """
    images = []
    angles = []
    for file in os.listdir(data_dir):
        if file.endswith(".jpg"):
            image_path = os.path.join(data_dir, file)
            image = cv2.imread(image_path)
            images.append(image)
            angle = float(file.split("_")[0])
            angles.append(angle)
    return images, angles

def preprocess_image(image):
    """
    Preprocesses a single image by resizing and normalizing it.

    Args:
        image (numpy.ndarray): A single image as a 3D numpy array.

    Returns:
        preprocessed_image (numpy.ndarray): The preprocessed image as a 3D numpy array.
    """
    # Resize the image to a fixed size
    preprocessed_image = cv2.resize(image, (66, 200))

    # Normalize the pixel values to be between -1 and 1
    preprocessed_image = preprocessed_image / 255.0 * 2.0 - 1.0

    return preprocessed_image

def preprocess_data(images, angles):
    """
Preprocesses the driving data by applying the preprocessing function to each image.

    Args:
        images (list): A list of image file paths.
        angles (list): A list of corresponding steering angles.

    Returns:
        preprocessed_images (list): A list of preprocessed images.
        preprocessed_angles (list): A list of preprocessed steering angles.
    """
    preprocessed_images = []
    preprocessed_angles = []
    for image, angle in zip(images, angles):
        preprocessed_image = preprocess_image(image)
        preprocessed_images.append(preprocessed_image)
        preprocessed_angles.append(angle)
    return preprocessed_images, preprocessed_angles
