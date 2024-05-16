import cv2
import numpy as np

def load_image(image_path):
    """
    Loads an image from the specified file path.

    Args:
        image_path (str): The file path of the image.

    Returns:
        image (numpy.ndarray): The loaded image as a 3D numpy array.
    """
    image = cv2.imread(image_path)
    return image

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
