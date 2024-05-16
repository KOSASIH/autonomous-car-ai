def load_steering_angles(data_dir):
    """
    Loads the steering angles data from the specified directory.

    Args:
        data_dir (str): The directory path containing the steering angles data.

    Returns:
        steering_angles (list): A list of steering angles.
    """
    steering_angles = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), 'r') as f:
                angle = float(f.read().strip())
                steering_angles.append(angle)
    return steering_angles
