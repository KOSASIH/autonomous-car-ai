def load_steering_angle(data_dir):
    """
    Loads the steering angles from the specified directory.

    Args:
        data_dir (str): The directory path containing the steering angles.

    Returns:
        steering_angles (list): A list of steering angles.
    """
    steering_angles = []
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            angle = float(file.split("_")[0])
            steering_angles.append(angle)
    return steering_angles
