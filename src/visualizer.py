import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self):
        pass

    def plot_history(self, history):
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(["Training", "Validation"], loc="upper right")

        plt.subplot(1, 2, 2)
        plt.plot(history.history["mean_absolute_error"])
        plt.plot(history.history["val_mean_absolute_error"])
        plt.title("Mean Absolute Error")
        plt.xlabel("Epoch")
        plt.ylabel("Value")
        plt.legend(["Training", "Validation"], loc="upper right")
        plt.show()
