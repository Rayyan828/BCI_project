import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_history(history):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history['accuracy'], label='train')
    plt.plot(history['val_accuracy'], label='val')
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(history['loss'], label='train')
    plt.plot(history['val_loss'], label='val')
    plt.title("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig("training_history.png", dpi=300)
    plt.show()

if __name__ == '__main__':
    if not os.path.exists("history.npy"):
        print("history.npy not found. Run training first.")
    else:
        h = np.load("history.npy", allow_pickle=True).item()
        plot_history(h)
