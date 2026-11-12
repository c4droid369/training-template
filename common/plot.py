from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def plot_history(history: dict, out_path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["train_loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("loss"); axes[0].legend()
    axes[1].plot(history["train_acc"], label="train")
    axes[1].plot(history["val_acc"], label="val")
    axes[1].set_title("acc"); axes[1].legend()
    fig.tight_layout()
    fig.savefig(Path(out_path), dpi=120)
    plt.show()


def plot_loss_curve(steps, losses, out_path):
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("step"); plt.ylabel("loss")
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, out_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.savefig(out_path, dpi=120)
    plt.show()