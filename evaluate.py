"""
Evaluate the trained LeNet-style CNN.
Run after main.py: python evaluate.py
"""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_PATH = OUTPUT_DIR / "lenet_mnist.pth"
DATA_DIR = ROOT / "data"

if not MODEL_PATH.exists():
    raise FileNotFoundError("Model not found. Run 'python main.py' first.")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_dataset = datasets.MNIST(
    root=str(DATA_DIR), train=False, download=True, transform=transform
)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=0)


# Keep the architecture identical to the training script.
import torch.nn as nn

class LeNetStyleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def main():
    model = LeNetStyleCNN()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    all_predictions, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            predictions = model(images).argmax(dim=1)
            all_predictions.extend(predictions.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f"Test accuracy: {accuracy * 100:.2f}%\n")
    print("Classification report:")
    report = classification_report(all_labels, all_predictions, digits=4)
    print(report)
    (OUTPUT_DIR / "classification_report.txt").write_text(
        f"Test accuracy: {accuracy * 100:.2f}%\n\n{report}", encoding="utf-8"
    )

    matrix = confusion_matrix(all_labels, all_predictions, labels=list(range(10)))
    plt.figure(figsize=(9, 7))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.title("MNIST Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=180)
    plt.close()

    # Save examples including any errors, if present.
    errors = [i for i, (true, pred) in enumerate(zip(all_labels, all_predictions)) if true != pred]
    chosen = errors[:10] if errors else list(range(10))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, idx in zip(axes.flat, chosen):
        image, true_label = test_dataset[idx]
        pred_label = all_predictions[idx]
        ax.imshow(image.squeeze().numpy() * 0.3081 + 0.1307, cmap="gray")
        ax.set_title(f"Pred: {pred_label} | True: {true_label}",
                     color="red" if pred_label != true_label else "green")
        ax.axis("off")
    fig.suptitle("Misclassified examples (red) or sample predictions (green)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "evaluation_examples.png", dpi=160)
    plt.close(fig)

    print(f"Saved confusion matrix and report to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
