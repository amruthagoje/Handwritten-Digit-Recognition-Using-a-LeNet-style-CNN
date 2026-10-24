"""
Handwritten Digit Recognition Using a LeNet-Style CNN
Run: python main.py
Downloads MNIST automatically on the first run.
"""
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# MNIST pixels are converted to tensors and normalized.
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root=str(DATA_DIR), train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root=str(DATA_DIR), train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class LeNetStyleCNN(nn.Module):
    """A compact LeNet-style CNN for 28x28 grayscale digit images."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, padding=2),  # 28x28 -> 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),              # 14x14 -> 10x10
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),  # Ten output scores: digits 0 through 9
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def run_epoch(model, loader, loss_function, optimizer=None):
    training = optimizer is not None
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            if training:
                optimizer.zero_grad()

            logits = model(images)
            loss = loss_function(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

    return total_loss / total, 100.0 * correct / total


def save_training_plot(train_losses, train_accuracies, test_accuracies):
    epochs = range(1, len(train_losses) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_losses, marker="o")
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, train_accuracies, marker="o", label="Train accuracy")
    axes[1].plot(epochs, test_accuracies, marker="o", label="Test accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_curves.png", dpi=160)
    plt.close(fig)


def save_feature_maps(model, dataset):
    """Save first-layer convolution maps for one test digit."""
    model.eval()
    image, label = dataset[0]
    with torch.no_grad():
        first_conv = model.features[0]
        maps = first_conv(image.unsqueeze(0))

    fig, axes = plt.subplots(2, 3, figsize=(8, 5))
    fig.suptitle(f"First Convolution Layer Feature Maps (true digit: {label})")
    for i, ax in enumerate(axes.flat):
        ax.imshow(maps[0, i].numpy(), cmap="viridis")
        ax.set_title(f"Filter {i + 1}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_maps.png", dpi=160)
    plt.close(fig)


def save_sample_predictions(model, dataset):
    model.eval()
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    with torch.no_grad():
        for i, ax in enumerate(axes.flat):
            image, label = dataset[i]
            predicted = model(image.unsqueeze(0)).argmax(dim=1).item()
            # Undo normalization for display.
            display_image = image.squeeze().numpy() * 0.3081 + 0.1307
            ax.imshow(display_image, cmap="gray")
            ax.set_title(f"Pred: {predicted} | True: {label}",
                         color="green" if predicted == label else "red")
            ax.axis("off")
    fig.suptitle("Sample MNIST Predictions (green = correct, red = incorrect)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=160)
    plt.close(fig)


def main():
    print("Device: CPU (suitable for this small MNIST project)")
    print(f"Training images: {len(train_dataset):,}")
    print(f"Testing images:  {len(test_dataset):,}")
    print("\nModel architecture:\n")
    model = LeNetStyleCNN()
    print(model)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_losses, train_accuracies, test_accuracies = [], [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss, train_acc = run_epoch(model, train_loader, loss_function, optimizer)
        test_loss, test_acc = run_epoch(model, test_loader, loss_function)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train loss: {train_loss:.4f} | "
            f"Train accuracy: {train_acc:.2f}% | "
            f"Test loss: {test_loss:.4f} | "
            f"Test accuracy: {test_acc:.2f}%"
        )

    model_path = OUTPUT_DIR / "lenet_mnist.pth"
    torch.save(model.state_dict(), model_path)
    save_training_plot(train_losses, train_accuracies, test_accuracies)
    save_feature_maps(model, test_dataset)
    save_sample_predictions(model, test_dataset)

    print("\nTraining finished.")
    print(f"Saved model: {model_path}")
    print(f"Saved results in: {OUTPUT_DIR}")
    print("Next run: python evaluate.py")


if __name__ == "__main__":
    main()
