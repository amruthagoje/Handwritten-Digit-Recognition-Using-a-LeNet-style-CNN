# Handwritten Digit Recognition Using a LeNet-Style CNN

A beginner-friendly Deep Learning project using PyTorch and the MNIST dataset.

## Requirements
- Python 3.11, 3.12, or a compatible Python 3.13 installation
- VS Code and the Microsoft Python extension
- Internet connection for the first MNIST download

## Setup on macOS

Open this folder in VS Code, then run these commands in the integrated terminal:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

In VS Code, select the `.venv` interpreter using `Cmd + Shift + P` → `Python: Select Interpreter`.

## Train the model

```bash
python main.py
```

The first run downloads MNIST automatically. It trains for five epochs and saves the model and visualizations in `outputs/`.

## Evaluate the model

After training finishes:

```bash
python evaluate.py
```

## Output files

- `outputs/lenet_mnist.pth` — trained model weights
- `outputs/training_curves.png` — loss and accuracy plots
- `outputs/feature_maps.png` — first convolution layer's feature maps
- `outputs/sample_predictions.png` — sample predictions
- `outputs/confusion_matrix.png` — 10-class confusion matrix
- `outputs/classification_report.txt` — precision, recall, F1-score and accuracy
- `outputs/evaluation_examples.png` — error examples (or correct examples if none are misclassified)

## Architecture

Input 1 × 28 × 28 → Conv(1,6,5) → ReLU → MaxPool → Conv(6,16,5) → ReLU → MaxPool → Flatten → FC(400,120) → FC(120,84) → FC(84,10).

This is LeNet-style rather than an exact reproduction of the original 1998 LeNet-5 architecture.

## Dataset

MNIST is loaded through `torchvision.datasets.MNIST`. It has 60,000 training and 10,000 test grayscale images, each 28 × 28 pixels.

## Important
The exact test accuracy depends on the training run and environment. Record the accuracy printed by your own run in your report; do not invent results.
