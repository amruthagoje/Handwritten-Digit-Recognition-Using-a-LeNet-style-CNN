# Handwritten Digit Recognition Using a LeNet-style CNN

A deep learning project that recognizes handwritten digits (0–9) using a **Convolutional Neural Network (CNN)** inspired by the LeNet-5 architecture. Built with Python, PyTorch, and the MNIST dataset.

## 📌 Project Overview

Handwritten digit recognition is a computer vision task that classifies handwritten numerical digits from images. This project implements a LeNet-style CNN to learn visual patterns from grayscale images and predict the corresponding digit.

The project includes dataset preprocessing, model training, performance evaluation, and visualizations to understand how the CNN learns and classifies handwritten digits.

## 🎯 Objectives

- Build a CNN model for handwritten digit classification.
- Train the model using the MNIST dataset.
- Apply image normalization and preprocessing.
- Evaluate model performance on unseen test images.
- Visualize training loss, accuracy, predictions, confusion matrix, and feature maps.
- Save the trained model for future inference.

## 📊 Dataset: MNIST

The **MNIST dataset** is a widely used benchmark for handwritten digit recognition.

| Property | Description |
|---|---|
| Dataset | MNIST |
| Training images | 60,000 |
| Testing images | 10,000 |
| Image dimensions | 28 × 28 pixels |
| Image type | Grayscale |
| Number of classes | 10 (digits 0–9) |
| Data loader | `torchvision.datasets.MNIST` |

Each image contains a handwritten digit represented by a grayscale pixel matrix.

## 🧠 Model Architecture

The model uses a LeNet-style CNN consisting of convolutional layers, activation functions, pooling layers, and fully connected layers.

```text
Input Image (1 × 28 × 28)
          ↓
Convolution (1 → 6, 5 × 5, padding=2)
          ↓
        ReLU
          ↓
    Max Pooling (2 × 2)
          ↓
Convolution (6 → 16, 5 × 5)
          ↓
        ReLU
          ↓
    Max Pooling (2 × 2)
          ↓
       Flatten (400)
          ↓
Fully Connected (400 → 120)
          ↓
        ReLU
          ↓
Fully Connected (120 → 84)
          ↓
        ReLU
          ↓
Fully Connected (84 → 10)
          ↓
Predicted Digit (0–9)
```

### Key Components

- **Convolutional layers:** Extract important image features.
- **ReLU activation:** Introduces non-linearity.
- **Max pooling:** Reduces spatial dimensions.
- **Fully connected layers:** Perform digit classification.
- **Cross-entropy loss:** Measures classification error.
- **Adam optimizer:** Updates model parameters during training.

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Deep Learning Framework:** PyTorch
- **Dataset Library:** Torchvision
- **Numerical Computing:** NumPy
- **Visualization:** Matplotlib and Seaborn
- **Evaluation:** Scikit-learn
- **Development Environment:** Visual Studio Code

## 📂 Project Structure

```text
handwritten-digit-recognition-cnn/
│
├── main.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── outputs/
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── feature_maps.png
    ├── sample_predictions_1.png
    └── sample_predictions_2.png
```

**File descriptions:**

- `main.py` — Loads the dataset, defines and trains the CNN, and generates training visualizations.
- `evaluate.py` — Evaluates the saved model and generates evaluation metrics and visualizations.
- `requirements.txt` — Lists the required Python packages.
- `outputs/` — Stores generated graphs, feature maps, and prediction images.

*Note: Ensure that your scripts save the images with these exact filenames, or update the paths in this README to match your actual files.*

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/amruthagoje/handwritten-digit-recognition-cnn.git
cd handwritten-digit-recognition-cnn
```

Replace the URL if your repository has a different name.

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

Activate the environment on macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Train the Model

```bash
python main.py
```

The script loads the MNIST dataset, preprocesses images, trains the CNN, and generates the configured outputs.

### 5. Evaluate the Model

```bash
python evaluate.py
```

This evaluates the saved model on the test dataset and produces the evaluation metrics supported by the implementation.

## 📈 Model Training Configuration

The initial training configuration is:

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 |
| Batch size | 128 |
| Number of epochs | 5 |
| Loss function | Cross-Entropy Loss |
| Input dimensions | 28 × 28 |
| Output classes | 10 |

These parameters can be adjusted to experiment with model performance.

## 📊 Results and Visualizations

The following visualizations help analyze the model's learning process and classification performance. The images are stored in the `outputs/` directory.

### 1. Training Loss and Accuracy

The training curves show how the loss and accuracy change across epochs. They help track learning progress and identify possible overfitting.

![Training Loss and Accuracy](<img width="1600" height="580" alt="image" src="https://github.com/user-attachments/assets/ecf3a623-bd0f-4ba8-9861-210cb29f82e4" />
)

### 2. Confusion Matrix

The confusion matrix compares actual labels with predicted labels for each digit from 0 to 9. It helps identify correctly classified digits and common classification errors.

![Confusion Matrix](<img width="1310" height="1014" alt="image" src="https://github.com/user-attachments/assets/31e167c2-9319-4fe1-bdfb-d4cfef8cec63" />
)

### 3. CNN Feature Maps

Feature maps display activations produced by convolutional filters. They illustrate how the CNN responds to visual patterns such as strokes, edges, and digit shapes.

![CNN Feature Maps](<img width="1600" height="996" alt="image" src="https://github.com/user-attachments/assets/90a0ce19-238c-43b7-8acd-3e4e607faae8" />
)

### 4. Sample Predictions — Example 1

This visualization displays handwritten digit images alongside their actual labels and predicted labels.

![Sample Predictions Example 1](<img width="1600" height="665" alt="image" src="https://github.com/user-attachments/assets/4db12ba4-292a-4164-a98f-bde1d7cca76d" />
)

### 5. Sample Predictions — Example 2

A second set of test images demonstrates model predictions on additional examples, including misclassified digits if present.

![Sample Predictions Example 2](<img width="1600" height="661" alt="image" src="https://github.com/user-attachments/assets/c9d80f21-a670-490e-9828-7ceb5d2fea5e" />
)

### 📌 Performance Metrics

Record the metrics obtained from your actual model evaluation.

| Metric | Result |
|---|---|
| Test accuracy | Add measured value |
| Precision | Add measured value |
| Recall | Add measured value |
| F1-score | Add measured value |

*Do not report estimated values as actual results. Update this table after running the evaluation script.*

## 🌍 Applications

- Postal code and ZIP code recognition.
- Bank cheque processing.
- Digit extraction from scanned documents.
- Automated form and document digitization.
- Educational computer vision applications.
- Optical character recognition (OCR) systems.

## 🚀 Future Enhancements

- Build a web interface for drawing handwritten digits.
- Add image-upload support for digit classification.
- Experiment with deeper CNN architectures and hyperparameters.
- Apply data augmentation to improve generalization.
- Deploy the model as a web application or API.
- Extend recognition to letters and more complex characters.

## 🎓 Learning Outcomes

Through this project, I explored:

- CNN architecture and image classification.
- Dataset loading and image normalization.
- Model training using PyTorch.
- Loss functions and optimization algorithms.
- Model evaluation using classification metrics.
- Confusion matrix interpretation.
- Visualization of predictions and learned feature maps.

## 👩‍💻 Author

**Amrutha Goje**

- GitHub: [@amruthagoje](https://github.com/amruthagoje)
- LinkedIn: [Amrutha Goje](https://www.linkedin.com/in/amrutha-goje/)

## 📚 References

1. LeCun, Y., et al., “Gradient-Based Learning Applied to Document Recognition,” *Proceedings of the IEEE*, 1998.
2. [MNIST Database of Handwritten Digits](http://yann.lecun.com/exdb/mnist/)
3. [PyTorch Documentation](https://pytorch.org/docs/stable/)
4. [Torchvision Documentation](https://docs.pytorch.org/vision/stable/)
5. [Scikit-learn Documentation](https://scikit-learn.org/stable/)


---

⭐ If you find this project useful, consider giving the repository a star!
