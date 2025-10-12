# Handwritten Digit Recognition using CNN

This project demonstrates handwritten digit recognition using Convolutional Neural Networks (CNN) on the MNIST dataset.

---

## Dataset

The MNIST dataset consists of 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is 28x28 pixels in grayscale.

---

## Model Architecture

- **Conv2D Layer**: 64 filters, kernel size (3,3), ReLU activation
- **BatchNormalization Layer**
- **MaxPooling2D Layer**: Pool size (2,2)
- **Conv2D Layer**: 64 filters, kernel size (3,3), ReLU activation
- **BatchNormalization Layer**
- **MaxPooling2D Layer**: Pool size (2,2)
- **Flatten Layer**
- **Dense Layer**: 128 units, ReLU activation
- **Dropout Layer**: Rate 0.2
- **Dense Layer**: 64 units, ReLU activation
- **Dropout Layer**: Rate 0.2
- **Dense Layer**: 10 units, Softmax activation

---

## Training Details

- **Optimizer**: Adam
- **Loss Function**: Sparse Categorical Crossentropy
- **Metrics**: Accuracy
- **Epochs**: 10
- **Validation Split**: 20%

---

## Results

The model achieves ~98% accuracy on the MNIST test dataset.

---

## How to Run

```bash
pip install -r requirements.txt
