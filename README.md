#  Handwritten Digit Recognition using Deep Learning

This project demonstrates the use of two types of neural network architectures — a Fully Connected Neural Network and a Convolutional Neural Network (CNN) — to classify handwritten digits from the MNIST dataset using **TensorFlow/Keras**.

##  Project Overview

- **Dataset**: MNIST (70,000 images of handwritten digits, 28×28 grayscale)
- **Models Used**:
  - Dense Neural Network
  - Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Tools**: Jupyter Notebook, NumPy, Matplotlib, Seaborn

---

##  Model Architectures

###  1. Fully Connected Neural Network

| Layer      | Type      | Units | Activation |
|------------|-----------|-------|------------|
| Flatten    | Input     | -     | -          |
| Dense      | Hidden    | 128   | ReLU       |
| Dense      | Output    | 10    | Softmax    |

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 5  
- **Accuracy Achieved**: ~97.4% on test data  

---

###  2. Convolutional Neural Network (CNN)

| Layer      | Type        | Filters | Kernel Size | Activation |
|------------|-------------|---------|-------------|------------|
| Conv2D     | Convolution | 32      | (3×3)       | ReLU       |
| MaxPooling | Pooling     | (2×2)   |             |            |
| Flatten    |             |         |             |            |
| Dense      | Hidden      | 64      |             | ReLU       |
| Dense      | Output      | 10      |             | Softmax    |

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 5  
- **Accuracy Achieved**: ~98.4% on test data  

---

##  Performance Comparison

| Model | Test Accuracy |
|-------|---------------|
| Dense | 97.4%         |
| CNN   | 98.4%         |

CNN outperforms the dense model, especially in handling spatial patterns in images due to convolutional lay
