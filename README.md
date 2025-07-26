A deep learning project comparing dense and convolutional networks for handwritten digit recognition using TensorFlow and the MNIST dataset.

# 🧠 Handwritten Digit Recognition using Deep Learning

This project demonstrates two types of neural network architectures — a Fully Connected Neural Network and a Convolutional Neural Network (CNN) — to classify handwritten digits from the MNIST dataset using **TensorFlow/Keras**.

---

## 📂 Project Overview

- **Dataset**: MNIST (70,000 images of handwritten digits, 28×28 grayscale)
- **Models Used**:
  - Dense Neural Network
  - Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Tools**: Jupyter Notebook, NumPy, Matplotlib, Seaborn

---

## 🚀 Model Architectures

### 🔷 1. Fully Connected Neural Network

| Layer   | Type    | Units | Activation |
|---------|---------|-------|------------|
| Input   | Flatten | -     | -          |
| Hidden  | Dense   | 128   | ReLU       |
| Output  | Dense   | 10    | Softmax    |

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 5  
- **Accuracy Achieved**: ~97.4%

---

### 🔶 2. Convolutional Neural Network (CNN)

| Layer       | Type        | Filters | Kernel Size | Activation |
|-------------|-------------|---------|-------------|------------|
| Conv2D      | Convolution | 32      | 3×3         | ReLU       |
| MaxPooling2D| Pooling     | 2×2     |             |            |
| Flatten     |             |         |             |            |
| Dense       | Hidden      | 64      |             | ReLU       |
| Dense       | Output      | 10      |             | Softmax    |

- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Epochs**: 5  
- **Accuracy Achieved**: ~98.4%

---

## 📊 Performance Comparison

| Model | Test Accuracy |
|-------|---------------|
| Dense | 97.4%         |
| CNN   | 98.4%         |

---

## 📌 Why These Parameters?

- **ReLU**: Non-linear activation; avoids vanishing gradients.
- **Softmax**: Ideal for multi-class classification.
- **Adam**: Adaptive optimizer that combines RMSProp and momentum.
- **3×3 kernels**: Standard for capturing spatial features.
- **128 / 64 units**: Balance between performance and overfitting.

---

## 🔍 Results Visualization

- Misclassified digit samples shown using matplotlib
- Used confusion matrix (with `seaborn`) for deeper evaluation

---

## 📁 Project Structure

```
digit-recognition/
├── digit_recognition.ipynb       # Jupyter notebook with both models
└── README.md                     # This file
```
