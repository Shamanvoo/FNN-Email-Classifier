# FNN-Email-Classifier

# 📧 Email Classification Using Neural Networks

## Overview

This project aims to classify emails into three categories — **Ham**, **Spam**, and **Phishing** — using deep learning techniques. Our team explored two different neural network architectures:
- A **Feedforward Neural Network (FNN)** using a multilayer perceptron (MLP) design
- A **Long Short-Term Memory (LSTM)** model to capture sequential dependencies in the email text

Our goal was to benchmark and compare the performance of both models on curated datasets and determine which architecture yields better classification accuracy.

---

## 📂 Dataset

We utilized high-quality, diverse email datasets from Kaggle. These datasets were selected for their realistic distribution and reliable labeling.

### Final dataset composition:
- **Ham (Legitimate)**: 50,592 samples
- **Phishing**: 42,891 samples
- **Spam**: 2,726 samples

While the dataset is imbalanced, this reflects real-world email traffic, where phishing and legitimate messages tend to be more prevalent than traditional spam.

---

## ⚙️ Data Preprocessing Pipeline

We implemented a robust preprocessing pipeline to clean and prepare the textual and metadata features for model training:

### 1. Text Cleaning
- Lowercasing
- Removing special characters and non-ASCII symbols
- Removing stop words
- Handling email-specific HTML formatting

### 2. Tokenization
- Tokenizing email content
- Preserving domain-specific terms
- Using n-grams for contextual insight

### 3. Lemmatization
- Converting words to base forms
- Preserving semantic meaning
- Enhancing generalization

### 4. Feature Engineering
- TF-IDF vectorization
- Extracting email metadata (headers, sender info)
- URL and domain analysis for phishing indicators
- Normalization of feature vectors

---

## 🧠 Model Architectures

We implemented and trained two different models for benchmarking:

### 1. Feedforward Neural Network (FNN / MLP)
- **Input Layer**: TF-IDF feature vector
- **Hidden Layers**: 2 layers, each with 1024 neurons and ReLU activation
- **Output Layer**: 3 neurons with Softmax activation (Ham, Spam, Phishing)

### 2. Long Short-Term Memory Model (LSTM)
- **Input Layer**: Text sequence input
- **LSTM Layer**: 2 stacked layers with 128 units, Dropout = 0.1
- **Fully Connected Layer**: Maps LSTM output to 3 classes
- **Output**: Final hidden state used for classification

---

## 🔧 Hyperparameters

Common hyperparameters for both models include:
- **Batch Size**: 512
- **Hidden Layer Size**: 1024
- **Output Size**: 3
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Max Epochs**: 1000

---

## 📌 Future Work

- Incorporating transformer-based architectures like BERT
- Improved data augmentation for spam class
- Real-time email filtering API integration

---

## 📄 License

This project is licensed under the MIT License. See `LICENSE` for more information.