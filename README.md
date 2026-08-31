# 🫁 PneumoVision

## Deep Learning-Based Pneumonia Detection from Chest X-Ray Images

PneumoVision is an end-to-end **Artificial Intelligence and Machine Learning application** that uses deep learning to classify chest X-ray images as **NORMAL** or **PNEUMONIA**.

The project uses **transfer learning with EfficientNetB0**, followed by model fine-tuning, and integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations for the model's predictions.

The trained model is served through an interactive **Streamlit web application**, allowing users to upload a chest X-ray and receive a prediction, confidence score, probability distribution, and Grad-CAM visualization.

---

## 🚀 Live Demo

> **Live Demo:** Coming soon

The application will be deployed as a publicly accessible Streamlit web application.

---

## ✨ Features

- 🫁 Chest X-ray image classification
- 🧠 EfficientNetB0-based deep learning model
- 🔄 Transfer learning and fine-tuning
- 🎯 Binary classification:
  - NORMAL
  - PNEUMONIA
- 📊 Model evaluation using multiple performance metrics
- 🔥 Grad-CAM explainability
- 📈 Pneumonia and Normal probability scores
- 🌐 Interactive Streamlit web application
- 💻 Easy local setup using `requirements.txt`

---

## 🧠 How It Works

The PneumoVision pipeline consists of the following stages:

```text
Chest X-Ray Image
        │
        ▼
Image Preprocessing
        │
        ▼
Data Augmentation
        │
        ▼
EfficientNetB0
(Transfer Learning)
        │
        ▼
Fine-Tuning
        │
        ▼
Binary Classification
        │
        ├───────────────┐
        ▼               ▼
     NORMAL         PNEUMONIA
        │               │
        └───────┬───────┘
                ▼
       Prediction + Confidence
                │
                ▼
            Grad-CAM
                │
                ▼
       Visual Explanation