# 🛰️ Satellite Image Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.9-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Accuracy](https://img.shields.io/badge/Accuracy-89%25-green)

A Deep Learning application that classifies satellite imagery into 10 distinct terrain categories (e.g., Forest, Industrial, River) using a custom Convolutional Neural Network (CNN). The project features a robust model trained on the EuroSAT dataset and a user-friendly web interface built with Streamlit.

## 🔗 Live Demo
**[Click here to try the App](https://satellite.streamlit.app)**
*(Note: If the app is asleep, click "Wake up" and wait ~30 seconds)*

## 🧠 Project Overview
Visual navigation and automated mapping are critical for autonomous UAVs and disaster management. This project solves the problem of terrain identification by analyzing low-resolution satellite data.

**Key Challenges Solved:**
* **Scale Invariance:** Initial models confused "Rivers" with "Highways" due to geometric similarities. Implemented **Data Augmentation** (random rotations, zooms, flips) to force the model to learn texture over shape.
* **Real-time Inference:** Optimized the model architecture to run efficiently in a browser-based environment.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow / Keras (CNN)
* **Interface:** Streamlit (Python Web Framework)
* **Image Processing:** NumPy, Pillow
* **Training Hardware:** NVIDIA RTX 3050 (Laptop GPU)
* **Dataset:** [EuroSAT](https://github.com/phelber/eurosat) (Sentinel-2 satellite images)

## 📊 Model Performance
The custom CNN was trained for 20 epochs with the following results:
* **Training Accuracy:** ~89%
* **Validation Accuracy:** ~88-89%
* **Loss:** minimized without significant overfitting (verified via loss curves).

## 🚀 How to Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/raneyash21/satellite-classifier.git
