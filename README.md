# Pneumonia Detection from Chest X-ray Images

## Project Overview
This project demonstrates a simple deep learning pipeline for detecting pneumonia from chest X-ray images using Transfer Learning.

The model uses MobileNetV2, a pre-trained Convolutional Neural Network (CNN), to classify chest X-ray images into two categories:
- NORMAL
- PNEUMONIA

**This project is for educational purposes only and is not intended for clinical diagnosis or medical decision-making.**

## Dataset
The project uses the Chest X-ray Images (Pneumonia) dataset from Kaggle.

Dataset: [Chest X-ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

The dataset contains two classes:
- NORMAL
- PNEUMONIA

The dataset is downloaded automatically using `kagglehub` in the notebook.

## Technologies Used
- Python
- TensorFlow & Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- KaggleHub

## Project Workflow
1. Dataset Download and Exploration
2. Image Loading and Preprocessing
3. Model Building using Transfer Learning
4. Model Training
5. Model Evaluation
6. Predictions on Unseen Test Images

## Model Architecture
MobileNetV2 is used as a pre-trained feature extractor with its original layers frozen.

A custom classification head is added, including:
- Global Average Pooling
- Dropout
- Dense layer with Sigmoid activation

## Evaluation Metrics
The model is evaluated using:
- Accuracy
- Precision
- Recall
- Confusion Matrix
- Classification Report

## How to Run
1. Open the notebook using Google Colab.
2. Select a GPU runtime if available.
3. Run the notebook cells sequentially.
4. The dataset will be downloaded automatically through KaggleHub.
5. Train and evaluate the model.

## Repository Structure
```text
pneumonia-detection-chest-xray/
│
├── pneumonia_detection_demo.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

## Disclaimer
This project is an educational demonstration of deep learning in medical imaging. It is not a validated medical device and must not be used for real-world diagnosis.</markdown>
