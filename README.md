# Pneumonia Detection from Chest X-ray Images

## Project Overview

This project demonstrates a deep learning approach for detecting pneumonia from chest X-ray images using Transfer Learning.

The model uses **MobileNetV2**, a pre-trained Convolutional Neural Network (CNN), to classify chest X-ray images into two categories:

- **NORMAL**
- **PNEUMONIA**

The project covers the complete machine learning workflow, from dataset exploration and preprocessing to model training, evaluation, and prediction.

> **Disclaimer:** This is an educational project and is not intended for clinical diagnosis or real-world medical decision-making.

## Dataset

The project uses the [Chest X-ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) from Kaggle.

The dataset contains two classes:
- NORMAL
- PNEUMONIA

### Dataset Distribution

The dataset was explored to understand the distribution of images across the training and testing sets.

| Dataset Split | NORMAL | PNEUMONIA | Total |
|---|---:|---:|---:|
| Training | 1,341 | 3,875 | 5,216 |
| Testing | 234 | 390 | 624 |
| **Total** | **1,575** | **4,265** | **5,840** |

### Key Data Observations

- The dataset is imbalanced, with a noticeably higher number of PNEUMONIA images compared to NORMAL images.
- In the training set, pneumonia images represent approximately 74.3% of the data, while normal images represent 25.7%.
- This imbalance may influence the model's predictions and contribute to differences in performance between the two classes.
- Therefore, accuracy alone is not sufficient to evaluate the model.

The dataset is downloaded automatically using `kagglehub` within the notebook.

## Technologies Used

- Python
- TensorFlow & Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn
- KaggleHub

## Project Workflow

1. **Dataset Download & Exploration:** Downloading the dataset and examining class distribution.
2. **Image Loading & Preprocessing:** Loading and resizing images to 160 × 160 pixels.
3. **Model Building:** Applying Transfer Learning using MobileNetV2.
4. **Model Training:** Training the classification head while keeping the pre-trained base frozen.
5. **Model Evaluation:** Evaluating performance using multiple classification metrics.
6. **Predictions:** Testing the model on unseen chest X-ray images.

## Model Architecture

MobileNetV2 is used as a pre-trained feature extractor with its original layers frozen.

A custom classification head is added, consisting of:

- Global Average Pooling
- Dropout
- Dense output layer with Sigmoid activation

The model is trained for 3 epochs using binary classification.

## Model Training Results

The training and validation curves show:

- Increasing accuracy across the three epochs.
- Decreasing training and validation loss.
- Training and validation performance improving consistently.

At the end of training:
- Training Accuracy: approximately 94.1%
- Validation Accuracy: approximately 94.5%

These results indicate that the model learned useful patterns during training. However, performance on the independent test set provides a more important assessment of generalization.

## Model Evaluation

The model was evaluated on 624 unseen test images.

### Classification Report

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| NORMAL | 0.93 | 0.55 | 0.69 | 234 |
| PNEUMONIA | 0.78 | 0.97 | 0.87 | 390 |
| **Accuracy** | | | **0.81** | **624** |

### Key Performance Observations

- **Overall Accuracy: 81%**
- **PNEUMONIA Recall: 97%** — The model correctly identifies a high proportion of actual pneumonia cases.
- **NORMAL Recall: 55%** — The model misses a considerable proportion of actual normal cases by classifying them as pneumonia.
- **PNEUMONIA Precision: 78%** — Some images predicted as pneumonia actually belong to the normal class.

### What Do These Results Tell Us?

Although the model achieves 81% test accuracy, its performance differs significantly between the two classes.

The high pneumonia recall indicates that the model detects most pneumonia cases in this test set. However, the relatively low normal recall suggests a tendency to predict pneumonia more frequently.

This behavior may be influenced by the class imbalance in the dataset.

Therefore, evaluating medical image classification models requires examining class-specific precision, recall, and F1-score rather than relying on accuracy alone.

## Limitations

Some limitations identified during this experiment include:

1. **Class Imbalance:** The dataset contains considerably more pneumonia images than normal images.
2. **Limited Training:** The model was trained for only three epochs as part of a short educational demonstration.
3. **Performance Difference Between Classes:** The model performs better at identifying pneumonia than normal cases.
4. **Generalization:** Evaluation was performed on the provided test set, without independent external validation.
5. **Model Explainability:** The current implementation does not include visual explanations of the regions influencing predictions.

## Future Improvements

Several improvements could be explored to enhance model performance:

1. **Address Class Imbalance:**
   - Apply class weighting during training.
   - Experiment with oversampling the minority class.
   - Explore balanced data augmentation.

2. **Improve Model Training:**
   - Increase the number of training epochs.
   - Use Early Stopping to prevent overfitting.
   - Experiment with different learning rates and optimizers.

3. **Fine-tuning:**
   - Unfreeze selected MobileNetV2 layers.
   - Fine-tune the pre-trained model using a lower learning rate.

4. **Experiment with Other Architectures:**
   - Compare MobileNetV2 with other CNN architectures such as ResNet50 and EfficientNet.

5. **Model Explainability:**
   - Apply Grad-CAM to visualize image regions contributing to the model's predictions.

6. **Further Validation:**
   - Evaluate the model on independent datasets.
   - Investigate potential dataset bias and differences in image acquisition.

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

This project is an educational demonstration of deep learning applications in medical imaging. It is not a validated medical device and must not be used for clinical diagnosis or treatment decisions.
