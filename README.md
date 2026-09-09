# google-landmark-recognition
Google Landmark Recognition using Deep Learning and MobileNetV2
# Google Landmark Recognition Using Deep Learning

## Project Overview

This project implements a Google Landmark Recognition system using Deep Learning and Transfer Learning techniques.

The system is designed to identify and classify landmarks from images using a pre-trained MobileNetV2 model.

A subset of the Google Landmark Recognition 2021 dataset is used for training and evaluation.

## Dataset

The project uses the Google Landmark Recognition 2021 dataset available through Kaggle.

The dataset contains:

- Training images
- Test images
- Landmark IDs
- Image metadata

For this project, the five landmark classes with the highest number of images were selected.

A maximum of 500 images from each landmark class was used.

Total images used:

- 2,500 images
- 2,000 training images
- 500 validation images

## Methodology

The following steps were used to build the project:

1. Load the Google Landmark Recognition dataset.
2. Analyze the landmark IDs and their image distribution.
3. Select the top five landmark classes.
4. Create a smaller dataset for efficient training.
5. Locate and load the corresponding images.
6. Split the dataset into training and validation sets.
7. Apply image preprocessing and data augmentation.
8. Build a deep learning model using MobileNetV2.
9. Train the model using transfer learning.
10. Evaluate the model using validation accuracy and loss.
11. Visualize training and validation performance.
12. Predict landmarks from new images.

## Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Kaggle Notebook

## Model Architecture

The project uses MobileNetV2 as the base model.

The architecture includes:

- MobileNetV2 pre-trained on ImageNet
- Global Average Pooling Layer
- Dense Layer with 256 neurons
- Dropout Layer
- Softmax Output Layer

Transfer learning is used to improve training efficiency.

## Data Preprocessing

The images are resized to:

224 x 224 pixels

The pixel values are normalized between 0 and 1.

Data augmentation techniques include:

- Rotation
- Width shifting
- Height shifting
- Zooming
- Horizontal flipping

## Model Training

The model is trained using the following configuration:

- Image Size: 224 x 224
- Batch Size: 32
- Epochs: 10
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy

The project also uses:

- Early Stopping
- Model Checkpoint

These techniques help prevent overfitting and save the best model.

## Results

The trained model achieved approximately:

- Validation Accuracy: 98.40%
- Validation Loss: 0.1103

The model was evaluated using validation data.

Performance visualization includes:

- Training Accuracy
- Validation Accuracy
- Training Loss
- Validation Loss
- Confusion Matrix
- Classification Report

## Project Structure

```text
google-landmark-recognition/
│
├── landmark_recognition.ipynb
├── best_landmark_model.keras
├── README.md
└── requirements.txt
