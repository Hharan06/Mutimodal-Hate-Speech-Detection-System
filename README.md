# Multimodal Hate Speech Detection System

This project implements a multimodal hate speech detection system capable of classifying both text comments and memes (images) to identify hate speech. The system distinguishes between targeted and untargeted hate speech and can identify whether it is directed at individuals or groups. For images, it categorizes content into specific classes such as blood and gore, NSFW (Not Safe for Work), or smoking.

## Table of Contents

1. [Introduction](#introduction)
2. [Requirements](#requirements)
3. [System Architecture](#system-architecture)
4. [Data Collection](#data-collection)
5. [Model Training](#model-training)
6. [Implementation](#implementation)
7. [Usage](#usage)
8. [Evaluation Metrics](#evaluation-metrics)
9. [Future Work](#future-work)
10. [License](#license)

## Introduction

The rise of online platforms has made it increasingly important to monitor and control hate speech, which can harm individuals and communities. This system combines text and image analysis to provide a comprehensive approach to detecting hate speech in various formats.

## Requirements

### Software
- Python 3.x
- Libraries:
  - TensorFlow or PyTorch
  - NLTK or spaCy (for text processing)
  - OpenCV or PIL (for image processing)
  - scikit-learn (for machine learning utilities)
  - Matplotlib (for data visualization)

### Hardware
- A computer with a capable GPU (recommended for training deep learning models)
- Sufficient storage for datasets and models

## System Architecture

The system consists of two primary modules:

1. **Text Classification Module**: Analyzes comments to classify them as:
   - Targeted or untargeted
   - Directed at individuals or groups

2. **Image Classification Module**: Processes memes and classifies them into:
   - Blood and gore
   - NSFW
   - Smoking

Both modules feed into a central decision system that combines the results for final hate speech detection.

## Data Collection

### Text Data
- Use publicly available datasets containing labeled comments, such as:
  - Twitter datasets on hate speech
  - Online forums or social media platforms (ensure compliance with their terms)
  - Used dataset link : https://github.com/idontflow/OLID/blob/master/own/cleaned_train_data_v0.csv

### Image Data
- Collect memes from social media, ensuring to label images based on the categories mentioned above.
- Use existing datasets of hateful images where available.

### Preprocessing
- Clean and tokenize text data.
- Resize and normalize images for consistent input to models.

## Model Training

1. **Text Classification**:
   - Utilize models such as BERT, LSTM, or CNNs for text analysis.
   - Split data into training, validation, and test sets.

2. **Image Classification**:
   - Implement CNN architectures like ResNet, VGG, or MobileNet for image analysis.
   - Augment the dataset to improve model robustness.

3. **Training Process**:
   - Use cross-validation to tune hyperparameters.
   - Monitor training with validation data to prevent overfitting.

## Implementation

1. **Setup**:
   - Clone the repository and install required libraries.
   - Place datasets in the designated directories.

2. **Training**:
   - Run training scripts for both text and image models.
   - Save the trained models for inference.

3. **Inference**:
   - Create a function that takes text and image inputs, preprocesses them, and runs them through their respective models.
   - Combine predictions to determine the final hate speech classification.


## Usage

1. Run the script to classify comments and memes.
2. Provide a text comment and the path to the meme image as input.
3. The output will indicate whether hate speech is detected and the category of the content.

## Evaluation Metrics

To assess the performance of the models, use the following metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Example Evaluation

After training, evaluate the model using a held-out test set to measure its effectiveness in detecting hate speech.

## Future Work

- Expand the dataset to improve model generalization.
- Implement real-time monitoring of social media platforms.
- Enhance the model to detect more nuanced forms of hate speech.
- Explore adversarial training to make models more robust against manipulation.

## License

This project is open-source and available for modification and use. If you adapt this work, please attribute the original source.
