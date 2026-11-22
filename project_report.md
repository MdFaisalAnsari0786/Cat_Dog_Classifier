# CAT VS DOG IMAGE CLASSIFICATION USING DEEP LEARNING

**Semester Project – Unit IV: Deep Learning for Computer Vision**

**Submitted by:** __________________________  
**Roll Number:** __________________________  
**Course / Semester:** _____________________  
**Department / College:** __________________  
**Submitted to:** __________________________

---

## 1. Abstract

This project develops an image-classification system that identifies whether an input image contains a cat or a dog. The system uses a Convolutional Neural Network (CNN) approach with MobileNetV2 transfer learning. A pre-trained convolutional base extracts useful visual features, while a new binary-classification layer learns the cat-versus-dog task. Data augmentation is used to improve generalization. The trained model is evaluated on a separate test dataset and deployed through a simple Streamlit web interface.

## 2. Introduction

Computer vision enables computers to interpret information from images. Image classification is a common computer-vision task in which a model assigns an image to one of several classes. In this project, the two classes are cats and dogs. CNNs are suitable for image tasks because convolutional layers learn local visual patterns such as edges, textures and shapes.

## 3. Problem Statement

To design and implement a deep-learning application that accepts an image and predicts whether the image belongs to the Cat class or the Dog class.

## 4. Objectives

1. Understand the working of convolutional neural networks.
2. Prepare and split an image dataset into training, validation and testing sets.
3. Apply data augmentation to improve model generalization.
4. Use transfer learning with a pre-trained MobileNetV2 network.
5. Evaluate the model using test data.
6. Build a simple interface for image upload and prediction.

## 5. Dataset

The project uses the Cats vs Dogs dataset distributed through a Microsoft-hosted archive and documented by TensorFlow Datasets. The catalogue reports 23,262 usable examples after 1,738 corrupted images are discarded. The project script checks image files and creates an approximate 80% training, 10% validation and 10% test split. The two class folders are `cats` and `dogs`.

## 6. Software and Hardware Requirements

**Software:** Python 3.10–3.12, TensorFlow/Keras, NumPy, Pillow, Matplotlib, Streamlit and Visual Studio Code (optional).

**Hardware:** A 64-bit computer with internet access and sufficient free storage. A GPU is helpful but not compulsory; CPU training may take longer.

## 7. Methodology

### 7.1 Data Preparation

The dataset archive is downloaded and extracted. Images that cannot be opened are excluded. Valid images are shuffled using a fixed random seed and copied into training, validation and testing folders.

### 7.2 Image Preprocessing

Each input image is resized to 160 × 160 pixels and represented as a three-channel RGB image. Pixel values are rescaled to the range expected by MobileNetV2.

### 7.3 Data Augmentation

Random horizontal flipping, small rotations and zoom transformations are applied during training. These transformations create varied training examples and help reduce overfitting.

### 7.4 CNN and Transfer Learning

MobileNetV2 pre-trained on ImageNet is used as the convolutional feature extractor. Its original classification head is removed. The convolutional base is initially frozen, and a new head consisting of Global Average Pooling, Dropout and a sigmoid Dense output is trained for binary classification.

### 7.5 Fine-Tuning

As an optional second stage, the last 30 layers of the convolutional base are unfrozen and trained with a small learning rate. Batch Normalization layers are kept frozen to support stable fine-tuning.

### 7.6 Prediction

The trained model outputs a probability for the dog class. A score of 0.5 or higher is classified as Dog; otherwise it is classified as Cat.

## 8. System Flow

1. Start the application.
2. Upload a JPG or PNG image.
3. Convert the image to RGB and resize it to 160 × 160.
4. Pass the image to the trained CNN model.
5. Obtain the prediction score.
6. Display Cat or Dog and the model confidence.

## 9. Evaluation and Results

Run the training program and record the values produced on your own computer.

- Test loss: ____________________
- Test accuracy: _________________
- Training epochs: _______________
- Fine-tuning used (Yes/No): ______

**Insert screenshot of the training accuracy graph:** `outputs/training_accuracy.png`  
**Insert screenshot of the Streamlit prediction screen:** ____________________

Do not fill in an estimated result. Use the actual values in `outputs/metrics.json` after running the project.

## 10. Advantages

- Automatically learns visual features from images.
- Transfer learning reduces the amount of training required compared with training a large CNN completely from scratch.
- Data augmentation helps the model handle image variations.
- The Streamlit interface makes the model easy to demonstrate.

## 11. Limitations

- The prediction may be incorrect for blurred, very small, unusual or mixed-subject images.
- The model only predicts two classes: cat and dog.
- Performance depends on dataset quality and the training environment.
- Confidence is a model score and should not be treated as a guarantee.

## 12. Future Scope

The project can be extended to classify more animal categories, use a larger dataset, compare multiple CNN architectures, deploy the application online, or add explainability visualizations such as Grad-CAM heatmaps.

## 13. Conclusion

The project demonstrates how CNN-based transfer learning can be used for binary image classification. It covers dataset preparation, preprocessing, data augmentation, feature extraction, optional fine-tuning, evaluation and application deployment. It provides a practical example of concepts studied in Deep Learning for Computer Vision.

## 14. References

1. TensorFlow Datasets, “Cats vs Dogs”: https://www.tensorflow.org/datasets/catalog/cats_vs_dogs
2. TensorFlow, “Transfer learning and fine-tuning”: https://www.tensorflow.org/tutorials/images/transfer_learning
3. TensorFlow, “Install TensorFlow”: https://www.tensorflow.org/install/
