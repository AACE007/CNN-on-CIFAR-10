# CNN-on-CIFAR-10

Dataset
The CIFAR dataset consists of 60,000 32x32 color images categorized into 10 distinct classes. Each class contains 6,000 images, and the dataset is widely used to evaluate and improve image classification models.


 classes in the dataset along with their respective labels:

Airplane: 0
Automobile: 1
Bird: 2
Cat: 3
Deer: 4
Dog: 5
Frog: 6
Horse: 7
Ship: 8
Truck: 9


## Object Recognition using ResNet50

A deep learning project implementing object recognition on the CIFAR-10 dataset using a Convolutional Neural Network (CNN) with a ResNet50 architecture.

Overview
This project demonstrates how to build an image classifier using the ResNet50 pre-trained model. It's trained on the CIFAR-10 dataset, which consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The goal is to accurately classify these images into their respective categories.

Key Features
ResNet50 Architecture: Leverages the power of ResNet50, a deep residual learning framework, for robust feature extraction.

CIFAR-10 Dataset: Utilizes the popular CIFAR-10 dataset for image classification.

Image Preprocessing: Includes image scaling and other preprocessing steps for optimal model performance.

Transfer Learning: Employs transfer learning by utilizing pre-trained weights of ResNet50.

Fine-Tuning: Fine-tunes the pre-trained model to adapt it to the specific characteristics of the CIFAR-10 dataset.

High Accuracy: Achieves high accuracy in image classification.





## 1. Data Exploration & Preparation
- Loaded the CIFAR-10 dataset containing 50,000 32x32 color images across 10 object categories
- Verified that the data is balanced (5,000 images per class)
- Created a dictionary to map text labels to numerical values (label encoding)
- Loaded image files from the training folder and converted them to NumPy arrays

## 2. Data Preprocessing
- Converted all images and labels to NumPy arrays for faster processing
- Split the dataset into training (80%) and testing (20%) sets using train_test_split
- Normalized pixel values by dividing by 255 to scale them between 0-1

## 3. Transfer Learning Implementation
- Used pre-trained ResNet50 architecture with ImageNet weights as the base model
- Removed the top classification layers from ResNet50 to use only the feature extraction part
- The base model expects 256×256 images, but CIFAR-10 has 32×32 images

## 4. Model Architecture Design
- Added UpSampling2D layers to resize the 32×32 images to 256×256 (required by ResNet50)
- Connected the ResNet50 convolutional base to extract features
- Added a Flatten layer to convert 2D feature maps to 1D feature vectors
- Implemented BatchNormalization layers to stabilize training
- Built a custom classifier with:
  * Dense layer with 128 neurons and ReLU activation
  * Dropout (0.5) to prevent overfitting
  * Dense layer with 64 neurons and ReLU activation
  * Final Dense layer with 10 neurons (one per class) and softmax activation

## 5. Model Training
- Compiled the model with:
  * RMSprop optimizer with a low learning rate (2e-5)
  * Sparse categorical cross-entropy loss function
  * Accuracy as the evaluation metric
- Trained for 10 epochs with 10% validation split
- Monitored both training and validation metrics to avoid overfitting

## 6. Performance Evaluation
- Evaluated model on the test set, achieving 93.3% accuracy
- Created visualizations to compare:
  * Training vs. validation loss
  * Training vs. validation accuracy
- These plots help verify the model isn't overfitting

The project successfully demonstrates practical implementation of transfer learning by adapting a powerful pre-trained architecture (ResNet50) to a smaller dataset, achieving high accuracy with relatively minimal training.

