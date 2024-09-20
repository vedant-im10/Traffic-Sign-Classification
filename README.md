# 🚦 Traffic Sign Classification: A Deep Learning Approach

**_"Using Convolutional Neural Networks to Classify Traffic Signs"_**

This project implements a deep learning model for classifying traffic signs using images. The system is designed to assist in autonomous driving technologies, helping cars recognize and respond to different traffic signs effectively. By utilizing **Convolutional Neural Networks (CNNs)**, the model achieves high accuracy in predicting traffic sign categories from a large dataset of traffic signs.

## 🌟 Project Motivation

In the age of autonomous vehicles, traffic sign recognition is critical for ensuring road safety. Automated systems need to reliably identify and interpret traffic signs in real-time. This project aims to develop a robust machine learning model that can classify traffic signs with high accuracy, supporting the advancement of autonomous driving and improving safety on the roads.

## 🔍 Key Features

- **🧠 Deep Learning Model**: Utilizes **Convolutional Neural Networks (CNNs)** for traffic sign classification, enabling the model to learn complex features from traffic sign images.
- **📈 Data Augmentation**: Applied various image augmentation techniques to increase the size of the training data and improve model generalization.
- **🚦 Real-Time Traffic Sign Detection**: Once trained, the model can be integrated into real-time systems for traffic sign detection in self-driving cars.
- **🗂️ Large Dataset**: The project leverages a large dataset of traffic sign images to ensure the model is capable of recognizing signs across different lighting conditions, angles, and environments.

## 🧬 Methodology

1. **Data Collection**: The dataset used contains traffic sign images from a variety of countries, providing a diverse set of traffic signs.
   
2. **Data Preprocessing**:
   - Loaded the dataset from CSV files (`Train.csv`, `Test.csv`, and `Meta.csv`).
   - Preprocessed the images by resizing, normalizing, and augmenting to improve model performance.
   - Augmentation techniques included random rotations, shifts, flips, and brightness changes to simulate real-world conditions.

3. **Model Architecture**:
   - Built a **Convolutional Neural Network (CNN)** with multiple layers of convolution, max-pooling, and dense layers.
   - The model is designed to automatically extract features from images and classify them into the appropriate traffic sign category.
   
4. **Model Training**:
   - Trained the model using the preprocessed dataset and evaluated its performance on a test set.
   - Utilized the **Adam optimizer** with categorical cross-entropy loss to improve learning efficiency.
   - Saved the trained model as `my_model.h5` and `traffic_classifier.h5`.

5. **Evaluation**:
   - Evaluated the model's performance using accuracy, precision, recall, and F1-score metrics.
   - The model was fine-tuned using hyperparameter optimization to achieve maximum accuracy.

6. **Deployment**:
   - Implemented the trained model in Python for real-time traffic sign recognition using the file `traffic_sign.py`.
   - Developed a simple graphical user interface (GUI) in `gui.py` to allow users to upload images and get predictions on traffic sign categories.

## 📊 Key Insights

- **High Accuracy**: The CNN model achieved a classification accuracy of over 90%, demonstrating its effectiveness in recognizing traffic signs.
- **Robust to Variations**: The model performed well on images with varying brightness, angles, and partial occlusions, making it suitable for real-world deployment in autonomous systems.
- **Scalable**: The architecture is scalable and can be easily extended to classify more complex image categories beyond traffic signs.

## 🔬 Outcomes and Impact

- **Enhanced Road Safety**: This project can be integrated into self-driving car systems to improve road safety by accurately detecting and interpreting traffic signs.
- **Scalability**: The model can be further trained on larger datasets to accommodate more traffic signs from different countries, improving its robustness.
- **Real-World Application**: The trained model can be deployed in autonomous vehicle systems or integrated into driver-assistance technologies.

## 🚀 Future Directions

- **Advanced Models**: Explore the use of advanced deep learning architectures like **ResNet** and **Inception** to further improve accuracy.
- **Real-Time Processing**: Integrate the model with a real-time video feed for live traffic sign detection.
- **Dataset Expansion**: Expand the dataset to include traffic signs from more countries and add signs from various weather conditions for a more robust model.
