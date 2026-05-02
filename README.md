# Astronomy Image Classifier

This project is a machine learning image classification project. The goal is to classify images of astronomical objects such as planets, the Moon, and asteroids using different machine learning models and compare each model's effectiveness

## Project Overview

The project compares multiple models for image classification:

1. Logistic Regression
2. A basic Neural Network
3. A Convolutional Neural Network

The images are organized into training, validation, and testing folders. Each folder contains subfolders for the 12 different object classes: asteroid, earth, galaxy, jupiter, mars, mercury, moon, neptune, pluto, saturn, uranus, venus.

## Dataset
The dataset was too large for GitHub but can be found below. The following two datasets from kaggle where merged and used for this project.

[astrophysical-objects-image-dataset](https://www.kaggle.com/datasets/engeddy/astrophysical-objects-image-dataset)

[sun-and-moon-images](kaggle.com/datasets/khushipitroda/sun-and-moon-images)

## Models 

### Logistic Regression

Logistic Regression is used as a baseline model. The image pixels are flattened into a one dimensional array before being passed into the model. The baseline accuracy was 0.800. 

### Neural Network

A simple neural network is used to improve on the baseline model. It uses dense layers and a softmax output layer to classify the images into different classes.

### Convolutional Neural Network

A CNN is used as the final model. It uses convolution layers to learn visual patterns from the images, pooling layers to reduce the size of feature maps, and dense layers to make the final classification. The final test accuracy of this model was 0.941
