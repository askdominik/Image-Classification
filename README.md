# CIFAR-10 Image Classification

This repository contains a simple Convolutional Neural Network (CNN) for classifying images from the CIFAR-10 dataset. The project is divided into two parts:
1. Training the model using the CIFAR-10 dataset.
2. Using the trained model to make predictions on new images.

## Requirements

- Python 3.6 or higher
- TensorFlow
- Keras
- OpenCV
- Matplotlib
- NumPy

You can install the required packages using `pip`:

```bash
pip3 install tensorflow keras opencv-python matplotlib numpy
```
## Training the Model

To train the model, run the cifar10_model.py script. This script will:

1. Load and preprocess the CIFAR-10 dataset.
2. Define and train a CNN model.
3. Save the trained model to a file named image_classifier.keras.
```bash
python3 cifar10_model.py
```
## Making Predictions

To use the trained model to make predictions on new images, run the predict.py script. Ensure that you have an image (e.g., images/plane.jpg) to test.

1. Update the image path in the predict.py script if needed.
2. Run the script to make a prediction.
```bash
python3 predict.py
```
