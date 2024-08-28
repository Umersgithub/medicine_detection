# Medicine Detection and Classification System

## Overview

This project consists of two Python scripts: `med-train.py` and `med-detect.py`. The purpose is to train a model to classify different medicines from images and then use this model to detect and announce the name of a medicine in real-time using a webcam.

## Files

`med-train.py` trains a deep learning model to classify medicines based on images. It saves the trained model as `medicine_model.h5` and the class names (labels) in `class_names.txt`. `med-detect.py` loads the trained model and class names, captures video from the webcam, detects and classifies the medicine in real-time, displays the predicted class and confidence on the video frame, and speaks out the name of the detected medicine if the confidence level is high.

## Prerequisites

Ensure that you have the necessary Python packages installed. You can install them using the `requirements.txt` file:


`pip install -r requirements.txt`


To train the model using your dataset, run: 
`python med-train.py`

 This will load the images from the med directory, train a model, and save the trained model as medicine_model.h5 along with the class names in class_names.txt. 
 
 Once the model is trained, start the real-time medicine detection by running: 
 `python med-detect.py` 
 
 This script will load the trained model and class names, access your webcam, start the detection process, display the predicted medicine name and confidence level on the video feed, and announce the detected medicine name aloud if the confidence is above the threshold.





