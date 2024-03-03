import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from inference import prediction
import cv2
from pathlib import Path
import pandas as pd

# Define custom loss functions
def focal_tversky(y_true, y_pred, alpha=0.7, beta=0.3, gamma=0.75):
    """Focal Tversky loss function."""
    smooth = 1e-5
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    focal_tversky = tf.keras.backend.pow((1 - tversky), gamma)
    return focal_tversky

def tversky(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-5):
    """Tversky metric function."""
    y_true_pos = tf.keras.backend.flatten(y_true)
    y_pred_pos = tf.keras.backend.flatten(y_pred)
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos)
    false_neg = tf.reduce_sum(y_true_pos * (1 - y_pred_pos))
    false_pos = tf.reduce_sum((1 - y_true_pos) * y_pred_pos)
    tversky = (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)
    return tversky

def load_segmentation_model(model_path):
    model_seg = load_model(model_path, custom_objects={'focal_tversky': focal_tversky , 'tversky': tversky})
    return model_seg

# Define the directory where the model file is located
model_directory = Path("C:/Users/vivek/Desktop/hack")

# Define the filename of the segmentation model
model_filename = "ResUNet-segModel-weights.hdf5"

# Construct the full path to the model file
model_path = model_directory / model_filename

# Load the segmentation model using the dynamic file path
model_seg = load_segmentation_model(model_path)

def load_classfication_model():
    try:
        model_path = Path("C:/Users/vivek/Desktop/hack/clf-densenet-weights.hdf5")
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None

# Function to perform segmentation
def segment_brain_mri(image_array):
    model_seg = load_segmentation_model()
    model_classification = load_classfication_model()
    if model_seg is None or model_classification is None:
        return None
    
    output = prediction(image_array, model_classification, model_seg)
    return output # Placeholder implementation

# Main function to run the application
def main():
    print("Brain MRI Segmentation")
    print("Upload an MRI image and let our model segment the brain region for you!")

    # Placeholder for uploaded image
    uploaded_file = None  # Assume no file uploaded

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)

        print("Uploaded MRI Image")

        if True:
            print("Segmenting...")
            # Placeholder for segmentation logic
            image_array = np.array(image)

            # Perform segmentation
            output = segment_brain_mri(image_array)
            if output is None:
                print("Segmentation failed.")
                return
            
            # Placeholder for displaying segmented image
            print("Segmentation complete!")

# Run the application
if __name__ == "__main__":
    main()
