import streamlit as st
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

# Define the directory where the model file is located
model_directory = Path("C:/Users/vivek/Desktop/hack")

# Define the filename of the segmentation model
model_filename = "ResUNet-segModel-weights.hdf5"

# Construct the full path to the model file
model_path = model_directory / model_filename

# Load the segmentation model using the dynamic file path
@st.cache_data
def load_segmentation_model(model_path):
    try:
        model_seg = load_model(model_path, custom_objects={'focal_tversky': focal_tversky , 'tversky': tversky})
        return model_seg
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None

@st.cache_data
def load_classfication_model():
    try:
        model_path = Path("C:/Users/vivek/Desktop/hack/clf-densenet-weights.hdf5")
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return None

# Function to perform segmentation
def segment_brain_mri(image_array):
    model_seg = load_segmentation_model(model_path)
    model_classification = load_classfication_model()
    if model_seg is None or model_classification is None:
        return None
    
    output = prediction(image_array, model_classification, model_seg)
    return output # Placeholder implementation

# Main function to run the application
def main():
    st.title("Brain MRI Segmentation")
    st.write("Upload an MRI image and let our model segment the brain region for you!")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "tif", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", width=300)

        if st.button("Segment"):
            with st.spinner("Segmenting..."):
                # prediction function ()
                image_array = np.array(image)

                # Perform segmentation
                output = segment_brain_mri(image_array)
                if output is None:
                    st.error("Segmentation failed.")
                    return
                
                segmented_image, mask = output.predicted_mask, output.has_mask
            st.success("Segmentation complete!")
            # Convert segmented_image to numpy array if it's a Pandas Series
            if isinstance(segmented_image, pd.Series):
                segmented_image = segmented_image.to_numpy()
                segmented_image = segmented_image.flatten()
                mask = segmented_image[0][0]
                image = np.array(image)
                img = cv2.resize(image, (256, 256))
                mask3d = cv2.merge([mask * 255, mask * 0, mask * 0]).astype(np.uint8)

                out = cv2.addWeighted(img, 0.7, mask3d, 0.7, 0)

            # Display the segmented image
            st.image(out, caption="Segmented Image", width=300)
            if mask[0].any() == 0:
                st.header("NO MASK :)")

# Run the application
if __name__ == "__main__":
    main()
