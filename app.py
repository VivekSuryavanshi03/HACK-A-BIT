import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from inference import prediction
import cv2
from pathlib import Path


# Load the pre-trained segmentation model
@st.cache(allow_output_mutation=True)
def load_segmentation_model():
    try:
        model_path = Path("C:/Users/vivek/Desktop/hack/ResUNet-segModel-weights.hdf5")
        model_seg = load_model(model_path, custom_objects={'focal_tversky': focal_tversky , 'tversky': tversky})
        return model_seg
    except Exception as e:
        st.error(f"Error loading segmentation model: {e}")
        return None

@st.cache(allow_output_mutation=True)
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
    model_seg = load_segmentation_model()
    model_classification = load_classfication_model()
    if model_seg is None or model_classification is None:
        return None
    
    output = prediction(image_array,model_classification,model_seg)
    return output # Placeholder implementation


# Main function to run the application
def main():
    st.title("Brain MRI Segmentation")
    st.write("Upload an MRI image and let our model segment the brain region for you!")

    uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg","tif", "png"])

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Image", width=300)

        if st.button("Segment"):
            with st.spinner("Segmenting..."):
                # prediction function ()
                image_array = np.array(image)

                # Perform segmentation - 
                output = segment_brain_mri(image_array)
                if output is None:
                    st.error("Segmentation failed.")
                    return
                
                segmented_image , mask = output.predicted_mask,output.has_mask
            st.success("Segmentation complete!")
            # Convert segmented_image to numpy array if it's a Pandas Series
            if isinstance(segmented_image, pd.Series):
                segmented_image = segmented_image.to_numpy()
                segmented_image=segmented_image.flatten()
                mask=segmented_image[0][0]
                image=np.array(image)
                img=cv2.resize(image,(256,256))
                mask3d=cv2.merge([mask*255,mask*0,mask*0]).astype(np.uint8)

                out=cv2.addWeighted(img,0.7,mask3d,0.7,0)

            # Display the segmented image
            st.image(out, caption="Segmented Image", width=300)
            if mask[0].any()==0:
                st.header("NO MASK :)")

# Run the application
if __name__ == "__main__":
    main()
