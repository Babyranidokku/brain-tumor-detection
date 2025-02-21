import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import streamlit as st
import gdown  # For downloading model from Google Drive

# Model Path
MODEL_PATH = "brain_tumor_cnn.keras"

# Google Drive Direct Link
GOOGLE_DRIVE_ID = "1Q-MsGDLj8tlJg_7Vo7S09SzFqLl-APXa"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_ID}"

# Function to download model if not present
def download_model():
    """Downloads the model if it's not found in the directory."""
    if not os.path.exists(MODEL_PATH):  
        st.info("🔄 Downloading model... Please wait.")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Model download failed: {e}")
            return False
    return True

# Ensure model is downloaded before loading
if download_model():
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model Loaded Successfully!")
    except Exception as e:
        st.error(f"🚨 Error loading model: {e}")
        model = None  # Set model to None if loading fails

# Image Preprocessing Function
def preprocess_image(image, target_size=(150, 150)):
    """Prepares the image for model prediction."""
    image = image.resize(target_size)
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Streamlit UI
st.sidebar.title("🔬 Brain Tumor Detection")
st.sidebar.info("Upload an MRI scan to detect if a brain tumor is present.")

st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan image to check for a brain tumor.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    if model is None:
        st.error("🚨 Model could not be loaded. Please check logs!")
    else:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded MRI Scan", width=300)

        # Process & Predict
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        # Display Result
        if prediction[0] > 0.5:
            st.error("🚨 **Tumor Detected**")
        else:
            st.success("✅ **No Tumor Detected**")

# Debugging - Check if model exists in Streamlit Cloud
st.write("🛠 Model file exists:", os.path.exists(MODEL_PATH))
