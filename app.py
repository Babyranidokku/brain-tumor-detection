import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import requests
import os

# Updated file name
MODEL_PATH = "brain_tumor_cnn.keras"

# OneDrive Direct Link or Google Drive Link (Use OneDrive or Google Drive)
MODEL_URL = f"https://drive.google.com/uc?export=download&id=1Q-MsGDLj8tlJg_7Vo7S09SzFqLl-APXa"
 # Change to your link

# Function to download model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):  
        st.info("🔄 Downloading the model... Please wait.")
        response = requests.get(MODEL_URL, allow_redirects=True)
        if response.status_code == 200:
            with open(MODEL_PATH, "wb") as model_file:
                model_file.write(response.content)
            st.success("✅ Model downloaded successfully!")
        else:
            st.error("❌ Failed to download model. Check your link!")


# Download & Load Model
download_model()
model = tf.keras.models.load_model(MODEL_PATH, compile=False)  # ✅ Fix file loading issue

# Image Preprocessing Function
def preprocess_image(image, target_size=(150, 150)):
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
