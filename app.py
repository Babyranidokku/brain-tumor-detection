import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image

# Load trained model
model = tf.keras.models.load_model('brain_tumor_cnn.h5')

# Function to preprocess image
def preprocess_image(image, target_size=(150, 150)):
    image = image.resize(target_size)
    image = image.convert("RGB")
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array

# Sidebar for additional information
st.sidebar.title("🔬 Brain Tumor Detection")
st.sidebar.info("Upload an MRI scan to detect if a brain tumor is present. This app uses a deep learning model for predictions.")

# Main Title
st.title("🧠 Brain Tumor Detection")
st.write("Upload an MRI scan image to check for a brain tumor.")

# Upload File
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded MRI Scan", width=300)

    # Preprocess and predict
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)

    # Display Result
    if prediction[0] > 0.5:
        st.error("🚨 **Tumor Detected**")
    else:
        st.success("✅ **No Tumor Detected**")
