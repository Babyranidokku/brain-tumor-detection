import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set dataset directory
dataset_dir = r"C:\Users\babyr\Braintumordetection\archive (1)"

# Image parameters
IMG_SIZE = (150, 150)  
BATCH_SIZE = 32

# Data Augmentation & Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Only normalization for test set

# Load Data from directories
train_generator = train_datagen.flow_from_directory(
    dataset_dir,  
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",  
    classes=["train_yes", "train_no"]  # Specify tumor vs. no tumor
)

test_generator = test_datagen.flow_from_directory(
    dataset_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    classes=["test_yes", "test_no"]
)

# Build CNN Model
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D(2, 2),

    layers.Flatten(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")  # Binary classification
])

# Compile Model
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy"]
)

# Train Model
EPOCHS = 20
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS
)

# Save Model
model.save("brain_tumor_cnn.h5")

print("Model training complete and saved as 'brain_tumor_cnn.h5'!")
