import tensorflow as tf

# Load your model in the correct format
model = tf.keras.models.load_model("brain_tumor_cnn.keras", compile=False)

# Re-save it properly
model.save("brain_tumor_cnn.keras")
print("✅ Model re-saved successfully!")
model.summary()
