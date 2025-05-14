import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# === Load Model and Class Indices ===
model = tf.keras.models.load_model("plant_disease_prediction_model.h5")
class_indices = json.load(open("class_indices.json"))

# === Preprocessing Function ===
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# === Prediction Function ===
def predict_image_class(model, image_file, class_indices):
    preprocessed_img = load_and_preprocess_image(image_file)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# === Streamlit App ===
st.title("🌿 Plant Disease Classifier")

uploaded_image = st.file_uploader("Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image).convert("RGB")
    col1, col2 = st.columns(2)

    with col1:
        st.image(image.resize((150, 150)), caption="Uploaded Image")

    with col2:
        if st.button("Classify"):
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: **{prediction}**")
