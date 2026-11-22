"""Streamlit interface for the trained Cat vs Dog classifier."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cat_dog_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"
IMG_SIZE = (160, 160)

st.set_page_config(page_title="Cat vs Dog Classifier", page_icon="🐾", layout="centered")
st.title("🐾 Cat vs Dog Image Classifier")
st.write("Upload a photo and the CNN model will predict whether it contains a cat or a dog.")
st.caption("Deep Learning for Computer Vision | CNN + MobileNetV2 Transfer Learning")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def predict_image(image: Image.Image) -> tuple[str, float]:
    image = image.convert("RGB").resize(IMG_SIZE)
    array = np.asarray(image, dtype=np.float32)
    probability_dog = float(load_model().predict(np.expand_dims(array, axis=0), verbose=0)[0][0])
    if CLASS_NAMES[1] == "dogs":
        if probability_dog >= 0.5:
            return "Dog", probability_dog
        return "Cat", 1.0 - probability_dog
    raise RuntimeError("Unexpected class order in class_names.json")


if not MODEL_PATH.exists() or not CLASS_PATH.exists():
    st.warning("The trained model is not available yet.")
    st.markdown("First follow the setup steps in **README.md**, then run:")
    st.code("python download_data.py\npython train.py\nstreamlit run app.py", language="bash")
    st.stop()

with open(CLASS_PATH, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)

uploaded_file = st.file_uploader("Choose a cat or dog image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    if st.button("Predict image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image..."):
            label, confidence = predict_image(image)
        st.success(f"Prediction: **{label}**")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption("Confidence is the model's score, not a guarantee. Results can be wrong for unusual or unclear images.")
