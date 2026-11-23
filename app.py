
"""Cat vs Dog Image Classifier with Invalid Image Detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# Project paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "cat_dog_model.keras"
CLASS_PATH = ROOT / "models" / "class_names.json"

IMG_SIZE = (160, 160)
VALIDATOR_SIZE = (224, 224)

# ImageNet class indices:
# Dog classes: 151 to 268
# Cat classes: 281 to 285
CAT_DOG_INDICES = set(range(151, 269)) | set(range(281, 286))

# Rejection threshold
MIN_CAT_DOG_SCORE = 0.25

# Streamlit page
st.set_page_config(
    page_title="Cat vs Dog Image Classifier",
    page_icon="🐾",
    layout="centered"
)

st.title("🐾 Cat vs Dog Image Classifier")

st.write(
    "Upload a Cat or Dog image. "
    "Other images will be rejected as invalid."
)

st.caption(
    "Deep Learning for Computer Vision | "
    "CNN + MobileNetV2 Transfer Learning"
)


# Load trained Cat/Dog model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


# Load pretrained ImageNet validation model
@st.cache_resource
def load_validator():
    return tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=True,
        weights="imagenet"
    )


# Check whether uploaded image contains a cat or dog
def is_cat_or_dog_image(image: Image.Image):

    validator = load_validator()

    image = image.convert("RGB").resize(VALIDATOR_SIZE)

    array = np.asarray(image, dtype=np.float32)

    batch = np.expand_dims(array, axis=0)

    batch = tf.keras.applications.mobilenet_v2.preprocess_input(
        batch
    )

    predictions = validator.predict(batch, verbose=0)[0]

    # Total probability assigned to all Cat/Dog ImageNet classes
    cat_dog_score = float(
        np.sum(predictions[list(CAT_DOG_INDICES)])
    )

    # Check whether a Cat/Dog class appears among top 5 predictions
    top5_indices = np.argsort(predictions)[-5:]

    top5_has_cat_dog = any(
        int(index) in CAT_DOG_INDICES
        for index in top5_indices
    )

    is_valid = (
        cat_dog_score >= MIN_CAT_DOG_SCORE
        and top5_has_cat_dog
    )

    return is_valid, cat_dog_score


# Predict Cat or Dog
def predict_image(image: Image.Image):

    # First validate the uploaded image
    is_valid, validation_score = is_cat_or_dog_image(image)

    if not is_valid:
        return "Not Classified", 0.0, validation_score

    # Prepare image for the trained project model
    image = image.convert("RGB").resize(IMG_SIZE)

    array = np.asarray(image, dtype=np.float32)

    batch = np.expand_dims(array, axis=0)

    # Predict Dog probability
    probability_dog = float(
        load_model().predict(batch, verbose=0)[0][0]
    )

    # Class order: cats, dogs
    if CLASS_NAMES != ["cats", "dogs"]:
        raise RuntimeError("Invalid class names in class_names.json")

    if probability_dog >= 0.5:
        return "Dog", probability_dog, validation_score

    return "Cat", 1.0 - probability_dog, validation_score


# Check trained model availability
if not MODEL_PATH.exists() or not CLASS_PATH.exists():

    st.warning("Trained model is not available.")

    st.code(
        "python download_data.py\n"
        "python train.py --batch-size 16\n"
        "streamlit run app.py"
    )

    st.stop()


# Load class names
with open(CLASS_PATH, "r", encoding="utf-8") as file:
    CLASS_NAMES = json.load(file)


if CLASS_NAMES != ["cats", "dogs"]:

    st.error("Model class labels do not match.")

    st.stop()


# Image upload
uploaded_file = st.file_uploader(
    "Upload Cat or Dog Image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

    except (OSError, ValueError):

        st.error("Invalid image file. Please upload a valid image.")

        st.stop()

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    # Prediction button
    if st.button(
        "Predict Image",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Checking uploaded image..."):

            try:
                label, confidence, validation_score = predict_image(image)

            except Exception as error:

                st.error(
                    "Image validation failed. "
                    "Please check the internet connection and app logs."
                )

                st.exception(error)

                st.stop()

        # Result 1: Invalid / Other image
        if label == "Not Classified":

            st.error("❌ NOT CLASSIFIED")

            st.warning(
                "Invalid Image! "
                "Please upload only a Cat or Dog image."
            )

        # Result 2: Valid Cat or Dog
        elif label == "Cat":

            st.success("🐱 Prediction: CAT")

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

        elif label == "Dog":

            st.success("🐶 Prediction: DOG")

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(confidence)

        st.caption(
            f"ImageNet Cat/Dog validation score: "
            f"{validation_score * 100:.2f}%"
        )

        st.caption(
            "This validation is a screening mechanism, "
            "not a guarantee. Some unusual images may be "
            "incorrectly accepted or rejected."
        )
