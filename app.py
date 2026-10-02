"""Streamlit interface for RecycleVision inference."""

import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


st.set_page_config(
    page_title="RecycleVision",
    page_icon="♻️",
    layout="centered"
)


@st.cache_resource
def load_artifacts(model_path: str):
    path = Path(model_path)

    labels_path = path.with_name(
        path.stem + "_labels.json"
    )

    labels = json.loads(labels_path.read_text())

    model = tf.keras.models.load_model(path)

    return model, labels


# Waste information
waste_info = {
    "cardboard": {
        "emoji": "📦",
        "message": "Cardboard can usually be recycled. Flatten boxes and keep them dry and clean."
    },
    "glass": {
        "emoji": "🍾",
        "message": "Glass containers are commonly recyclable. Rinse them before recycling."
    },
    "metal": {
        "emoji": "🥫",
        "message": "Metal cans can often be recycled. Empty and rinse them before disposal."
    },
    "paper": {
        "emoji": "📄",
        "message": "Clean paper is generally recyclable. Avoid recycling heavily contaminated paper."
    },
    "plastic": {
        "emoji": "🧴",
        "message": "Check your local recycling rules for plastic. Empty and rinse containers when possible."
    },
    "trash": {
        "emoji": "🗑️",
        "message": "This item may not belong in standard recycling. Check your local waste disposal rules."
    }
}


# Header
st.title("♻️ RecycleVision")

st.caption(
    "AI-powered waste classification using Deep Learning"
)

st.divider()


# Sidebar
st.sidebar.header("⚙️ Model Settings")

model_file = st.sidebar.text_input(
    "Model file",
    "artifacts/models/mobilenetv2.keras"
)

st.sidebar.divider()

st.sidebar.subheader("🤖 Model Information")

st.sidebar.write("**Architecture:** MobileNetV2")
st.sidebar.write("**Task:** Image Classification")
st.sidebar.write("**Classes:** 6")
st.sidebar.write("**Input Size:** 224 × 224")

st.sidebar.divider()

st.sidebar.info(
    "Upload an image of a waste item and RecycleVision "
    "will predict its recycling category."
)


# Image upload
st.subheader("📸 Upload Waste Image")

uploaded = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png", "webp"]
)


# Prediction
if uploaded:

    image = Image.open(uploaded).convert("RGB")

    st.image(
        image,
        caption="Uploaded Waste Image",
        use_container_width=True
    )

    st.divider()

    if not Path(model_file).exists():

        st.error(
            "❌ Model file not found. "
            "Please train the model first or select the correct model."
        )

    else:

        with st.spinner("🤖 Analyzing image..."):

            model, labels = load_artifacts(model_file)

            resized = image.resize((224, 224))

            image_array = np.asarray(
                resized,
                dtype=np.float32
            )

            image_array = np.expand_dims(
                image_array,
                axis=0
            )

            probabilities = model.predict(
                image_array,
                verbose=0
            )[0]

        # Top 3 predictions
        order = probabilities.argsort()[::-1][:3]

        best = order[0]

        predicted_label = labels[best].lower()

        confidence = float(probabilities[best])


        # Main result
        st.subheader("🎯 Prediction")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Predicted Category",
                labels[best].title()
            )

        with col2:
            st.metric(
                "Confidence",
                f"{confidence:.1%}"
            )


        # Recycling information
        if predicted_label in waste_info:

            info = waste_info[predicted_label]

            st.success(
                f"{info['emoji']} **{labels[best].title()}**\n\n"
                f"{info['message']}"
            )


        # Top 3 predictions
        st.subheader("🏆 Top 3 Predictions")

        for rank, idx in enumerate(order, start=1):

            label = labels[idx].title()

            probability = float(probabilities[idx])

            st.write(
                f"**{rank}. {label}** — "
                f"{probability:.1%}"
            )

            st.progress(probability)


        # Confidence message
        st.divider()

        if confidence >= 0.80:

            st.success(
                "✅ The model is highly confident in this prediction."
            )

        elif confidence >= 0.50:

            st.warning(
                "⚠️ The model has moderate confidence. "
                "Try uploading a clearer image."
            )

        else:

            st.warning(
                "⚠️ The model has low confidence. "
                "Try a clearer image showing a single waste item."
            )


else:

    st.info(
        "👆 Upload a photo of a single waste item to begin."
    )

    st.markdown(
        """
        ### Supported Categories

        📦 Cardboard &nbsp;&nbsp; 🍾 Glass &nbsp;&nbsp; 🥫 Metal

        📄 Paper &nbsp;&nbsp; 🧴 Plastic &nbsp;&nbsp; 🗑️ Trash
        """
    )