import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="Skin Disease Classifier",
    page_icon="🩺",
    layout="centered"
)

# Cache model loading so it only loads once
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('mobilenetv2_skin_disease_classifier.keras')

try:
    model = load_my_model()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loaded = False

# Class names mapping
class_names = [
    "BA- cellulitis",
    "BA-impetigo",
    "FU-athlete-foot",
    "FU-nail-fungus",
    "FU-ringworm",
    "PA-cutaneous-larva-migrans",
    "VI-chickenpox",
    "VI-shingles"
]

st.title("🩺 Skin Disease Image Classification")
st.write("Research prototype using MobileNetV2. Upload an image to classify.")

uploaded_file = st.file_uploader("Choose a skin image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model_loaded:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Classifying...")
    
    # Preprocess image
    image_resized = image.convert("RGB").resize((224, 224))
    image_array = np.asarray(image_resized, dtype=np.float32) / 255.0
    image_batch = np.expand_dims(image_array, axis=0)
    
    # Predict
    probabilities = model.predict(image_batch, verbose=0)[0]
    top_indices = np.argsort(probabilities)[::-1][:3]
    
    st.subheader("Top Predictions:")
    for i in top_indices:
        st.write(f"**{class_names[i]}**: {probabilities[i]*100:.2f}%")
        st.progress(float(probabilities[i]))
