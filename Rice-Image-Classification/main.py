import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = load_model('Rice_Model.h5')

# Define class names
class_names = ['Arborio', 'Basmati', 'Ipsala','Jasmine','Karacadag']

st.title("Rice Image Classification(CNN)")

# File uploader
uploaded_file = st.file_uploader("Upload Rice image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    img = img.resize((128, 128))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=(0, -1))

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(img, caption=f"Predicted: {predicted_class} ({confidence:.2f})", use_column_width=True)