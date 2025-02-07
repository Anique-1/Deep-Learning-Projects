import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
model = load_model('lungs_x_ray.h5')

# Define class names
class_names = ['Lung_Opacity', 'Normal', 'Viral Pneumonia']

st.title("Lungs X-ray Classification(CNN)")

# File uploader
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_img= tf.keras.utils.load_img(uploaded_file, target_size=(180,180 ))
    input_img_array=tf.keras.utils.img_to_array(input_img)
    img_array = np.expand_dims(np.array(input_img_array), axis=(0, -1))

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(input_img, caption=f"Predicted: {predicted_class} ({confidence:.2f})", use_column_width=True)
