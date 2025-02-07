import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import tensorflow as tf

# Load the model
model = load_model('blood_cell_cancer.h5')

# Define class names
class_names = ['Benign', '[Malignant] Pre-B', '[Malignant] Pro-B', '[Malignant] early Pre-B']

st.title("Blood Cell Cancer Classification(CNN)")

# File uploader
uploaded_file = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    input_img= tf.keras.utils.load_img(uploaded_file, target_size=(180,180 ))
    input_img_array=tf.keras.utils.img_to_array(input_img)
    img_array = np.expand_dims(np.array(input_img_array), axis=(0, -1))

    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.image(input_img, caption=f"Predicted: {predicted_class} ({confidence:.2f})", use_column_width=True)
