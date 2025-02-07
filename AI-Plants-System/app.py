import os
import torch
import joblib
import pickle
import streamlit as st
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# Load all models with caching to prevent reloading on each run
@st.cache_resource
def load_models():
    models = {}
    try:
        models['crop_growth'] = joblib.load('corn crop growth\\crop_growth_model.pkl')
    except Exception as e:
        st.error(f"Error loading crop growth model: {e}")
        models['crop_growth'] = None

    try:
        models['fertilizer'] = joblib.load('crop and fretilizer\\fertilizer_type_model.pkl')
        # with open('crop and fretilizer\\fertilizer_type_model.pkl', 'rb') as file:
        #     models['fertilizer'] = pickle.load(file)
    except Exception as e:
        st.error(f"Error loading fertilizer model: {e}")
        models['fertilizer'] = None

    try:
        models['disease_processor'] = AutoImageProcessor.from_pretrained("wambugu71/crop_leaf_diseases_vit")
        models['disease_model'] = AutoModelForImageClassification.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    except Exception as e:
        st.error(f"Error loading leaf disease model: {e}")
        models['disease_processor'], models['disease_model'] = None, None

    return models

# Prediction functions
def predict_crop_growth(models, temperature, humidity, soil_moisture):
    if models['crop_growth']:
        input_data = np.array([[temperature, humidity, soil_moisture]])
        prediction = models['crop_growth'].predict(input_data)[0]
        st.success(f'Predicted Crop Growth: {prediction:.2f} cm')
    else:
        st.error("Crop Growth Model not available")
#Soil_color	Nitrogen	Phosphorus	Potassium	pH	Rainfall	Temperature	Crop
def recommend_fertilizer(models, soil_type, temperature, Phosphorus, Potassium, nitrogen, pH, Rainfall, crop_type):
    if models['fertilizer']:
        FERTILIZER_TYPES = {0: '10:10:10 NPK', 1: '10:26:26 NPK', 2: '12:32:16 NPK', 3: '13:32:16 NPK', 4: '18:46:00 NPK', 5: '19:19:19',
                            6:'20:20:20 NPK', 7:'50:26:26 NPK', 8:'Ammonium Sulphate', 9:'Chilated Micronutrient', 10:'DAP',
                            11:'Ferrous Sulphate', 12:'Hydrated Lime', 13:'MOP', 14:'Magnesium Sulphate', 15:'SSP',
                            16:'Sulphur', 17:'Urea', 18:'White Potash'}
        soil_map = {'Black':0, 'Red':4, 'Dark Brown':1, 'Dark Red':5, 'Reddish Brown': 6, 'Light Brown': 2, 'Medium Brown':3}
        crop_map = {'Sugercane':11, 'Wheat':15, 'Cotton':0, 'Jowar':5, 'Maize':6, 'Rice':9, 'GroundNut':4, 'Tur': 12, 'Grapes':3, 'Ginger':1,
                    'Urad':14, 'Moong':8, 'Gram':2, 'Turmeric':13, 'Soybean':10, 'Masoor':7}

        input_data = np.array([[soil_map[soil_type], nitrogen, Phosphorus, Potassium, pH, Rainfall, temperature, crop_map[crop_type]]])
        prediction = models['fertilizer'].predict(input_data)[0]
        st.success(f'Recommended Fertilizer: {FERTILIZER_TYPES.get(prediction, "Unknown")}')
    else:
        st.error("Fertilizer Recommendation Model not available")

def classify_leaf_disease(models, image):
    if models['disease_processor'] and models['disease_model']:
        inputs = models['disease_processor'](images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = models['disease_model'](**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(1).item()
        class_labels = models['disease_model'].config.id2label
        confidence = torch.softmax(logits, dim=1).max().item() * 100
        st.success(f"Predicted Disease: {class_labels[predicted_class_idx]} ({confidence:.2f}%)")
    else:
        st.error("Leaf Disease Classification Model not available")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Agriculture ML Dashboard", page_icon="🌾")
    st.title("🌱 Agriculture Machine Learning Dashboard")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select Application", ["Crop Growth Prediction", "Fertilizer Recommendation", "Leaf Disease Classification"])
    models = load_models()

    if page == "Crop Growth Prediction":
        st.header("Crop Growth Prediction")
        temperature = st.slider('Temperature (°C)', 10.0, 40.0, 25.0)
        humidity = st.slider('Humidity (%)', 30.0, 90.0, 60.0)
        soil_moisture = st.slider('Soil Moisture (%)', 10.0, 40.0, 20.0)
        if st.button('Predict Crop Growth'):
            predict_crop_growth(models, temperature, humidity, soil_moisture)

    elif page == "Fertilizer Recommendation":
        st.header("Fertilizer Recommendation")
        soil_type = st.selectbox('Soil Type', ['Black', 'Red', 'Dark Brown', 'Dark Red', 'Reddish Brown', 'Light Brown', 'Medium Brown'])
        crop_type = st.selectbox('Crop Type', ['Sugercane', 'Wheat','Cotton', 'Jowar', 'Maize', 'Rice', 'GroundNut', 'Tur', 'Grapes', 'Ginger', 'Ured', 'Moong', 'Gram', 'Turmeric', 'Soybean', 'Masoor' ])
        temperature = st.slider('Temperature (°C)', 0.0, 50.0, 25.0)
        humidity = st.slider('Humidity (%)', 0, 200, 100)
        moisture = st.slider('Soil Moisture (%)', 0, 200, 100)
        nitrogen = st.slider('Nitrogen Level', 0, 200, 100)
        phosphorus = st.slider('Phosphorus Level', 0, 200, 100)
        potassium = st.slider('Potassium Level', 0, 200, 100)
        if st.button('Recommend Fertilizer'):
            recommend_fertilizer(models, soil_type, temperature, humidity, moisture, nitrogen, phosphorus, potassium, crop_type)

    elif page == "Leaf Disease Classification":
        st.header("Leaf Disease Classification")
        uploaded_file = st.file_uploader("Upload a crop leaf image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            if st.button('Classify Disease'):
                classify_leaf_disease(models, image)

if __name__ == "__main__":
    main()
