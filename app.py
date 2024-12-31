import streamlit as st
import pandas as pd
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import os

# Load the models
@st.cache_resource  # Use st.cache_resource for models
def load_models():
    crop_model = pickle.load(open('crop_model.pkl', 'rb'))
    fertilizer_model = pickle.load(open('fertilizer_model.pkl', 'rb'))
    plant_disease_model = tf.keras.models.load_model('plant_disease_model.h5')
    return crop_model, fertilizer_model, plant_disease_model

crop_model, fertilizer_model, plant_disease_model = load_models()

# Function to preprocess the input image for disease detection
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to predict the disease
def predict_disease(image_path):
    img_array = preprocess_image(image_path)
    predictions = plant_disease_model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    
    # Path to the dataset folder
    dataset_path = 'PlantDisease/train'  # Updated to match your folder structure
    
    # Check if the dataset folder exists
    if not os.path.exists(dataset_path):
        st.error(f"Dataset folder not found: {dataset_path}")
        return "Unknown Disease"
    
    # Get the list of class names from the dataset folder
    try:
        subfolder_names = sorted([f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f)) and f != "PlantVillage"])
    except FileNotFoundError:
        st.error(f"Dataset folder not found: {dataset_path}")
        return "Unknown Disease"
    
    # Ensure the predicted index is within the range of subfolder_names
    if predicted_class_index < len(subfolder_names):
        predicted_subfolder_name = subfolder_names[predicted_class_index]
        return predicted_subfolder_name
    else:
        return "Unknown Disease"

# List of fertilizer classes
fertilizer_classes = [
    "Mono Potassium Phosphate", "N.P.K. (22-22-11)", "Diammonium Phosphate", "N.P.K. (12:11:18 with MgO)",
    "N.P.K. (16:16:16)", "N.P.K. 15::9:20", "NPK (18-18-18)", "Urea Phosphate", "Zincated NPK (10:26:26:0.5)",
    "Zincated NPK (12:32:16:0.5)", "Boronated DAP (18:46:0:0.3)", "Boronated NPK (10:26:26:0.3)",
    "Boronated NPK (12:32:16:0.3)", "Diammonium Phosphate", "Mono Ammonium Phosphate", "N.P.K. (14-28-14)",
    "N.P.K. (14-35-14)", "N.P.K. (15-15-15)", "N.P.K. (15:15:15:9 S)", "N.P.K. (17-17-17)", "N.P.K. (19-19-19)",
    "N.P.K. 15:15:15", "NPK (13-5-26)", "NPK (19-19-19)", "NPK (20-20-20)", "Nitrophosphate", "Nitrophosphate with Potash",
    "Potassium Nitrate", "Urea Ammonium Phosphate", "Ammonium Phosphate", "Ammonium Phosphate Sulphate",
    "Ammonium Phosphate Sulphate Nitrate", "N.P.K. (20-10-10)", "N.P.K.(12-32-16)", "Nitro Phosphate",
    "Urea Ammonium Phosphates", "Urea Ammonium phosphate"
]

# Streamlit app
st.title("Crop Connect")

st.sidebar.title("Navigation")
options = st.sidebar.radio("Select a service:", 
                           ["Home", "Disease Detection", "Crop Suggestion", "Fertilizer Recommendation"])

if options == "Home":
    st.write("""
    ### Welcome to Crop Connect!
    This platform provides various services related to agriculture, including:
    - **Disease Detection**: Detects plant diseases using deep learning techniques.
    - **Crop Suggestion**: Recommends suitable crops based on NPK values.
    - **Fertilizer Recommendation**: Recommends fertilizers based on NPK values.
    """)

elif options == "Disease Detection":
    st.header("Plant Disease Detection")
    uploaded_file = st.file_uploader("Upload an image of a plant", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)  # Updated parameter
        st.write("")
        st.write("Classifying...")
        predicted_disease = predict_disease(uploaded_file)
        st.write(f"Predicted Disease: {predicted_disease}")

elif options == "Crop Suggestion":
    st.header("Crop Suggestion")
    st.write("Enter the NPK values of your soil to get crop recommendations.")
    n = st.number_input("Nitrogen (N) value", min_value=0)
    p = st.number_input("Phosphorus (P) value", min_value=0)
    k = st.number_input("Potassium (K) value", min_value=0)
    if st.button("Get Crop Recommendation"):
        new_data_crop = [[n, p, k, 21, 82, 7, 203]]  # Example data for crop prediction
        predicted_crop = crop_model.predict(new_data_crop)
        st.write(f"Recommended Crop: {predicted_crop[0]}")

elif options == "Fertilizer Recommendation":
    st.header("Fertilizer Recommendation")
    st.write("Enter the NPK values of your soil to get fertilizer recommendations.")
    n = st.number_input("Nitrogen (N) value", min_value=0)
    p = st.number_input("Phosphorus (P) value", min_value=0)
    k = st.number_input("Potassium (K) value", min_value=0)
    if st.button("Get Fertilizer Recommendation"):
        new_data_fertilizer = [[n, p, k]]
        predicted_fertilizer_index = fertilizer_model.predict(new_data_fertilizer)
        
        # Ensure predicted_fertilizer_index is an integer or string
        try:
            # If the model returns a string, use it directly
            if isinstance(predicted_fertilizer_index[0], str):
                predicted_fertilizer = predicted_fertilizer_index[0]
            else:
                # If the model returns an integer, use it as an index
                predicted_fertilizer_index = int(predicted_fertilizer_index[0])
                predicted_fertilizer = fertilizer_classes[predicted_fertilizer_index]
            st.write(f"Recommended Fertilizer: {predicted_fertilizer}")
        except (IndexError, ValueError) as e:
            st.error(f"Error predicting fertilizer: {e}")