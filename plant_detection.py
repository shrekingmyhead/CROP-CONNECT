import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import warnings

# Suppress TensorFlow warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the saved model
def load_model():
    try:
        return tf.keras.models.load_model('plant_disease_model.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Function to preprocess the input image
def preprocess_image(image_path):
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to predict the class name (disease name) of the image
def predict_disease(image_path, model, class_names):
    if model is None:
        print("Model not loaded. Cannot make predictions.")
        return None
    img_array = preprocess_image(image_path)
    if img_array is None:
        return None
    try:
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions)
        predicted_class_name = class_names[predicted_class_index]
        return predicted_class_name
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None

# Main function
def main():
    # Load the model
    model = load_model()
    if model is None:
        return

    # Define the class names based on the dataset
    class_names = [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy"
    ]

    # Predict the disease for an image
    image_path = 'plant_disease.jpg'  # Replace with the path to your image
    predicted_class_name = predict_disease(image_path, model, class_names)
    if predicted_class_name is not None:
        print('Predicted class name:', predicted_class_name)

if __name__ == "__main__":
    main()