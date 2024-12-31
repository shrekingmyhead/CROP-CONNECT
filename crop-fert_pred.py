import pandas as pd
import pickle
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
# Load the trained models from pickle files
def load_models():
    try:
        with open('crop_model.pkl', 'rb') as f:
            crop_model = pickle.load(f)
        with open('fertilizer_model.pkl', 'rb') as f:
            fertilizer_model = pickle.load(f)
        return crop_model, fertilizer_model
    except Exception as e:
        print(f"Error loading models: {e}")
        return None, None

crop_model, fertilizer_model = load_models()

# Take NPK values input from the user
def get_npk_values():
    try:
        n, p, k = map(int, input("Enter N, P, K values separated by space: ").split())
        return n, p, k
    except Exception as e:
        print(f"Invalid input: {e}")
        return None, None, None

# Predict crop and fertilizer
def predict_crop_and_fertilizer(n, p, k):
    if crop_model is None or fertilizer_model is None:
        print("Models not loaded. Cannot make predictions.")
        return None, None

    # Create a DataFrame with feature names
    new_data_crop = pd.DataFrame([[n, p, k, 21, 82, 7, 203]], 
                                 columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'])
    new_data_fertilizer = pd.DataFrame([[n, p, k]], columns=['N', 'P', 'K'])

    try:
        predicted_crop = crop_model.predict(new_data_crop)
        predicted_fertilizer = fertilizer_model.predict(new_data_fertilizer)
        return predicted_crop, predicted_fertilizer
    except Exception as e:
        print(f"Error making predictions: {e}")
        return None, None

# Main function
def main():
    n, p, k = get_npk_values()
    if n is not None and p is not None and k is not None:
        predicted_crop, predicted_fertilizer = predict_crop_and_fertilizer(n, p, k)
        if predicted_crop is not None and predicted_fertilizer is not None:
            print("Predicted Crop:", predicted_crop[0])
            print("Predicted Fertilizer:", predicted_fertilizer[0])

if __name__ == "__main__":
    main()