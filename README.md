# CROP-CONNECT 

Crop Connect is a platform aimed at providing various services related to agriculture, including disease detection, crop suggestion, and fertilizer recommendation based on NPK values. This project leverages machine learning and deep learning techniques to assist farmers and agricultural enthusiasts in making informed decisions.

## Features 

1. **Disease Detection**:
   - Detects plant diseases using deep learning techniques.
   - Upload an image of a plant, and the system will predict the disease.

2. **Crop Suggestion**:
   - Recommends suitable crops based on the NPK (Nitrogen, Phosphorus, Potassium) values of the soil.
   - Enter the NPK values, and the system will suggest the best crop for your soil.

3. **Fertilizer Recommendation**:
   - Recommends fertilizers based on the NPK values of the soil.
   - Enter the NPK values, and the system will suggest the appropriate fertilizer.

4. **User-Friendly Interface**:
   - Built with Streamlit, providing an intuitive and interactive web interface.
   - Easy to use for farmers and agricultural professionals.

## Technologies Used 

- **Python**: Primary programming language.
- **Streamlit**: For building the web application.
- **TensorFlow/Keras**: For deep learning-based disease detection.
- **Scikit-learn**: For crop and fertilizer recommendation models.
- **Pandas/Numpy**: For data manipulation and processing.
- **OpenCV/Pillow**: For image preprocessing.

## Dataset 

The dataset used for disease detection has been taken from the **PlantVillage dataset**. This dataset contains images of healthy and diseased plants, categorized by plant type and disease. It is widely used for training machine learning models in agricultural applications.

- **Dataset Source**: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/emmarex/plantdisease)

## Installation 

Follow these steps to set up the project locally:

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Steps

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/CROP-CONNECT.git
   cd CROP-CONNECT
   ```
2. Create a Virtual Environment (Optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate # On Windows: venv\Scripts\activate
   ```
3. Install Dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Application:
    ```bash
    streamlit run app.py
    ```
5. Access the App:
   Open your browser and go to http://localhost:8501

## Project Structure 
```
CROP-CONNECT/
├── data/                    # Dataset for disease detection
│   └── PlantDisease/
│       └── train/           # Training images for disease detection
├── models/                  # Pre-trained models
│   ├── crop_model.pkl       # Crop suggestion model
│   ├── fertilizer_model.pkl # Fertilizer recommendation model
│   └── plant_disease_model.h5 # Plant disease detection model
├── scripts/                 # Python scripts
│   ├── app.py               # Streamlit app
│   ├── crop-fert_pred.py    # Crop and fertilizer prediction logic
│   ├── modify_model.py      # Model modification scripts
│   └── plant_detection.py   # Plant disease detection logic
├── requirements.txt         # List of dependencies
├── README.md                # Project documentation
└── .gitignore               # Files to ignore in Git
```
