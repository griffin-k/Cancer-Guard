from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Load dataset to get feature metadata (column names and preprocessing details)
def load_dataset():
    data = pd.read_csv("model/data.csv")
    data.columns = data.columns.str.strip()  # Remove leading/trailing spaces from column names
    X = data.drop(columns=["LUNG_CANCER"])
    return X

# Load trained model
def load_model():
    try:
        model = joblib.load("model/logistic_regression_model.pkl")
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found. Please ensure 'logistic_regression_model.pkl' is in the same directory.")
    return model

# Preprocess user input
def preprocess_input(user_inputs, categorical_features, numerical_features, preprocessor):
    input_data = pd.DataFrame([user_inputs])
    # Apply preprocessing
    input_data_transformed = preprocessor.transform(input_data)
    return input_data_transformed

# Convert 'Yes'/'No' to 2/1 respectively
def map_yes_no_to_numeric(value):
    return 2 if value == "YES" else 1

# Django view function
def lung_cancer_prediction(request):
    # Load dataset and model
    X = load_dataset()
    categorical_features = X.select_dtypes(include=["object"]).columns
    numerical_features = X.select_dtypes(exclude=["object"]).columns
    model = load_model()

    # Define preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(sparse_output=False, handle_unknown="ignore"), categorical_features),
        ]
    )
    # Fit the preprocessor using the original dataset
    preprocessor.fit(X)

    # Handle form submission
    result = None
    if request.method == "POST":
        # Convert 'Yes'/'No' to 2/1 in the user inputs
        user_inputs = {
            "GENDER": request.POST.get("gender"),
            "AGE": int(request.POST.get("age")),
            "SMOKING": map_yes_no_to_numeric(request.POST.get("smoking")),
            "YELLOW_FINGERS": map_yes_no_to_numeric(request.POST.get("yellow_fingers")),
            "ANXIETY": map_yes_no_to_numeric(request.POST.get("anxiety")),
            "PEER_PRESSURE": map_yes_no_to_numeric(request.POST.get("peer_pressure")),
            "CHRONIC DISEASE": map_yes_no_to_numeric(request.POST.get("chronic_disease")),
            "FATIGUE": map_yes_no_to_numeric(request.POST.get("fatigue").strip()),  # Strip any extra spaces
            "ALLERGY": map_yes_no_to_numeric(request.POST.get("allergy").strip()),  # Strip any extra spaces
            "WHEEZING": map_yes_no_to_numeric(request.POST.get("wheezing")),
            "ALCOHOL CONSUMING": map_yes_no_to_numeric(request.POST.get("alcohol_consuming")),
            "COUGHING": map_yes_no_to_numeric(request.POST.get("coughing")),
            "SHORTNESS OF BREATH": map_yes_no_to_numeric(request.POST.get("shortness_of_breath")),
            "SWALLOWING DIFFICULTY": map_yes_no_to_numeric(request.POST.get("swallowing_difficulty")),
            "CHEST PAIN": map_yes_no_to_numeric(request.POST.get("chest_pain")),
        }

        # Preprocess user input and predict
        input_data = preprocess_input(user_inputs, categorical_features, numerical_features, preprocessor)
        try:
            prediction = model.predict(input_data)[0]
            result = f"Lung Cancer Detected" if prediction == "YES" else "No Lung Cancer Detected"
        except Exception as e:
            result = f"An error occurred during prediction: {e}"

    # Render the page with form and result (if any)
    return render(request, "dashboard/detection.html", {"result": result})
