import streamlit as st
import pandas as pd
import joblib
import os

# --- 1. ROBUST MODEL LOADING ---
# This finds the file regardless of if you are on Windows, Mac, or Streamlit Cloud
base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "rfiris.pkl")

# If it's definitely inside a subfolder, use this instead:
# model_path = os.path.join(base_path, "ml_cok_deployment", "rfiris.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    # Fallback for different folder structures
    model = joblib.load("rfiris.pkl")

# --- 2. UI SETUP ---
st.title("IRIS FLOWER CLASSIFICATION")
st.write("Predict the species of an Iris Flower Using a Random Forest")

with st.form("iris_form"):
    st.subheader("Enter Flower Measurements")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.1)
        sepal_width = st.number_input("Sepal Width", min_value=1.0, max_value=4.5, value=3.5)
        
    with col2:
        petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, value=1.4)
        petal_width = st.number_input("Petal Width", min_value=0.1, max_value=2.5, value=0.2)
        
    submit_button = st.form_submit_button("Predict")

# --- 3. PREDICTION LOGIC ---
if submit_button:
    # IMPORTANT: The column names must match the TRAINING data exactly.
    # Usually, sklearn models trained on Iris expect this specific order:
    input_data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal_width (cm)": [petal_width]
    })
    
    # If the model still complains about feature names, use the .values to pass a raw array:
    try:
        prediction = model.predict(input_data)
    except ValueError:
        # Fallback if the model doesn't like the DataFrame column names
        prediction = model.predict(input_data.values)

    st.subheader("Prediction Result")
    st.success(f"The predicted species is: **{prediction[0]}**")
