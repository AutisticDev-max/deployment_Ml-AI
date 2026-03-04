import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np

base_path = os.path.dirname(__file__)
model_path = os.path.join(base_path, "rfiris.pkl")

try:
    model = joblib.load(model_path)
except FileNotFoundError:
    model = joblib.load("rfiris.pkl")

st.title("IRIS FLOWER CLASSIFICATION")
st.write("Predict the species of an Iris Flower and see the model's confidence.")

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

if submit_button:
    input_data = pd.DataFrame({
        "sepal length (cm)": [sepal_length],
        "sepal width (cm)": [sepal_width],
        "petal length (cm)": [petal_length],
        "petal width (cm)": [petal_width]
    })
    
    try:
        prediction = model.predict(input_data)
        probabilities = model.predict_proba(input_data)
    except ValueError:
        prediction = model.predict(input_data.values)
        probabilities = model.predict_proba(input_data.values)

    st.subheader("Prediction Result")
    st.success(f"The predicted species is: **{prediction[0]}**")

    st.subheader("Prediction Confidence")
    class_names = model.classes_
    for name, prob in zip(class_names, probabilities[0]):
        percent = prob * 100
        st.write(f"**{name}**: {percent:.2f}%")
        st.progress(prob)
