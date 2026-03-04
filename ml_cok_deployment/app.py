import streamlit as st 
import pandas as pd 
import joblib 

model = joblib.load("rfiris.pkl")

st.title(IRIS FLOWER )

model = joblib.load("rfiris.pkl")

st.write("IRIS FLOWER CLASSIFICATION APPLICATION")

st.title("Predict the species of an Iris Flower Using a Random Forest ")
form = st.form("iris form")


form.subheader("Enter Flower Measurement")

sepal_length = form.number_input (
    "sepal_length (cm)",
    min_value = 4.0,
    max_value = 8.0,
    value = 5.1


)
sepal_width = form.number_input (
    "sepal_width (cm)",
    min_value = 1.0,
    max_value = 4.5,
    value = 4.0


)

petal_length = form.number_input (
    "petal_length (cm)",
    min_value = 1.0,
    max_value = 7.0,
    value = 5.0


)

petal_width = form.number_input (
    "petal_width (cm)",
    min_value = 0.1,
    max_value = 2.5,
    value = 0.2


)
submit_button = form.form_submit_button("predict")

if submit_button:
    input_data = pd.DataFrame({
        "sepal_length (cm)" : [sepal_length],
        "petal_length (cm)" : [petal_length],
        "sepal_width (cm)" : [sepal_width],
        "petal_width (cm)" : [petal_width]

    })

    prediction = model.predict (input_data)

    st.subheader ("Prediction Result")
    st.success(f"Predicted species: {prediction[0]}")