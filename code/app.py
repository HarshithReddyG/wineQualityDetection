# app.py

import streamlit as st
import pandas as pd
from joblib import load

# Load the trained stacking model, scaler, and feature names
stacking_model = load("../models/stacking_model.joblib")
scaler = load("../models/scaler.joblib")
feature_names = load("../models/feature_names.joblib")

# App title and description
st.title("Wine Quality Prediction Using Stacking Model")
st.write("""
This application predicts wine quality (**low**, **medium**, **high**) using a stacking model that combines Random Forest and LightGBM.
""")

# Sidebar for user input
st.sidebar.header("Input Wine Features")
def user_input_features():
    data = {
        "fixed acidity": st.sidebar.slider("Fixed Acidity", 4.0, 16.0, 8.0),
        "volatile acidity": st.sidebar.slider("Volatile Acidity", 0.1, 1.5, 0.5),
        "residual sugar": st.sidebar.slider("Residual Sugar", 0.0, 65.0, 10.0),
        "chlorides": st.sidebar.slider("Chlorides", 0.01, 0.2, 0.05),
        "total sulfur dioxide": st.sidebar.slider("Total Sulfur Dioxide", 0.0, 300.0, 120.0),
        "density": st.sidebar.slider("Density", 0.990, 1.005, 0.996),
        "pH": st.sidebar.slider("pH", 2.8, 4.0, 3.3),
        "sulphates": st.sidebar.slider("Sulphates", 0.2, 2.0, 0.6),
        "alcohol": st.sidebar.slider("Alcohol", 8.0, 15.0, 10.0)
    }
    return pd.DataFrame(data, index=[0], columns=feature_names)

# Get user input
input_df = user_input_features()

# Display user input
st.subheader("Input Features")
st.write(input_df)

# Scale input data
scaled_input = scaler.transform(input_df)

# Predict with stacking model
quality_mapping = {0: "Low", 1: "Medium", 2: "High"}
prediction = stacking_model.predict(scaled_input)
prediction_proba = stacking_model.predict_proba(scaled_input)

# Display prediction results
st.subheader("Prediction")
st.write(f"Predicted Quality: **{quality_mapping[prediction[0]]}**")

# Display prediction probabilities
st.subheader("Prediction Probabilities")
proba_df = pd.DataFrame(prediction_proba, columns=quality_mapping.values())
st.write(proba_df)

# Bar chart for probabilities
st.bar_chart(proba_df.T)
