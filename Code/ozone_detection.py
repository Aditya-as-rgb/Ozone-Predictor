import streamlit as st
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt

# Load trained model
with open("ozone_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load scaler
with open("scaler.pkl", "rb") as file:
    scaler = pickle.load(file)

# Load historical data (replace with your actual CSV name)
data = pd.read_csv("antarctic-ozone-hole-area.csv")  # Must have 'Year' and 'Mean Ozone Hole Area' columns

# Title
st.title("Ozone Hole Area Prediction")
st.write("Predict the **Mean Ozone Hole Area (in km²)** for a given year using historical data.")

# Input: Year
year = st.number_input("Enter Year", min_value=int(data['Year'].min()), max_value=2100, value=2025)

# Predict button
if st.button("Predict"):
    try:
        # Prepare input for model
        features = np.array([[year]])
        scaled_prediction = model.predict(features)

        # Create dummy array for inverse transformation
        dummy_array = np.zeros((1, scaler.n_features_in_))
        dummy_array[:, -1] = scaled_prediction.ravel()
        prediction_km2 = scaler.inverse_transform(dummy_array)[:, -1][0]

        # Display prediction
        st.success(f"Predicted Mean Ozone Hole Area for {year}: {prediction_km2:,.2f} km²")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
