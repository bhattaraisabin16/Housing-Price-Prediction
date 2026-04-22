import streamlit as st
import pandas as pd
import joblib

# 1. Load the model
# Ensure 'housing_model.pkl' is uploaded to your GitHub repo
model = joblib.load('housing_model.pkl')

# 2. Create the User Interface
st.title("California House Price Predictor 🏠")
st.write("Adjust the values below to predict the median house value in a district.")

# Organizing inputs into two columns for a cleaner look
col1, col2 = st.columns(2)

with col1:
    income = st.slider("Median Income (in $10k)", 0.5, 15.0, 3.8)
    age = st.number_input("Housing Median Age", 1, 52, 28)
    rooms = st.number_input("Total Rooms", 10, 40000, 2500)
    bedrooms = st.number_input("Total Bedrooms", 1, 7000, 500)

with col2:
    pop = st.number_input("Population", 1, 35000, 1400)
    households = st.number_input("Households", 1, 6000, 500)
    lat = st.number_input("Latitude", 32.5, 42.0, 35.0)
    lon = st.number_input("Longitude", -124.3, -114.3, -119.0)

# Ocean Proximity (Simplified for this version - using a 0/1 placeholder)
# Note: If your model used One-Hot Encoding for Ocean Proximity, 
# you'll need additional sliders/logic for those specific columns.
ocean = 0 

# 3. Prediction Logic
if st.button("Calculate Predicted Price"):
    # The order of this list MUST match your model's training data exactly
    features = [[
        lon, lat, age, rooms, bedrooms, pop, households, income, ocean
    ]]
    
    prediction = model.predict(features)
    st.success(f"The estimated median house value is: ${prediction[0]:,.2f}")
