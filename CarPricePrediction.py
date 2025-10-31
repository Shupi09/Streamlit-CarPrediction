import streamlit as st
import pickle
import pandas as pd
import os 

import gdown

drive_url = drive_url = 'https://drive.google.com/uc?id=1u0cilrHNwD7G-xVu8IKswj2_0tf9CdoU&confirm=t'
 # Replace with your file's actual ID
filename = 'rf_pipeline.pkl'


if not os.path.isfile(filename):
    with st.spinner('Downloading model...'):
        gdown.download(drive_url, filename, quiet=False)
# Example categorical options from your training data
car_name_options = ['Maruti Alto', 'Hyundai Grand', 'Hyundai i20', 'Ford Ecosport', 'Maruti Wagon R', 'Volvo XC60', 'Honda CR', 'Jaguar XF', 'Tata Altroz', 'Force Gurkha']
  # Replace with actual unique values
brand_options = ['Maruti', 'Hyundai', 'Ford', 'Renault', 'Mercedes-Benz', 'Nissan', 'Land Rover', 'Jeep', 'Jaguar', 'Force']
model_options = ['Alto', 'Grand', 'i20', 'Ecosport', 'Wagon R', 'XC60', 'CR', 'XF', 'Altroz', 'Gurkha']
fuel_type_options = ['Petrol', 'Diesel', 'CNG']
seller_type_options = ['Dealer', 'Individual']
transmission_type_options = ['Manual', 'Automatic']

st.title("Vehicle Price Prediction App")
st.header("Please enter vehicle details")

# User inputs for numerical features
vehicle_age = st.slider('Vehicle Age (years)', 0, 30, 5)
km_driven = st.number_input('Kilometers Driven', 0, 300000, 50000)
mileage = st.number_input('Mileage (kmpl)', 0.0, 50.0, 15.0)
engine = st.number_input('Engine capacity (CC)', 500, 8000, 1500)
max_power = st.number_input('Max Power (bhp)', 10.0, 500.0, 100.0)
seats = st.number_input('Number of Seats', 2, 16, 5)

# User inputs for categorical features
car_name = st.selectbox('Car Name', car_name_options)
brand = st.selectbox('Brand', brand_options)
model = st.selectbox('Model', model_options)
fuel_type = st.selectbox('Fuel Type', fuel_type_options)
seller_type = st.selectbox('Seller Type', seller_type_options)
transmission_type = st.selectbox('Transmission Type', transmission_type_options)

# Prepare input data as DataFrame with the same order & columns as training data
input_df = pd.DataFrame({
    'vehicle_age': [vehicle_age],
    'km_driven': [km_driven],
    'mileage': [mileage],
    'engine': [engine],
    'max_power': [max_power],
    'seats': [seats],
    'car_name': [car_name],
    'brand': [brand],
    'model': [model],
    'fuel_type': [fuel_type],
    'seller_type': [seller_type],
    'transmission_type': [transmission_type]
})

if st.button("Get the prediction"):
    # Load your saved pipeline model
    with open('rf_pipeline.pkl', 'rb') as f:
        model = pickle.load(f)

    # Make prediction
    predicted_price = model.predict(input_df)
    
    st.success(f"The predicted price of the vehicle is: R{predicted_price[0]:.2f}")
