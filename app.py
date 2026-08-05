import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# -------------------------
# Page Configuration
# -------------------------
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Car Price Prediction")
st.write("Predict the selling price of a used car using Machine Learning.")

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv("Car_Price.csv")

# Feature Engineering
df["Car_Age"] = 2026 - df["Year"]

# Convert categorical columns into numerical values
df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol":0,
    "Diesel":1,
    "CNG":2
})

df["Selling_type"] = df["Selling_type"].map({
    "Dealer":0,
    "Individual":1
})

df["Transmission"] = df["Transmission"].map({
    "Manual":0,
    "Automatic":1
})

# -------------------------
# Prepare Data
# -------------------------
X = df[[
    "Present_Price",
    "Driven_kms",
    "Fuel_Type",
    "Selling_type",
    "Transmission",
    "Owner",
    "Car_Age"
]]

y = df["Selling_Price"]

# Train Model
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------
# Sidebar
# -------------------------
st.sidebar.header("Enter Car Details")

present_price = st.sidebar.number_input(
    "Present Price (Lakhs)",
    min_value=0.0,
    value=5.0
)

driven_kms = st.sidebar.number_input(
    "Kilometers Driven",
    min_value=0,
    value=30000
)

fuel = st.sidebar.selectbox(
    "Fuel Type",
    ["Petrol","Diesel","CNG"]
)

selling = st.sidebar.selectbox(
    "Selling Type",
    ["Dealer","Individual"]
)

transmission = st.sidebar.selectbox(
    "Transmission",
    ["Manual","Automatic"]
)

owner = st.sidebar.selectbox(
    "Owner",
    [0,1,2,3]
)

year = st.sidebar.number_input(
    "Manufacturing Year",
    min_value=2000,
    max_value=2026,
    value=2018
)

car_age = 2026 - year

# Encode user input

fuel_map = {
    "Petrol":0,
    "Diesel":1,
    "CNG":2
}

selling_map = {
    "Dealer":0,
    "Individual":1
}

transmission_map = {
    "Manual":0,
    "Automatic":1
}

# -------------------------
# Prediction
# -------------------------
if st.button("Predict Selling Price"):

    input_data = [[
        present_price,
        driven_kms,
        fuel_map[fuel],
        selling_map[selling],
        transmission_map[transmission],
        owner,
        car_age
    ]]

    prediction = model.predict(input_data)

    st.success(f"💰 Estimated Selling Price: ₹ {prediction[0]:.2f} Lakhs")

# -------------------------
# Dataset Preview
# -------------------------
st.subheader("Dataset Preview")
st.dataframe(df.head())

# -------------------------
# Project Information
# -------------------------
st.info("""
This project predicts the selling price of a used car using Linear Regression.

Features used:
• Present Price
• Kilometers Driven
• Fuel Type
• Selling Type
• Transmission
• Owner
• Car Age
""")
