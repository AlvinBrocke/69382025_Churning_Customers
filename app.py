import streamlit as st
import pickle
import numpy as np
import pandas as pd
import sklearn

# Load the scaler and model
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("best_churn_model.pkl", "rb") as model_file:
    churn_model = pickle.load(model_file)

# Streamlit app
st.title("Churn Prediction App")

# User input
total_charges = st.number_input(
    "Total Charges:  \n The total amount of charges incurred by the customer over the past year",
    min_value=0.0,
    step=1.0,
)
monthly_charges = st.number_input(
    "Monthly Charges:  \n The customer's monthly bill amount", min_value=0.0, step=1.0
)
tenure = st.number_input(
    "Tenure (months): \n The number of months the customer has been a customer",
    min_value=0.0,
    step=1.0,
)
contract = st.selectbox(
    "Contract:  \n The type of contract the customer has with the company (month-to-month, other)",
    ["Month-to-month", "Other"],
)
online_security_no = st.selectbox(
    "Online Security:  \n Whether or not the customer has online security enabled on their account",
    ["Yes", "No"],
)
payment_method_electronic_check = st.selectbox(
    "Payment Method (Electronic Check): Whether the customer use Electronic Payment Check",
    ["Yes", "No"],
)
gender = st.selectbox("Gender:  \n The customer's gender.", ["Male", "Female"])
internet_service_fiber_optic = st.selectbox(
    "Internet Service(Fiber Optic):  \n Whether the customer has Fiber Optic Internet service or not ",
    ["Yes", "No"],
)
paperless_billing = st.selectbox(
    "Paperless Billing  \n Whether or not the customer does paperless billing",
    ["Yes", "No"],
)
tech_support_no = st.selectbox(
    "Tech Support:  \n Whether or not the customer has tech support enabled on their account",
    ["Yes", "No"],
)
partner = st.selectbox(
    "Partner :  \n Whether or not the customer has a partner.", ["Yes", "No"]
)
senior_citizen = st.selectbox(
    "Senior Citizen :  \n Whether the customer is senior citizen or not", ["Yes", "No"]
)

# Processing user input
data = pd.DataFrame(
    {
        "TotalCharges": [total_charges],
        "MonthlyCharges": [monthly_charges],
        "tenure": [tenure],
        "Contract_Month-to-month": [1 if contract == "Month-to-month" else 0],
        "OnlineSecurity_No": [1 if online_security_no == "No" else 0],
        "InternetService_Fiber optic": [
            1 if internet_service_fiber_optic == "Yes" else 0
        ],
        "gender": [1 if gender == "Male" else 0],
        "PaymentMethod_Electronic check": [
            1 if payment_method_electronic_check == "Electronic check" else 0
        ],
        "PaperlessBilling": [1 if paperless_billing == "Yes" else 0],
        "TechSupport_No": [1 if tech_support_no == "No" else 0],
        "Partner": [1 if partner == "Yes" else 0],
        "SeniorCitizen": [1 if senior_citizen == "Yes" else 0],
    }
)

# Scale the input data using the loaded scaler
scaled_input_data = scaler.transform(data)

if st.button("Predict"):
    # Make predictions using the model
    prediction = churn_model.predict(scaled_input_data)

    # Display the prediction result and confidence
    confidence_rate = round(float(prediction[0]), 2) * 100
    st.write(f"The Prediction is {confidence_rate}% confident.")

    if prediction[0] > 0.5:
        st.write("Customer is likely to churn")
    else:
        st.write("Customer is not likely to churn")


st.write(
    "This is a simple Streamlit web app to demonstrate a machine learning model prediction."
)
