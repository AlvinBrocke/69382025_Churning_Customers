import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the scaler and model
with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

with open("best_churn_model.pkl", "rb") as model_file:
    churn_model = pickle.load(model_file)

# Streamlit app
st.title("Churn Prediction App")

# User input
total_charges = st.number_input("Total Charges", min_value=0.0, value=0.0)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=0.0)
tenure = st.number_input("Tenure (months)", min_value=0, value=0)
contract_month_to_month = st.checkbox("Contract: Month-to-month")
online_security_no = st.checkbox("No Online Security")
payment_method_electronic_check = st.checkbox("Payment Method: Electronic check")
gender = st.selectbox("Gender", ["Male", "Female"])
internet_service_fiber_optic = st.checkbox("Internet Service: Fiber optic")
paperless_billing = st.checkbox("Paperless Billing")
tech_support_no = st.checkbox("No Tech Support")
partner = st.checkbox("Has Partner")
senior_citizen = st.checkbox("Senior Citizen")

# Button for making predictions
if st.button("Predict"):
    # Preprocess user inputs
    gender_male = 1 if gender == "Male" else 0

    # Create a feature array
    data = np.array(
        [
            [
                total_charges,
                monthly_charges,
                tenure,
                int(contract_month_to_month),
                int(online_security_no),
                int(payment_method_electronic_check),
                gender_male,
                int(internet_service_fiber_optic),
                int(paperless_billing),
                int(tech_support_no),
                int(partner),
                int(senior_citizen),
            ]
        ]
    )

    # Scale the input data using the loaded scaler
    scaled_input_data = scaler.transform(data)

    # Make predictions
    prediction = model.predict(scaled_input_data)[0]

    # Display the prediction result and confidence
    st.subheader("Churn Prediction Result:")

    st.subheader("Prediction:")
    st.write(f"The predicted output is: {prediction}")
        st.warning(
            f"The customer is likely to churn with {prediction_proba[0, 1] * 100:.2f}% confidence."
        )
    else:
        st.success(
            f"The customer is likely to stay with {prediction_proba[0, 0] * 100:.2f}% confidence."
        )

st.write(
    "This is a simple Streamlit web app to demonstrate a machine learning model prediction."
)
