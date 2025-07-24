import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
model = joblib.load('final_stacked_model.pkl')

st.title("📦 Delivery Delay Prediction App")

st.markdown("Predict how many days the delivery might be delayed based on the order details.")

# 👇 Your inputs (they will stay the same as before)
order_approved_days = st.number_input("Days between Purchase & Approval", min_value=0.0, step=0.1)
shipping_limit_days = st.number_input("Days to Ship the Product", min_value=0.0, step=0.1)
freight_value = st.number_input("Freight Value", min_value=0.0, step=0.1)
product_weight_g = st.number_input("Product Weight (grams)", min_value=0.0, step=1.0)
product_volume_cm3 = st.number_input("Product Volume (cm³)", min_value=0.0, step=1.0)
payment_value = st.number_input("Payment Value", min_value=0.0, step=0.1)

# 👇 Create a dictionary with exact column names (and dummy values for any missing features)
input_data = {
    'purchase_month': 6,  # example value, update according to your data
    'purchase_day_of_week': 2,  # example value
    'purchase_hour': 14,  # example value
    'approval_time': order_approved_days,  # from input
    'carrier_time': shipping_limit_days,  # from input
    'estimated_wait_time': 10,  # example value
    'same_state': 1,  # example value
    'same_city': 0,  # example value
    'order_item_id': 1,  # example value
    'price': 50.0,  # example value
    'freight_value': freight_value,  # from input
    'product_weight_g': product_weight_g,  # from input
    'product_volume_cm3': product_volume_cm3,  # from input
    'payment_installments': 1,  # example value
    'payment_value': payment_value,  # from input
    'payment_boleto': 0,  # example value
    'payment_credit_card': 1,  # example value
    'payment_debit_card': 0,  # example value
    'payment_voucher': 0,  # example value
}

# 👇 Ensure the column order matches exactly the model's expected feature order
model_columns = ['purchase_month', 'purchase_day_of_week', 'purchase_hour', 'approval_time', 'carrier_time',
                 'estimated_wait_time', 'same_state', 'same_city', 'order_item_id', 'price', 'freight_value',
                 'product_weight_g', 'product_volume_cm3', 'payment_installments', 'payment_value', 'payment_boleto',
                 'payment_credit_card', 'payment_debit_card', 'payment_voucher']

# Create DataFrame with columns in the correct order
input_df = pd.DataFrame([input_data], columns=model_columns)

# 👇 Run the prediction
if st.button("Predict Delay"):
    prediction = model.predict(input_df)[0]
    st.success(f"🚚 Predicted Delivery Delay: {prediction:.2f} days")



