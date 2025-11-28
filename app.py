import streamlit as st
import pickle
import pandas as pd
import numpy as np


with open("xgboost_fraud_model2.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = [
    'account_age_days', 'amount', 'promo_used', 'avs_match', 'cvv_result',
    'three_ds_flag', 'shipping_distance_km', 'country_match', 'country_ES',
    'country_FR', 'country_GB', 'country_IT', 'country_NL', 'country_PL',
    'country_RO', 'country_TR', 'country_US', 'bin_country_ES',
    'bin_country_FR', 'bin_country_GB', 'bin_country_IT', 'bin_country_NL',
    'bin_country_PL', 'bin_country_RO', 'bin_country_TR', 'bin_country_US',
    'channel_web'
]

COUNTRY_LABELS = {
    "ES": "Spain",
    "FR": "France",
    "GB": "United Kingdom",
    "IT": "Italy",
    "NL": "Netherlands",
    "PL": "Poland",
    "RO": "Romania",
    "TR": "Turkey",
    "US": "United States"
}

st.title("Fraud Detection Demo")


amount = st.number_input("Amount", min_value=0.0)
account_age_days = st.number_input("Account age (days)", min_value=0)
shipping_km = st.number_input("Shipping distance (km)", min_value=0.0)

country_label = st.selectbox("Country", list(COUNTRY_LABELS.values()))
bin_country_label = st.selectbox("BIN Country", list(COUNTRY_LABELS.values()))

country = [code for code, label in COUNTRY_LABELS.items() if label == country_label][0]
bin_country = [code for code, label in COUNTRY_LABELS.items() if label == bin_country_label][0]

promo_used = st.checkbox("Promo used")
avs_match = st.checkbox("AVS match")
cvv_result = st.checkbox("CVV OK")
three_ds = st.checkbox("3DS Passed")

channel = st.radio("Channel", ["Web", "App"])


data = {col: 0 for col in FEATURES}

data.update({
    'amount': amount,
    'account_age_days': account_age_days,
    'shipping_distance_km': shipping_km,
    'promo_used': int(promo_used),
    'avs_match': int(avs_match),
    'cvv_result': int(cvv_result),
    'three_ds_flag': int(three_ds),
    'channel_web': 1 if channel == "Web" else 0
})

data[f"country_{country}"] = 1
data[f"bin_country_{bin_country}"] = 1

data['country_match'] = int(country == bin_country)

df = pd.DataFrame([data])[FEATURES]

with st.expander("check the features sent to the model"):
    st.write(df.T)

best_threshold = 0.93

if st.button("Predict Fraud Probability"):
    proba = model.predict_proba(df)[0][1]
    prediction = "FRAUD" if proba >= best_threshold else "NOT FRAUD"

    if prediction == "FRAUD":
        st.error(f"Prediction: {prediction}")
    else:
        st.success(f"Prediction: {prediction}")
