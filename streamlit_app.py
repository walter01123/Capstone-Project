
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

MODEL_PATH = './cc_delinquency_model/rf_pipeline.pkl'

st.set_page_config(page_title='Credit Card Delinquency Risk (Regression)', layout='centered')

st.title('Credit Card Delinquency Risk - Demo (Regression)')

if not os.path.exists(MODEL_PATH):
    st.error(f'Model not found at {MODEL_PATH}. Run train_model.py first.')
else:
    model = joblib.load(MODEL_PATH)

    st.write('Enter customer features:')

    def user_input_features():
        Credit_Limit = st.number_input('Credit Limit (₹)', value=50000, step=1000)
        Utilisation = st.number_input('Utilisation %', min_value=0.0, max_value=100.0, value=30.0)
        Avg_Payment_Ratio = st.number_input('Avg Payment Ratio (%)', min_value=0.0, max_value=100.0, value=80.0)
        Min_Due_Paid_Frequency = st.number_input('Min Due Paid Frequency (%)', min_value=0.0, max_value=100.0, value=10.0)
        Merchant_Mix_Index = st.number_input('Merchant Mix Index (0-1)', min_value=0.0, max_value=1.0, value=0.5, format="%.2f")
        Cash_Withdrawal = st.number_input('Cash Withdrawal %', min_value=0.0, max_value=100.0, value=5.0)
        Recent_Spend_Change = st.number_input('Recent Spend Change %', value=0.0)
        data = {'Credit_Limit':[Credit_Limit], 'Utilisation_%':[Utilisation], 'Avg_Payment_Ratio':[Avg_Payment_Ratio],
                'Min_Due_Paid_Frequency':[Min_Due_Paid_Frequency], 'Merchant_Mix_Index':[Merchant_Mix_Index],
                'Cash_Withdrawal_%':[Cash_Withdrawal], 'Recent_Spend_Change_%':[Recent_Spend_Change]}
        features = pd.DataFrame(data)
        return features

    input_df = user_input_features()

    if st.button('Predict risk score (DPD bucket predicted as numeric value)'):
        pred = model.predict(input_df)[0]
        st.metric('Predicted DPD Bucket (numeric)', round(float(pred), 3))
        st.write('Advice:')
        if pred >= 2:
            st.write('- High risk: consider outreach, payment plans, card limit review.')
        elif pred >= 1:
            st.write('- Medium risk: gentle reminders, offers for payment scheduling.')
        else:
            st.write('- Low risk: standard monitoring.')

    st.write('---')
    st.write('Model artifacts are expected in ./cc_delinquency_model/')
