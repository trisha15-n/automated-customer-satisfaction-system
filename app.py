import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from src.utils import load_object

st.set_page_config(page_title="Customer Satisfaction Predictor", layout="wide")

st.title("Customer Satisfaction Predictor")
st.markdown("Predicts whether a customer is 'Satisfied' or 'Not Satisfied' based on ticket data.")

st.sidebar.header("Model Settings")

threshold = st.sidebar.slider(
  "Happy Threshold (Sensitivity)",
  min_value=0.0,
  max_value=1.0,
  value=0.5,
  step=0.05,
  help="Adjust the threshold for classifying a customer as 'Satisfied'."
)

model_path = os.path.join("artifacts", "model.pkl")
preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

try:
    model = load_object(model_path)
    preprocessor = load_object(preprocessor_path)

    st.sidebar.success("Model and preprocessor loaded successfully.")
except Exception as e:
    st.sidebar.error(f"Error loading model or preprocessor: {e}")
    st.stop()


with st.form("prediction_form"):
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Customer Details")
        age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        product = st.selectbox("Product Purchased", ["iPhone16", "MacBook Pro", "iPad Air", "AirPods Pro", "Apple Watch", "Samsung Galaxy S23", "Dell XPS 13", "Google Pixel 7"])

    with col2:
        st.subheader("Ticket Details")
        ticket_type = st.selectbox("Ticket Type", ["Technical Issue", "Billing Issue", "Account Management", "General Inquiry"])
        priority = st.selectbox("Priority Level", ["Low", "Medium", "High", "Critical"])
        channel = st.selectbox("Ticket Channel", ["Email", "Phone", "Chat", "Social Media"])
        status = st.selectbox("Ticket Status", ["Open","Pending", "Closed"]) 

    st.subheader("Complaint Description")
    subject = st.text_input("Subject", "Product not working properly")
    description = st.text_area("Description", "I bought it last week and it's already broken. I am frustrated.")

    submitted = st.form_submit_button("Predict Satisfaction")

if submitted:
    input_data = pd.DataFrame({
        'Customer Age': [age],
        'Customer Gender': [gender],
        'Product Purchased': [product],
        'Ticket Type': [ticket_type],
        'Ticket Priority': [priority],
        'Ticket Channel':[channel],
        'Ticket Status': [status],
        'Ticket Subject': [subject],
        'Ticket Description': [description],
        'Ticket ID': [0],
        'Customer Name': ["User"],
        'Customer Email': ["user@example.com"],
        'Date of Purchase': ["2023-01-01"],
        'Resolution': ["None"],
        'First Response Time': ["0"],
        'Time to Resolution': ["0"]
    })
    try:
        
        import nltk
        from nltk.sentiment import SentimentIntensityAnalyzer
        nltk.download('vader_lexicon', quiet=True)
        sa = SentimentIntensityAnalyzer()

        input_data['full_text'] = input_data['Ticket Subject'] + ' ' + input_data['Ticket Description']

        input_data['sentiment_score'] = input_data['full_text'].apply(lambda x: sa.polarity_scores(x)['compound'])


        trandformed_data = preprocessor.transform(input_data)

        probs = model.predict_proba(trandformed_data)
        prob_happy = probs[0][1]

        if prob_happy >= threshold:
            prediction = 1
            result_text = "The customer is predicted to be Satisfied 😊"
        else:
            prediction = 0
            result_text = "The customer is predicted to be Not Satisfied 😞"

        st.markdown(f"### Prediction Result: {result_text}")

        st.progress(prob_happy, text=f"Predicted Satisfaction Probability: {prob_happy:.2f}")

        with st.expander("Prediction Details"):
            st.write("Raw Input Data:", input_data)
            st.write(f"Calculated Probability of Satisfaction: {prob_happy:.4f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")        
