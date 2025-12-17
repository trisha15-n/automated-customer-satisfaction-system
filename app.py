import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import nltk
from src.utils import load_object
from src.components.priority_engine import PriorityEngine
from nltk.sentiment import SentimentIntensityAnalyzer
from src.logger import info
from src.exception import CustomException


st.set_page_config(page_title="Customer Support System", layout="wide")
st.title("Customer Support System")
st.markdown("""
The system uses three AI models to automate Ticket Handling:
1. Ticket Classifier: Auto-detects ticket's category.
2. Priority Engine: Detects urgency of the task based on keyword.
3. Sentiment Analysis: Analyse Customer's sentiments while addressing the issue and likeability to churn.      
""")

st.sidebar.header("Model Threshold")
threshold = st.sidebar.slider(
  "Like to Churn",
  min_value = 0.0, max_value = 1.0, value=0.5, step=0.5,
  help = "Increase the value to catch customers who can churn"
)

@st.cache_resource
def load_all_models():
  
  sat_model = load_object(os.path.join('artifacts', 'model.pkl'))
  sat_prep = load_object(os.path.join('artifacts', 'preprocessor.pkl'))

  cat_model = load_object(os.path.join('artifacts', 'model_category.pkl'))
  cat_prep = load_object(os.path.join('artifacts', 'preprocessor_category.pkl'))
  cat_le = load_object(os.path.join('artifacts', 'label_encoder_category.pkl'))

  priority_engine = PriorityEngine()

  nltk.download('vader_lexicon', quiet=True)
  sia = SentimentIntensityAnalyzer()

  return sat_model, sat_prep, cat_model, cat_prep, cat_le, priority_engine, sia


try:
  sat_model, sat_prep, cat_model, cat_prep, cat_le, priority_engine, sia = load_all_models()
  st.sidebar.success("All AI Models Loaded Successfully.")

except Exception as e:
  st.error(f"Error loading models: {e}")
  st.stop()


with st.form("ticket_form"):
  col1, col2 = st.columns(2)

  with col1:
    st.subheader("Customer Profile")
    age = st.number_input("Customer Age", min_value=18, max_value=100, value=30)
    gender = st.selectbox("Gender", ['Male', 'Female', 'Other'])
    product = st.selectbox("Product Purchased", ["iPhone16", "MacBook Pro", "iPad Air", "AirPods Pro", "Apple Watch", "Samsung Galaxy S23", "Dell XPS 13", "Google Pixel 7"])
    channel = st.selectbox("Channel", ["Email", "Chat", 'Phone', "Social Media"])
    status = st.selectbox("Current Status", ['Open', 'Pending', "Closed"])

  with col2:
    st.subheader("Ticket Information")
    subject = st.text_input("Subject", "My screen is broken")
    description = st.text_input("Description","I dropped my phone and the screen cracked. I need a repair immediately.")

  submitted = st.form_submit_button("Analyse")


if submitted:
  st.divider()

  full_text = f"{subject} {description}"

  sentiment_score = sia.polarity_scores(full_text)['compound']

  positive_keywords = ["no issue", "no problem", "thank you", "thanks", "great job", "all good", "no complaints", "not bad", "satisfied", "happy","invoice", "receipt", "bill request", "tracking", "status", "address"]

  is_praise = False

  if sentiment_score > 0.5 or any(phase in full_text.lower() for phase in positive_keywords):
    is_praise = True

  pred_priority = priority_engine.predict_priority(subject, description)

  critical_keywords = ["login", "password", "access", "unable", "fail", "reset", "locked", "crash"]

  is_critical = any(k in full_text.lower() for k in critical_keywords)
  
  if is_critical:
    pred_priority = "High"
    is_praise = False
  elif is_praise:
    pred_priority = "Low"

  cat_input = pd.DataFrame({
    'Customer Age':[age],
    'Ticket Priority': [pred_priority],
    'Ticket Channel': [channel],
    'Product Purchased': [product],
    'Ticket Status': [status],
    'Customer Gender': [gender],
    'Ticket Subject': [subject],
    'Ticket Description': [description]

  })

  cat_input['full_text'] = cat_input['Ticket Subject'].fillna('') + " "+ cat_input['Ticket Description'].fillna('')

  if is_praise:
    category_name = "General Inquiry / Positive Feedback"
  else:
    cat_transformed = cat_prep.transform(cat_input)
    if hasattr(cat_transformed, "toarray"):
      cat_transformed = cat_transformed.toarray()

    cat_pred_index = cat_model.predict(cat_transformed)

    category_map = {
      0:'Account Access',
      1:'General Inquiry',
      2:'Firmware/Update Issue',
      3: 'Technical Support',
      4: 'Product Issue'
    }
    category_name = category_map.get(cat_pred_index[0], "Unknown")

  r_col1, r_col2 = st.columns(2)
  with r_col1:
    st.success(f"Predicted Category: {category_name}")
  with r_col2:
    #prior_color = "red" if pred_priority in ["Critical", "High"] else "blue"
    if pred_priority in ["High", "Critical"]:
      st.error(f"Predicted Priority: {pred_priority}")
    else:
      st.success(f"Predicted Priority: {pred_priority}") 


  sat_input_df = pd.DataFrame({
    'Customer Age': [age],
        'Customer Gender': [gender],
        'Product Purchased': [product],
        'Ticket Type': [category_name], 
        'Ticket Priority': [pred_priority], 
        'Ticket Channel': [channel],
        'Ticket Status': [status],
        'Ticket Subject': [subject],
        'Ticket Description': [description],
        'Ticket ID': [0], 'Customer Name': ["User"], 'Customer Email': ["x"], 
        'Date of Purchase': ["2023-01-01"], 'Resolution': ["None"], 
        'First Response Time': ["0"], 'Time to Resolution': ["0"]
  }) 

  sat_input_df['full_text'] = sat_input_df['Ticket Subject'] + " " + sat_input_df['Ticket Description']

  sat_input_df['sentiment_score'] = sentiment_score

  if is_praise:
    prob_not_churn = 0.99
    sentiment_rs = "Not Likely to Churn."

  else:
    sat_transformed = sat_prep.transform(sat_input_df)
    probs = sat_model.predict_proba(sat_transformed)
    prob_not_churn = probs[0][1]

    if prob_not_churn >= 0.5:
      sentiment_rs = "Not Likely to Churn."

    else:
      sentiment_rs = "Likely to Churn."


  st.markdown(f"Customer is  {sentiment_rs}")
  st.progress(prob_not_churn, text=f"Probabilty of Retention: {prob_not_churn:.2%}")

  with st.expander("See Behind the Scenes"):
        st.write("Sentiment Score (VADER):", sat_input_df['sentiment_score'][0])
        st.write("Is Critical?", is_critical)
        st.write("Is Praise?", is_praise)
        st.write("Data sent to Satisfaction Model:", sat_input_df)


  

