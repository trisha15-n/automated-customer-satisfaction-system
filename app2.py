import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os
import csv
from datetime import datetime
from src.utils import load_object
from src.components.priority_engine import PriorityEngine
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# --- CONFIGURATION ---
st.set_page_config(page_title="Customer Support AI System", layout="wide", page_icon="🎧")

# --- LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Adjust paths if necessary
        sat_model = load_object(os.path.join('artifacts', 'model.pkl'))
        sat_prep = load_object(os.path.join('artifacts', 'preprocessor.pkl'))
        cat_model = load_object(os.path.join('artifacts', 'model_category.pkl'))
        cat_prep = load_object(os.path.join('artifacts', 'preprocessor_category.pkl'))
        
        priority_engine = PriorityEngine()
        nltk.download('vader_lexicon', quiet=True)
        sia = SentimentIntensityAnalyzer()
        return sat_model, sat_prep, cat_model, cat_prep, priority_engine, sia
    except Exception as e:
        return None, None, None, None, None, None

sat_model, sat_prep, cat_model, cat_prep, priority_engine, sia = load_resources()

# --- HELPER: LOGGING ---
HISTORY_FILE = "artifacts/history.csv"
def log_prediction(subject, category, priority, churn, sentiment, product):
    file_exists = os.path.isfile(HISTORY_FILE)
    with open(HISTORY_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "Subject", "Category", "Priority", "Churn Risk", "Sentiment Score", "Product"])
        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), subject, category, priority, churn, sentiment, product])

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🎧 Support System")
page = st.sidebar.radio("Go to:", ["🤖 Ticket Predictor", "📈 Raw Data Analysis", "📊 Live Manager Dashboard"])

# ====================================================
# PAGE 1: TICKET PREDICTOR (The Agent Tool)
# ====================================================
if page == "🤖 Ticket Predictor":
    st.title("🤖 Automated Ticket Triage")
    st.markdown("AI-powered classification and priority detection for support agents.")

    if not sat_model:
        st.error("⚠️ Models not found in 'artifacts/'. Please train your models first.")
        st.stop()

    with st.form("ticket_form"):
        c1, c2 = st.columns(2)
        with c1:
            age = st.number_input("Customer Age", 18, 100, 30)
            gender = st.selectbox("Gender", ['Male', 'Female'])
            product = st.selectbox("Product", ["iPhone16", "MacBook Pro", "Samsung S23", "Dell XPS"])
            channel = st.selectbox("Channel", ["Email", "Chat", 'Phone'])
            status = st.selectbox("Status", ['Open', 'Pending', "Closed"])
        with c2:
            subject = st.text_input("Subject", "Login failing")
            description = st.text_input("Description", "I cannot access my account.")
        
        submitted = st.form_submit_button("Analyze Ticket")

    if submitted:
        full_text = f"{subject} {description}"
        sentiment_score = sia.polarity_scores(full_text)['compound']
        
        # --- GUARDRAILS ---
        safe_keywords = ["thank", "great job", "no issue", "invoice", "receipt"]
        is_safe = (sentiment_score > 0.05) or (any(k in full_text.lower() for k in safe_keywords) and sentiment_score > -0.1)
        
        pred_priority = priority_engine.predict_priority(subject, description)
        crit_keywords = ["login", "password", "access", "fail", "crash", "hacked"]
        if any(k in full_text.lower() for k in crit_keywords): pred_priority = "High"
        elif is_safe: pred_priority = "Low"

        # --- PREDICTION ---
        if is_safe:
            category_name = "General Inquiry"
            churn_res = "Low Risk (Retained)"
            prob_safe = 0.99
        else:
            # 1. Category
            cat_input = pd.DataFrame({'Customer Age':[age], 'Ticket Priority': [pred_priority], 
                                      'Ticket Channel': [channel], 'Product Purchased': [product], 
                                      'Ticket Status': [status], 'Customer Gender': [gender], 
                                      'Ticket Subject': [subject], 'Ticket Description': [description]})
            cat_input['full_text'] = full_text
            cat_trans = cat_prep.transform(cat_input)
            if hasattr(cat_trans, "toarray"): cat_trans = cat_trans.toarray()
            cat_pred = cat_model.predict(cat_trans)
            category_map = {0:'Account Access', 1:'General Inquiry', 2:'Firmware', 3:'Technical Support', 4:'Product Issue'}
            category_name = category_map.get(cat_pred[0], "Unknown")

            # 2. Churn
            sat_input = pd.DataFrame({'Customer Age': [age], 'Customer Gender': [gender], 'Product Purchased': [product],
                'Ticket Type': [category_name], 'Ticket Priority': [pred_priority], 'Ticket Channel': [channel],
                'Ticket Status': [status], 'Ticket Subject': [subject], 'Ticket Description': [description],
                'Ticket ID': [0], 'Customer Name': ["User"], 'Customer Email': ["x"], 'Date of Purchase': ["2023"], 
                'Resolution': ["None"], 'First Response Time': ["0"], 'Time to Resolution': ["0"]})
            sat_input['full_text'] = full_text
            sat_input['sentiment_score'] = sentiment_score
            sat_trans = sat_prep.transform(sat_input)
            probs = sat_model.predict_proba(sat_trans)
            prob_safe = probs[0][1]
            churn_res = "Likely to Churn" if prob_safe < 0.5 else "Low Risk (Retained)"

        # --- OUTPUT ---
        st.divider()
        k1, k2, k3 = st.columns(3)
        k1.success(f"📂 {category_name}")
        if pred_priority == "High": k2.error(f"🔥 {pred_priority}")
        else: k2.info(f"🟢 {pred_priority}")
        k3.metric("Retention Probability", f"{prob_safe:.1%}", delta_color="normal")
        
        log_prediction(subject, category_name, pred_priority, churn_res, sentiment_score, product)
        st.toast("✅ Logged to Database")

# ====================================================
# PAGE 2: RAW DATA ANALYSIS (EDA)
# ====================================================
elif page == "📈 Raw Data Analysis":
    st.title("📈 Raw Data Analysis")
    st.markdown("Upload your historical CSV to analyze trends, imbalance, and correlations.")
    
    uploaded_file = st.file_uploader("Upload Ticket CSV", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Ticket Priority Distribution")
            fig = px.pie(df, names='Ticket Priority', title="Check for Imbalance")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.subheader("Customer Age Distribution")
            fig = px.histogram(df, x="Customer Age", nbins=20, title="Age Demographics")
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Category vs. Status")
        if 'Ticket Type' in df.columns and 'Ticket Status' in df.columns:
            fig = px.histogram(df, x="Ticket Type", color="Ticket Status", barmode="group")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Columns 'Ticket Type' and 'Ticket Status' needed for this chart.")

# ====================================================
# PAGE 3: LIVE MANAGER DASHBOARD (The Monitoring Tool)
# ====================================================
elif page == "📊 Live Manager Dashboard":
    st.title("📊 Live System Monitor")
    st.markdown("Monitoring the AI tool's performance in real-time.")
    
    if os.path.exists(HISTORY_FILE):
        df = pd.read_csv(HISTORY_FILE)
        
        # KPIS
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Predictions", len(df))
        m2.metric("Critical Alerts", len(df[df['Priority']=='High']), delta_color="inverse")
        m3.metric("Churn Risks", len(df[df['Churn Risk']=='Likely to Churn']), delta_color="inverse")
        m4.metric("Avg Sentiment", f"{df['Sentiment Score'].mean():.2f}")
        
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Real-time Sentiment Trend")
            df['Timestamp'] = pd.to_datetime(df['Timestamp'])
            fig = px.line(df, x='Timestamp', y='Sentiment Score', markers=True)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.subheader("Issues by Product")
            fig = px.bar(df, x='Product', color='Priority')
            st.plotly_chart(fig, use_container_width=True)
            
        st.subheader("Recent Logs")
        st.dataframe(df.sort_values(by="Timestamp", ascending=False).head(10))
    else:
        st.info("No live data yet. Go to 'Ticket Predictor' and analyze some tickets!")