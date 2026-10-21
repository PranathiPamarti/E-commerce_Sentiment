import streamlit as st

st.set_page_config(page_title="E-Commerce Dashboard", layout="wide")

st.title("📊 E-Commerce Sentiment-Enhanced Recommendation System")

st.markdown("""
This dashboard showcases:
- Sentiment analysis using Transformers  
- Content-based similarity (TF-IDF)  
- Manual collaborative filtering (Matrix Factorization)  
- Hybrid recommendations  
""")

st.info("Use the sidebar to explore insights and recommendations.")
