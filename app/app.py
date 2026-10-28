from pathlib import Path

import streamlit as st


APP_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Review Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

overview = st.Page(
    APP_ROOT / "pages" / "0_Project_Overview.py",
    title="Project overview",
    default=True,
)
data_profile = st.Page(
    APP_ROOT / "pages" / "1_Data_Profile.py",
    title="Dataset and data quality",
)
sentiment_model = st.Page(
    APP_ROOT / "pages" / "2_Sentiment_Model.py",
    title="Sentiment model",
)
issue_intelligence = st.Page(
    APP_ROOT / "pages" / "5_Review_Intelligence.py",
    title="Issues and products",
)
reviewer_signals = st.Page(
    APP_ROOT / "pages" / "7_Reviewer_Signals.py",
    title="Reviewer signals",
)
issue_audit = st.Page(
    APP_ROOT / "pages" / "6_Issue_Tag_Audit.py",
    title="Human issue-tag audit",
)
synthetic_insights = st.Page(
    APP_ROOT / "pages" / "1_Product_Insights.py",
    title="Product insights · synthetic",
)
recommendation_demo = st.Page(
    APP_ROOT / "pages" / "2_Recommendation_Demo.py",
    title="Recommendation prototype · synthetic",
)
product_explorer = st.Page(
    APP_ROOT / "pages" / "3_Product_Explorer.py",
    title="Product explorer · synthetic",
)
synthetic_data = st.Page(
    APP_ROOT / "pages" / "4_Dataset_Explorer.py",
    title="Dataset explorer · synthetic",
)

navigation = st.navigation(
    {
        "Project": [overview],
        "Review intelligence": [
            data_profile,
            sentiment_model,
            issue_intelligence,
            reviewer_signals,
            issue_audit,
        ],
        "Synthetic demo": [
            synthetic_insights,
            recommendation_demo,
            product_explorer,
            synthetic_data,
        ],
    }
)
navigation.run()
