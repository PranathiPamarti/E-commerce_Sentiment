import streamlit as st
import pandas as pd
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sentiment_reviews.csv"

@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["sentiment_value"] = (df["avg_sentiment_score"] > 0).astype(int)
    df["Time"] = pd.to_datetime(df["Time"], unit="s", errors="coerce", utc=True)
    return df

# -----------------------------------
st.title("📚 Dataset Explorer")

df = load_data()

# -----------------------------------
st.subheader("👀 Preview")
st.dataframe(df.head(50))

# -----------------------------------
st.subheader("🔎 Filter by User")
user = st.selectbox("User", df["UserId"].unique())
st.dataframe(df[df["UserId"] == user].head(50))

# -----------------------------------
st.subheader("🛒 Filter by Product")
product = st.selectbox("Product", df["ProductId"].unique())
st.dataframe(df[df["ProductId"] == product].head(50))
