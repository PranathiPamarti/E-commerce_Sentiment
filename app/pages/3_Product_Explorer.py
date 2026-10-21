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
st.title("🔍 Product Explorer")

df = load_data()

product_id = st.selectbox("Select Product", df["ProductId"].unique())

prod_df = df[df["ProductId"] == product_id].sort_values("Time", ascending=False)

# -----------------------------------
st.subheader("📌 Product Summary")

cols = ["ProductId", "Summary", "Score"]
st.write(prod_df.head(1)[cols])

# -----------------------------------
st.subheader("😊 Recent Positive Reviews")
st.write(prod_df[prod_df["sentiment_value"] == 1][["Time", "Text"]].head(5))

# -----------------------------------
st.subheader("😞 Recent Negative Reviews")
st.write(prod_df[prod_df["sentiment_value"] == 0][["Time", "Text"]].head(5))
