import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------
# Cached data loader
# -----------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(PROJECT_ROOT / "data" / "sentiment_reviews.csv")

    if "sentiment_value" not in df.columns:
        df["sentiment_value"] = (df["avg_sentiment_score"] > 0).astype(int)

    return df

@st.cache_resource(show_spinner=True)
def load_hybrid_model():
    with open(PROJECT_ROOT / "models" / "hybrid_model.pkl", "rb") as f:
        return pickle.load(f)

# -----------------------------------
st.title("🎯 Hybrid Recommendation Demo")

df = load_data()
hybrid = load_hybrid_model()

P = hybrid["P"]
Q = hybrid["Q"]
cosine_sim = hybrid["cosine_sim"]
sentiment_map = hybrid["sentiment_map"]
product_text = hybrid["product_text"]
indices = pd.Series(product_text.index, index=product_text["ProductId"])

# -----------------------------------
# Helper functions
# -----------------------------------
def get_cf_score(user_idx, prod_idx):
    return float(np.dot(P[user_idx], Q[prod_idx]))

def get_sentiment_score(pid):
    s = sentiment_map.get(pid, 0)
    return (s + 1) / 2     # normalize to 0–1 range

def get_content_similarity(pid1, pid2):
    if pid1 not in indices or pid2 not in indices:
        return 0
    return float(cosine_sim[indices[pid1]][indices[pid2]])

def hybrid_score(uid, pid, ref=None):
    try:
        user_idx = df[df["UserId"] == uid]["user_index"].iloc[0]
        prod_idx = df[df["ProductId"] == pid]["product_index"].iloc[0]
    except:
        return 0

    cf = get_cf_score(user_idx, prod_idx)
    sent = get_sentiment_score(pid)
    cont = get_content_similarity(ref, pid) if ref else 0

    return 0.6*cf + 0.2*sent + 0.2*cont

# -----------------------------------
# User inputs
# -----------------------------------
user_id = st.selectbox("Select User", df["UserId"].unique())
reference_product = st.selectbox("Reference Product (optional)", df["ProductId"].unique())

# -----------------------------------
st.subheader("Top Recommendations")

popular_products = df["ProductId"].value_counts().head(200).index

scores = [(pid, hybrid_score(user_id, pid, reference_product)) for pid in popular_products]
scores = sorted(scores, key=lambda x: x[1], reverse=True)[:10]

st.dataframe(pd.DataFrame(scores, columns=["ProductId", "Final Score"]))
