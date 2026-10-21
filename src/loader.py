import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# -------------------------
# CACHE: Load Dataset
# -------------------------
@st.cache_data(show_spinner=True)
def load_dataset():
    df = pd.read_csv(PROJECT_ROOT / "data" / "sentiment_reviews.csv")
    return df

# -------------------------
# CACHE: Load Hybrid Model Components
# -------------------------
@st.cache_resource(show_spinner=True)
def load_hybrid_model():
    with open(PROJECT_ROOT / "models" / "hybrid_model.pkl", "rb") as f:
        hybrid = pickle.load(f)
    return hybrid

# -------------------------
# CACHE: Load Cosine Similarity
# -------------------------
@st.cache_resource(show_spinner=True)
def load_cosine_matrix():
    return np.load(PROJECT_ROOT / "models" / "cosine_sim.npy")

# -------------------------
# CACHE: Load Product Text DF
# -------------------------
@st.cache_data(show_spinner=True)
def load_product_text():
    return pd.read_csv(PROJECT_ROOT / "data" / "product_text.csv")
