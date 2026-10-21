import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "sentiment_reviews.csv"

# -----------------------------------
# Cached data loader
# -----------------------------------
@st.cache_data(show_spinner=True)
def load_data():
    df = pd.read_csv(DATA_PATH)

    # Create missing binary sentiment_value column
    if "sentiment_value" not in df.columns:
        df["sentiment_value"] = (df["avg_sentiment_score"] > 0).astype(int)

    return df

# -----------------------------------
st.title("📦 Product Insights Dashboard")

df = load_data()

# -----------------------------------
st.subheader("⭐ Rating Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.countplot(x=df["Score"], ax=ax)
st.pyplot(fig)
plt.close()

# -----------------------------------
st.subheader("💬 Sentiment Distribution per Product")
product_sentiment = df.groupby("ProductId")["avg_sentiment_score"].mean()

fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(product_sentiment, bins=30, ax=ax)
st.pyplot(fig)
plt.close()

# -----------------------------------
st.subheader("☁ Word Cloud of Reviews")
text_column = "clean_text" if "clean_text" in df.columns else "Text"
review_text = df[text_column].dropna().astype(str).str.strip()
review_text = review_text[review_text.ne("")]
sample_text = " ".join(
    review_text.sample(min(1500, len(review_text)), random_state=42)
)

if sample_text:
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
    ).generate(sample_text)

    fig = plt.figure(figsize=(12, 6))
    plt.imshow(np.array(wordcloud.to_image()), interpolation="bilinear")
    plt.axis("off")
    st.pyplot(fig)
    plt.close(fig)
else:
    st.info("There is no review text available to build a word cloud.")
