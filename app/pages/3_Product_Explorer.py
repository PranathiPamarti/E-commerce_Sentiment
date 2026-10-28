import streamlit as st
from pathlib import Path
from src.review_data import load_review_data, review_data_version

PROJECT_ROOT = Path(__file__).resolve().parents[2]

@st.cache_data(show_spinner=True)
def load_data(data_version):
    return load_review_data(PROJECT_ROOT)

# -----------------------------------
st.title("Product explorer · synthetic")
st.caption(
    "Review records are synthetic; sentiment labels are predicted from review text, "
    "not derived from star ratings."
)

try:
    df, products = load_data(review_data_version(PROJECT_ROOT))
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

product_ids = df["ProductId"].dropna().unique()
product_id = st.selectbox("Select Product", product_ids)

prod_df = df[df["ProductId"] == product_id].sort_values("Time", ascending=False)
product = products.loc[products["ProductId"] == product_id].iloc[0]

# -----------------------------------
st.subheader("📌 Product Summary")

cols = ["ProductId", "Summary", "Score"]
st.markdown(
    f"**{product['product_title']}** · {product['brand']} · "
    f"{product['product_category'].title()} · ${product['price']:.2f}"
)
st.write(
    prod_df[cols].head(1).assign(
        Review_Count=len(prod_df),
        Average_Rating=prod_df["Score"].mean(),
        Average_Sentiment=prod_df["avg_sentiment_score"].mean(),
    )
)

# -----------------------------------
st.subheader("😊 Positive text sentiment")
st.write(prod_df[prod_df["sentiment_value"] == 1][["Time", "Score", "Summary", "Text"]].head(5))

# -----------------------------------
st.subheader("Negative text sentiment")
st.write(prod_df[prod_df["sentiment_value"] == 0][["Time", "Score", "Summary", "Text"]].head(5))
