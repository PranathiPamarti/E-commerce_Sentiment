from pathlib import Path

import streamlit as st

from src.review_data import load_review_data, review_data_version


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=True)
def load_data(data_version):
    return load_review_data(PROJECT_ROOT)


st.title("Dataset Explorer")
st.caption(
    "This page and Product Insights use the same prepared review records and sentiment labels."
)

try:
    df, products = load_data(review_data_version(PROJECT_ROOT))
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

category_options = ["All"] + sorted(df["ProductCategory"].dropna().unique().tolist())
selected_category = st.selectbox("Category", category_options)
user_options = ["All"] + sorted(df["UserId"].dropna().unique().tolist())
selected_user = st.selectbox("User", user_options)
product_options = ["All"] + sorted(df["ProductId"].dropna().unique().tolist())
selected_product = st.selectbox("Product", product_options)
sentiment_options = ["All"] + sorted(df["sentiment_label"].dropna().unique().tolist())
selected_sentiment = st.selectbox("Text sentiment", sentiment_options)

filtered = df.copy()
if selected_category != "All":
    filtered = filtered[filtered["ProductCategory"] == selected_category]
if selected_user != "All":
    filtered = filtered[filtered["UserId"] == selected_user]
if selected_product != "All":
    filtered = filtered[filtered["ProductId"] == selected_product]
if selected_sentiment != "All":
    filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

if filtered.empty:
    st.info("No reviews match these filters.")
    st.stop()

positive_share = filtered["sentiment_value"].mean() * 100
columns = st.columns(4)
columns[0].metric("Matching reviews", f"{len(filtered):,}")
columns[1].metric("Products", f"{filtered['ProductId'].nunique():,}")
columns[2].metric("Average rating", f"{filtered['Score'].mean():.2f} / 5")
columns[3].metric("Positive text sentiment", f"{positive_share:.1f}%")

product_lookup = products[
    ["ProductId", "product_title", "brand", "product_category", "price"]
].rename(columns={"product_category": "CatalogCategory"})
display_data = filtered.merge(
    product_lookup,
    on="ProductId",
    how="left",
    validate="many_to_one",
    suffixes=("", "_catalog"),
)

st.subheader("Matching review records")
st.dataframe(
    display_data.sort_values("Time", ascending=False, na_position="last"),
    hide_index=True,
    use_container_width=True,
)
st.caption(
    f"Showing all {len(display_data):,} matching rows. "
    "Text sentiment is positive when avg_sentiment_score > 0; this is distinct from a 4–5 star rating."
)
