import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from src.review_data import load_review_data, review_data_version

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# -----------------------------------
# Cached data loader
# -----------------------------------
@st.cache_data(show_spinner=True)
def load_data(data_version):
    reviews, _ = load_review_data(PROJECT_ROOT)
    return reviews

@st.cache_resource(show_spinner=True)
def load_hybrid_model(model_version):
    with open(PROJECT_ROOT / "models" / "hybrid_model.pkl", "rb") as f:
        return pickle.load(f)

# -----------------------------------
st.title("Recommendation prototype")
st.caption(
    "Synthetic demonstration only. Its user/product latent factors are generated "
    "by the data simulator; they are not fitted by a recommendation-training step."
)

try:
    df = load_data(review_data_version(PROJECT_ROOT))
    model_path = PROJECT_ROOT / "models" / "hybrid_model.pkl"
    model_stat = model_path.stat()
except FileNotFoundError as error:
    st.error(
        "Demo data or model files are missing. From the project root, run "
        "`python -m src.generate_synthetic_data`."
    )
    st.stop()

hybrid = load_hybrid_model((model_stat.st_mtime_ns, model_stat.st_size))

P = hybrid["P"]
Q = hybrid["Q"]
cosine_sim = hybrid["cosine_sim"]
product_text = hybrid["product_text"]
if "ProductId" not in product_text.columns or product_text["ProductId"].duplicated().any():
    st.error("The model product catalog must contain unique ProductId values.")
    st.stop()
indices = pd.Series(product_text.index, index=product_text["ProductId"])

# -----------------------------------
# Helper functions
# -----------------------------------
def get_cf_score(user_idx, prod_idx):
    return float(np.dot(P[user_idx], Q[prod_idx]))

def normalize_scores(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("Collaborative-filtering scores contain non-finite values.")
    score_range = np.ptp(scores)
    if score_range == 0:
        return np.full(scores.shape, 0.5)
    return (scores - scores.min()) / score_range


def get_content_scores(candidate_ids, reference_product):
    if reference_product not in indices:
        raise ValueError(f"Reference product {reference_product!r} is missing from product text.")
    reference_index = int(indices[reference_product])
    if cosine_sim.ndim != 2 or cosine_sim.shape[0] != cosine_sim.shape[1]:
        raise ValueError("The content-similarity matrix must be square.")
    if cosine_sim.shape[0] != len(product_text):
        raise ValueError("The content-similarity matrix does not match the product catalog.")

    scores = []
    for product_id in candidate_ids:
        if product_id not in indices:
            raise ValueError(f"Product {product_id!r} is missing from product text.")
        product_index = int(indices[product_id])
        scores.append(float(cosine_sim[reference_index, product_index]))
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or (scores < -1e-6).any() or (scores > 1 + 1e-6).any():
        raise ValueError("Content-similarity scores must be finite values between 0 and 1.")
    return np.clip(scores, 0, 1)

# -----------------------------------
# User inputs
# -----------------------------------
user_ids = sorted(df["UserId"].dropna().unique().tolist())
user_id = st.selectbox("Select User", user_ids)
product_ids = sorted(df["ProductId"].dropna().unique().tolist())
reference_product = st.selectbox(
    "Reference Product (optional)",
    [None, *product_ids],
    format_func=lambda value: "None (no reference product)" if value is None else value,
)

# -----------------------------------
st.subheader("Top Recommendations")

required_columns = {
    "UserId",
    "ProductId",
    "user_index",
    "product_index",
    "avg_sentiment_score",
}
missing_columns = required_columns.difference(df.columns)
if missing_columns:
    st.error(f"Review data is missing required columns: {', '.join(sorted(missing_columns))}.")
    st.stop()

user_rows = df.loc[df["UserId"] == user_id, "user_index"].dropna().unique()
if len(user_rows) != 1:
    st.error(f"Expected one model index for user {user_id!r}; found {len(user_rows)}.")
    st.stop()

user_index = int(user_rows[0])
if float(user_rows[0]) != user_index:
    st.error(f"User index {user_rows[0]!r} is not an integer model index.")
    st.stop()
if P.ndim != 2 or Q.ndim != 2 or P.shape[1] != Q.shape[1]:
    st.error("The collaborative-filtering model has incompatible factor dimensions.")
    st.stop()
if user_index < 0 or user_index >= P.shape[0]:
    st.error(f"User index {user_index} is outside the collaborative-filtering model.")
    st.stop()

product_rows = df.dropna(subset=["ProductId", "product_index"])
product_index_counts = product_rows.groupby("ProductId")["product_index"].nunique()
if product_index_counts.gt(1).any():
    st.error("Review data maps a product to multiple collaborative-filtering indices.")
    st.stop()

seen_products = set(df.loc[df["UserId"] == user_id, "ProductId"].dropna())
popular_products = df["ProductId"].value_counts().head(200).index
candidate_ids = [
    product_id
    for product_id in popular_products
    if product_id not in seen_products and product_id != reference_product
]

if not candidate_ids:
    st.info("No unseen products are available to recommend for this user.")
    st.stop()

product_index_by_id = product_rows.drop_duplicates("ProductId").set_index("ProductId")["product_index"]
candidate_model_indices = product_index_by_id.reindex(candidate_ids)
if candidate_model_indices.isna().any():
    missing_products = candidate_model_indices[candidate_model_indices.isna()].index.tolist()
    st.error(f"Products are missing collaborative-filtering indices: {missing_products[:5]}.")
    st.stop()
candidate_indices = candidate_model_indices.to_numpy(dtype=int)
if not np.equal(candidate_model_indices.to_numpy(dtype=float), candidate_indices).all():
    st.error("Product indices must be integers.")
    st.stop()
if (candidate_indices < 0).any() or (candidate_indices >= Q.shape[0]).any():
    st.error("A product index is outside the collaborative-filtering model.")
    st.stop()

cf_scores = np.asarray([get_cf_score(user_index, product_index) for product_index in candidate_indices])
cf_scores = normalize_scores(cf_scores)
sentiment_by_product = df.groupby("ProductId")["avg_sentiment_score"].mean()
candidate_sentiment = sentiment_by_product.reindex(candidate_ids).to_numpy(dtype=float)
if not np.isfinite(candidate_sentiment).all() or (candidate_sentiment < -1).any() or (candidate_sentiment > 1).any():
    st.error("Product sentiment scores must be finite values between -1 and 1.")
    st.stop()
sentiment_scores = np.asarray(
    (candidate_sentiment + 1) / 2
)

if reference_product is None:
    cf_weight = 0.75
    sentiment_weight = 0.25
    content_weight = 0.0
    content_scores = np.zeros(len(candidate_ids), dtype=float)
else:
    content_scores = get_content_scores(candidate_ids, reference_product)
    if not np.isfinite(content_scores).all():
        st.error("The content-similarity matrix contains non-finite values.")
        st.stop()
    cf_weight = 0.6
    sentiment_weight = 0.2
    content_weight = 0.2

cf_contributions = cf_weight * cf_scores
sentiment_contributions = sentiment_weight * sentiment_scores
content_contributions = content_weight * content_scores
final_scores = cf_contributions + sentiment_contributions + content_contributions

ranked_indices = sorted(
    range(len(candidate_ids)), key=lambda index: final_scores[index], reverse=True
)[:10]
reason_labels = []
for index in ranked_indices:
    contributions = {
        "Personalized fit (simulated model)": cf_contributions[index],
        "Review sentiment (text model)": sentiment_contributions[index],
    }
    if reference_product is not None:
        contributions["Similarity to reference product"] = content_contributions[index]
    reason_labels.append(max(contributions, key=contributions.get))

user_category_ratings = (
    df.loc[df["UserId"] == user_id]
    .groupby("ProductCategory")["Score"]
    .mean()
    .sort_values(ascending=False)
)
if not user_category_ratings.empty:
    preferred_categories = ", ".join(
        f"{category} ({rating:.1f}★)"
        for category, rating in user_category_ratings.head(3).items()
    )
    st.caption(f"Personalized for {user_id}. Highest-rated categories: {preferred_categories}.")

recommended = pd.DataFrame(
    {
        "ProductId": [candidate_ids[index] for index in ranked_indices],
        "Recommendation score": [final_scores[index] for index in ranked_indices],
        "Personalization contribution": [
            cf_contributions[index] for index in ranked_indices
        ],
        "Review sentiment contribution": [
            sentiment_contributions[index] for index in ranked_indices
        ],
        "Reference similarity contribution": [
            content_contributions[index] for index in ranked_indices
        ],
        "Why recommended": reason_labels,
    }
)
catalog_columns = [
    column
    for column in ("ProductId", "product_title", "brand", "product_category", "price")
    if column in product_text.columns
]
recommended = recommended.merge(
    product_text[catalog_columns],
    on="ProductId",
    how="left",
    validate="one_to_one",
)
product_feedback = (
    df.groupby("ProductId", as_index=False)
    .agg(
        Average_Rating=("Score", "mean"),
        Average_Sentiment=("avg_sentiment_score", "mean"),
        Reviews=("Score", "size"),
    )
)
recommended = recommended.merge(
    product_feedback,
    on="ProductId",
    how="left",
    validate="one_to_one",
)
with st.expander("How each recommendation is scored"):
    st.markdown(
        f"""
Each contribution below is the component score multiplied by its weight. The displayed
recommendation score is their sum.

- **Personalized fit:** relative collaborative-filtering score, min-max scaled across
  unseen candidates; weight **{cf_weight:.0%}**.
- **Review sentiment:** average TF-IDF model prediction mapped from -1–+1 to 0–1; weight
  **{sentiment_weight:.0%}**. The classifier was trained on UCI-labeled Amazon product
  reviews; the app's review records remain synthetic.
- **Reference similarity:** product-description similarity to the selected reference
  product; weight **{content_weight:.0%}**.

These weights and the simulator-generated factors are experimental, not learned or
validated for real-world shopping recommendations. "Why recommended" identifies the
largest weighted contribution, not a causal explanation.
"""
    )
st.dataframe(
    recommended,
    hide_index=True,
    use_container_width=True,
    column_config={
        "Recommendation score": st.column_config.ProgressColumn(
            "Recommendation score",
            min_value=0,
            max_value=1,
            format="%.2f",
        ),
        "Personalization contribution": st.column_config.NumberColumn(
            "Personalization impact",
            help="Weighted contribution from the simulated collaborative-filtering score.",
            format="%.3f",
        ),
        "Review sentiment contribution": st.column_config.NumberColumn(
            "Sentiment impact",
            help="Weighted contribution from the synthetic rating-derived sentiment proxy.",
            format="%.3f",
        ),
        "Reference similarity contribution": st.column_config.NumberColumn(
            "Similarity impact",
            help="Weighted contribution from similarity to the selected reference product.",
            format="%.3f",
        ),
        "Average_Sentiment": st.column_config.NumberColumn(
            "Text sentiment",
            format="%.2f",
        ),
        "Average_Rating": st.column_config.NumberColumn(
            "Average rating",
            format="%.2f / 5",
        ),
        "price": st.column_config.NumberColumn("Price", format="$%.2f"),
    },
)
