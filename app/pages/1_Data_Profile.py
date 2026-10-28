import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.train_review_sentiment import DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = PROJECT_ROOT / "data" / "review_intelligence_stats.json"

st.title("Dataset and data quality")
st.caption(
    "Amazon Reviews 2023 · All_Beauty · reproducible local sample. "
    "Raw reviews are sampled and processed locally."
)

if not STATS_PATH.is_file() or not DATA_PATH.is_file():
    st.warning(
        "Prepared review data is missing. Run "
        "`python -m src.prepare_review_intelligence` from the project root."
    )
    st.stop()

stats = json.loads(STATS_PATH.read_text(encoding="utf-8"))
try:
    reviews = pd.read_parquet(DATA_PATH)
except (OSError, ValueError, ImportError) as error:
    st.error(f"Could not read prepared review data: {error}")
    st.stop()

reviews["review_time"] = pd.to_datetime(reviews["review_time"], utc=True, errors="coerce")
required_columns = [
    "review_id",
    "product_id",
    "reviewer_id",
    "rating",
    "review_time",
    "title",
    "review_text",
]
missing_columns = set(required_columns).difference(reviews.columns)
if missing_columns:
    st.error(f"Prepared data is missing columns: {', '.join(sorted(missing_columns))}.")
    st.stop()

metrics = st.columns(4)
metrics[0].metric("Source rows", f"{stats['source_rows']:,}")
metrics[1].metric("Valid sampled reviews", f"{len(reviews):,}")
metrics[2].metric("Products", f"{reviews['product_id'].nunique():,}")
metrics[3].metric("Reviewers", f"{reviews['reviewer_id'].nunique():,}")

quality = pd.DataFrame(
    [
        ("Invalid sampled records dropped", stats["dropped_invalid_rows"]),
        ("Duplicate review IDs", int(reviews["review_id"].duplicated().sum())),
        ("Missing required values", int(reviews[required_columns].isna().sum().sum())),
        (
            "Empty review bodies",
            int(reviews["review_text"].fillna("").astype(str).str.strip().eq("").sum()),
        ),
        ("Ratings outside 1–5", int((~reviews["rating"].between(1, 5)).sum())),
        ("Unique review IDs", int(reviews["review_id"].nunique())),
        ("Sampling seed", stats["random_sample_seed"]),
    ],
    columns=["Check", "Observed"],
)

st.subheader("Data quality")
st.dataframe(quality, hide_index=True, width="stretch")
st.caption(
    "The sampled dataset contains unique review IDs and valid 1–5 ratings. "
    "Preparation drops sampled rows without required review fields."
)

st.subheader("Ratings and time coverage")
rating_counts = (
    reviews["rating"].value_counts().sort_index().rename_axis("Stars").reset_index(name="Reviews")
)
monthly_counts = (
    reviews.dropna(subset=["review_time"])
    .assign(Month=lambda frame: frame["review_time"].dt.tz_convert(None).dt.to_period("M").astype(str))
    .groupby("Month", as_index=False)
    .size()
    .rename(columns={"size": "Reviews"})
)
left, right = st.columns(2)
left.plotly_chart(
    px.bar(
        rating_counts,
        x="Stars",
        y="Reviews",
        title="Review counts by star rating",
        color_discrete_sequence=["#68766C"],
        template="plotly_white",
    ),
    width="stretch",
)
right.plotly_chart(
    px.line(
        monthly_counts,
        x="Month",
        y="Reviews",
        title="Review volume by month",
        template="plotly_white",
        color_discrete_sequence=["#68766C"],
    ),
    width="stretch",
)

text_length = reviews["review_text"].fillna("").astype(str).str.len()
profile = pd.DataFrame(
    {
        "Statistic": [
            "First review date (UTC)",
            "Last review date (UTC)",
            "Median review length (characters)",
            "90th percentile review length",
            "95th percentile review length",
            "Ratings with 5 stars",
        ],
        "Value": [
            str(reviews["review_time"].min()),
            str(reviews["review_time"].max()),
            f"{text_length.median():,.0f}",
            f"{text_length.quantile(0.90):,.0f}",
            f"{text_length.quantile(0.95):,.0f}",
            f"{reviews['rating'].eq(5).sum():,} ({reviews['rating'].eq(5).mean():.1%})",
        ],
    }
)
st.dataframe(profile, hide_index=True, width="stretch")

with st.expander("Sample records"):
    st.dataframe(
        reviews[["review_id", "product_id", "rating", "review_time", "title", "review_text"]]
        .head(10),
        hide_index=True,
        width="stretch",
    )
