from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.reviewer_analysis import build_reviewer_features
from src.train_review_sentiment import SCORED_DATA_PATH


st.title("Reviewer signals")
st.caption(
    "Descriptive activity patterns for investigation. These flags do not identify "
    "reviewers as fraudulent or prove coordinated behavior."
)

if not SCORED_DATA_PATH.is_file():
    st.warning(
        "Scored review data is missing. Run "
        "`python -m src.train_review_sentiment` from the project root."
    )
    st.stop()

data_stat = SCORED_DATA_PATH.stat()


@st.cache_data(show_spinner="Calculating reviewer features...")
def load_reviewer_features(data_version: tuple[int, int]) -> pd.DataFrame:
    if not data_version:
        raise ValueError("A data version is required to calculate reviewer features.")
    reviews = pd.read_parquet(SCORED_DATA_PATH)
    reviews["review_time"] = pd.to_datetime(
        reviews["review_time"],
        utc=True,
        errors="coerce",
    )
    reviews = reviews.dropna(subset=["review_time"])
    return build_reviewer_features(reviews)


try:
    features = load_reviewer_features((data_stat.st_mtime_ns, data_stat.st_size))
except (OSError, ValueError, ImportError) as error:
    st.error(f"Could not calculate reviewer features: {error}")
    st.stop()

if features.empty:
    st.info("No reviewer rows are available.")
    st.stop()

total_reviewers = features["reviewer_id"].nunique()
multiple_reviews = int(features["review_count"].gt(1).sum())
reviewer_metrics = st.columns(3)
reviewer_metrics[0].metric("Reviewers", f"{total_reviewers:,}")
reviewer_metrics[1].metric("With multiple reviews", f"{multiple_reviews:,}")
reviewer_metrics[2].metric(
    "Reviewers with out-of-time sentiment",
    f"{features['negative_share'].notna().sum():,}",
)

st.subheader("Activity and repeated text")
left, right = st.columns(2)
left.plotly_chart(
    px.histogram(
        features,
        x="review_rate_per_active_month",
        nbins=40,
        title="Reviews per active month",
        labels={"review_rate_per_active_month": "Reviews / active month"},
        template="plotly_white",
        color_discrete_sequence=["#68766C"],
    ),
    width="stretch",
)
eligible_for_sentiment = features.dropna(subset=["negative_share"])
if eligible_for_sentiment.empty:
    right.info("No reviewers have out-of-time sentiment scores in this run.")
else:
    right.plotly_chart(
        px.scatter(
            eligible_for_sentiment.sample(
                min(8000, len(eligible_for_sentiment)),
                random_state=42,
            ),
            x="review_count",
            y="negative_share",
            size="reused_text_share",
            color="pattern_flags",
            hover_name="reviewer_id",
            title="Review volume and out-of-time negative share",
            labels={
                "review_count": "Reviews",
                "negative_share": "Share predicted negative (scored reviews only)",
                "reused_text_share": "Share of text reused by 3+ reviewers",
                "pattern_flags": "Pattern signals",
            },
            template="plotly_white",
            color_continuous_scale=["#DADDD9", "#68766C", "#343B36"],
        ),
        width="stretch",
    )

st.subheader("Pattern definitions")
st.markdown(
    """
- **High volume:** reviewer is at or above the portfolio's 99th percentile of review count.
- **Unusually negative:** at least 10 reviews and the negative share is more than 40 percentage points above the out-of-time portfolio baseline.
- **Burst activity:** at least 10 reviews and a review rate above the 99th percentile among reviewers with at least 10 reviews.
- **Repeated text:** at least 3 reviews and more than half of the review text is shared with text posted by at least 3 distinct reviewers.
"""
)

st.subheader("Reviewers with the most pattern signals")
flagged = features.loc[features["pattern_flags"].gt(0)].sort_values(
    ["pattern_flags", "review_count"],
    ascending=False,
)
st.dataframe(flagged.head(100), hide_index=True, width="stretch")
st.caption(
    "Negative-share features only use out-of-time predictions. Reviewers with no "
    "scored reviews have a missing negative share, not an assumed non-negative label. "
    "Patterns are leads for further review, not conclusions about intent."
)
