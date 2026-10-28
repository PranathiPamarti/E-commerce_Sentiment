from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from src.issue_tagging import TAG_DESCRIPTIONS, has_issue_tags
from src.train_review_sentiment import SCORED_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATS_PATH = PROJECT_ROOT / "data" / "review_intelligence_stats.json"


@st.cache_data(show_spinner="Loading prepared review data...")
def load_reviews(data_version):
    if not data_version:
        raise ValueError("A data version is required to load prepared reviews.")
    return pd.read_parquet(SCORED_DATA_PATH)


def issue_rows(reviews):
    reviews = reviews.copy()
    reviews["issue_tags"] = reviews["issue_tags"].map(
        lambda value: value.tolist() if hasattr(value, "tolist") else value
    )
    rows = reviews.explode("issue_tags").dropna(subset=["issue_tags"]).copy()
    rows["issue_description"] = rows["issue_tags"].map(TAG_DESCRIPTIONS)
    return rows


st.title("Issue and product intelligence")
st.caption(
    "Explore transparent issue-tag candidates, changing issue rates, and product-level "
    "review patterns. Tags are not human-validated defects."
)

if not SCORED_DATA_PATH.is_file() or not STATS_PATH.is_file():
    st.info(
        "Prepare and score the public dataset first. From the project root run "
        "`python -m src.prepare_review_intelligence`, then "
        "`python -m src.train_review_sentiment`."
    )
    st.stop()
stats = STATS_PATH.stat()
data_version = (
    stats.st_mtime_ns,
    stats.st_size,
    SCORED_DATA_PATH.stat().st_mtime_ns,
)
try:
    df = load_reviews(data_version)
except (OSError, ValueError, ImportError) as error:
    st.error(f"Could not load prepared review data: {error}")
    st.stop()

if df.empty:
    st.error("Prepared review data contains no rows.")
    st.stop()
required_scoring_columns = {
    "sentiment_is_out_of_sample",
    "sentiment_prediction_split",
}
missing_scoring_columns = required_scoring_columns.difference(df.columns)
if missing_scoring_columns:
    st.error(
        "The scored data predates the out-of-time sentiment evaluation. Run "
        "`python -m src.train_review_sentiment` to regenerate it."
    )
    st.stop()

df["review_time"] = pd.to_datetime(df["review_time"], utc=True, errors="coerce")
df = df.dropna(subset=["review_time"])
product_filter, issue_filter = st.columns(2)
selected_product = product_filter.text_input(
    "Filter to a product ID",
    help="Leave blank to analyze the full product portfolio.",
)
issue_options = ["All issues", *sorted(TAG_DESCRIPTIONS)]
selected_issue = issue_filter.selectbox("Issue tag", issue_options)

filtered = df
if selected_product.strip():
    filtered = filtered.loc[
        filtered["product_id"].astype(str).str.contains(
            selected_product.strip(),
            case=False,
            regex=False,
        )
    ]
if selected_issue != "All issues":
    filtered = filtered.loc[
        filtered["issue_tags"].map(lambda tags: selected_issue in tags)
    ]
if filtered.empty:
    st.info("No review records match these filters.")
    st.stop()

total_products = filtered["product_id"].nunique()
total_reviewers = filtered["reviewer_id"].nunique()
tagged_count = int(filtered["issue_tags"].map(has_issue_tags).sum())
out_of_time_filtered = filtered.loc[filtered["sentiment_is_out_of_sample"]]
out_of_time_count = len(out_of_time_filtered)
latest_time = filtered["review_time"].max()
kpis = st.columns(3)
kpis[0].metric("Reviews", f"{len(filtered):,}")
kpis[1].metric("Products", f"{total_products:,}")
kpis[2].metric("Reviewers", f"{total_reviewers:,}")
context_metrics = st.columns(3)
context_metrics[0].metric("Candidate-tagged reviews", f"{tagged_count:,}")
context_metrics[1].metric("Out-of-time sentiment scores", f"{out_of_time_count:,}")
context_metrics[2].metric("Latest review", latest_time.strftime("%Y-%m-%d"))

with st.expander("How to interpret this page"):
    st.write(
        "Sentiment scores on product charts use only out-of-time validation/test "
        "predictions. They were trained against star-rating-derived labels, not "
        "independent human sentiment labels. Keyword issue tags are exploratory "
        "candidates; confirm them in the Human issue-tag audit before calling them "
        "validated issues."
    )

st.subheader("Out-of-time sentiment and review volume")
sentiment_counts = (
    out_of_time_filtered["sentiment_label"]
    .value_counts()
    .rename_axis("Sentiment")
    .reset_index(name="Reviews")
)
volume_by_month = (
    filtered.assign(
        Month=filtered["review_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    .groupby("Month", as_index=False)
    .agg(
        Reviews=("review_id", "size"),
        Average_Rating=("rating", "mean"),
    )
)
sentiment_by_month = (
    out_of_time_filtered.assign(
        Month=out_of_time_filtered["review_time"]
        .dt.tz_convert(None)
        .dt.to_period("M")
        .astype(str)
    )
    .groupby("Month", as_index=False)
    .agg(
        Average_Sentiment=("sentiment_score", "mean"),
        Sentiment_Reviews=("review_id", "size"),
    )
)
ratings = (
    filtered["rating"]
    .value_counts()
    .sort_index()
    .rename_axis("Rating")
    .reset_index(name="Reviews")
)
left, right = st.columns(2)
if out_of_time_count:
    left.plotly_chart(
        px.pie(
            sentiment_counts,
            names="Sentiment",
            values="Reviews",
            title="Out-of-time predicted sentiment mix",
            hole=0.4,
        ),
        width="stretch",
    )
else:
    left.info("No out-of-time sentiment predictions match the current filters.")
right.plotly_chart(
    px.bar(ratings, x="Rating", y="Reviews", title="Review distribution by star rating"),
    width="stretch",
)
left, right = st.columns(2)
left.plotly_chart(
    px.line(
        volume_by_month,
        x="Month",
        y="Reviews",
        markers=True,
        title="Monthly review volume",
    ),
    width="stretch",
)
if out_of_time_count:
    right.plotly_chart(
        px.line(
            sentiment_by_month,
            x="Month",
            y="Average_Sentiment",
            markers=True,
            title="Monthly out-of-time sentiment shifts",
            hover_data={"Sentiment_Reviews": True},
        ).update_yaxes(range=[-1, 1]),
        width="stretch",
    )
    st.caption(
        "Sentiment charts use only validation/test-period predictions produced "
        "after their corresponding training cutoffs; training-period and duplicate "
        "training-text rows are unscored. Review-volume and issue charts use all "
        "filtered reviews."
    )

tagged = issue_rows(filtered)
st.subheader("Recurring and emerging product issues")
if tagged.empty:
    st.info("No issue-tag candidates were found for the current filters.")
else:
    tag_counts = (
        tagged.groupby(["issue_tags", "issue_description"], as_index=False)
        .agg(
            Reviews=("review_id", "nunique"),
            Products=("product_id", "nunique"),
        )
        .sort_values("Reviews", ascending=False)
    )
    recurring_candidate_count = int(
        (tag_counts["Reviews"].ge(25) & tag_counts["Products"].ge(3)).sum()
    )
    st.metric(
        "Recurring candidate tag types",
        recurring_candidate_count,
        help="At least 25 candidate-tagged reviews across at least 3 products; not human-validated.",
    )
    left, right = st.columns(2)
    left.plotly_chart(
        px.bar(
            tag_counts.head(20).sort_values("Reviews"),
            x="Reviews",
            y="issue_description",
            orientation="h",
            title="Most common candidate pain points",
            hover_data={"Products": True},
            labels={"issue_description": "Candidate issue"},
        ),
        width="stretch",
    )

    top_tags = tag_counts.head(12)["issue_tags"].tolist()
    issue_month = (
        tagged.loc[tagged["issue_tags"].isin(top_tags)]
        .assign(
            Month=lambda frame: frame["review_time"]
            .dt.tz_convert(None)
            .dt.to_period("M")
            .astype(str)
        )
        .groupby(["issue_tags", "Month"])
        .size()
        .rename("Reviews")
        .reset_index()
    )
    monthly_total = (
        filtered.assign(
            Month=lambda frame: frame["review_time"]
            .dt.tz_convert(None)
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month")
        .size()
        .rename("Portfolio_Reviews")
    )
    issue_month["Issue_Rate_Per_1000"] = (
        issue_month["Reviews"] / issue_month["Month"].map(monthly_total) * 1000
    )
    heatmap = issue_month.pivot(
        index="issue_tags",
        columns="Month",
        values="Issue_Rate_Per_1000",
    ).fillna(0)
    right.plotly_chart(
        px.imshow(
            heatmap,
            aspect="auto",
            labels={"x": "Month", "y": "Candidate issue", "color": "Per 1,000 reviews"},
            title="Candidate issue rate over time",
        ),
        width="stretch",
    )

    recent_start = latest_time - pd.Timedelta(days=90)
    baseline_start = latest_time - pd.Timedelta(days=180)
    recent_review_count = int(filtered["review_time"].ge(recent_start).sum())
    baseline_review_count = int(
        (
            filtered["review_time"].ge(baseline_start)
            & filtered["review_time"].lt(recent_start)
        ).sum()
    )
    recent = tagged.loc[tagged["review_time"].ge(recent_start)].groupby("issue_tags").size()
    baseline = tagged.loc[
        tagged["review_time"].ge(baseline_start)
        & tagged["review_time"].lt(recent_start)
    ].groupby("issue_tags").size()
    emerging = pd.concat(
        [recent.rename("Recent_90d"), baseline.rename("Previous_90d")],
        axis=1,
    ).fillna(0)
    emerging = emerging.loc[
        emerging["Recent_90d"].ge(5) & emerging["Previous_90d"].ge(5)
    ].copy()
    if recent_review_count and baseline_review_count:
        emerging["Recent_Rate_Per_1000"] = (
            emerging["Recent_90d"] / recent_review_count * 1000
        )
        emerging["Previous_Rate_Per_1000"] = (
            emerging["Previous_90d"] / baseline_review_count * 1000
        )
        emerging["Change_Percent"] = (
            (emerging["Recent_Rate_Per_1000"] - emerging["Previous_Rate_Per_1000"])
            / emerging["Previous_Rate_Per_1000"]
            * 100
        )
    else:
        emerging["Recent_Rate_Per_1000"] = 0.0
        emerging["Previous_Rate_Per_1000"] = 0.0
        emerging["Change_Percent"] = 0.0
    emerging["Issue"] = emerging.index.map(TAG_DESCRIPTIONS)
    emerging = emerging.sort_values("Change_Percent", ascending=False).head(15)
    st.plotly_chart(
        px.bar(
            emerging.sort_values("Change_Percent"),
            x="Change_Percent",
            y="Issue",
            orientation="h",
            title="Candidate issues increasing in the latest 90 days",
            labels={"Change_Percent": "Change in issue rate vs previous 90 days (%)"},
            hover_data={
                "Recent_90d": True,
                "Previous_90d": True,
                "Recent_Rate_Per_1000": ":.2f",
                "Previous_Rate_Per_1000": ":.2f",
            },
        ),
        width="stretch",
    )
    st.caption(
        "Issue tags are transparent keyword-based candidates. Rising counts are signals "
        "for human investigation, not confirmed defects."
    )

st.subheader("Product comparisons and drill-down")
product_stats = (
    filtered.groupby("product_id", as_index=False)
    .agg(
        Reviews=("review_id", "size"),
        Average_Rating=("rating", "mean"),
    )
)
product_sentiment = (
    out_of_time_filtered.groupby("product_id")["sentiment_score"]
    .mean()
    .rename("Average_Sentiment")
    .reset_index()
)
product_stats = product_stats.merge(
    product_sentiment,
    on="product_id",
    how="left",
    validate="one_to_one",
)
product_issue_stats = (
    tagged.groupby("product_id")
    .agg(
        Candidate_Issue_Mentions=("review_id", "size"),
        Candidate_Issue_Reviews=("review_id", "nunique"),
    )
    .reset_index()
)
product_stats = product_stats.merge(
    product_issue_stats,
    on="product_id",
    how="left",
    validate="one_to_one",
)
product_stats[["Candidate_Issue_Mentions", "Candidate_Issue_Reviews"]] = product_stats[
    ["Candidate_Issue_Mentions", "Candidate_Issue_Reviews"]
].fillna(0)
product_stats["Issue_Rate"] = (
    product_stats["Candidate_Issue_Reviews"] / product_stats["Reviews"]
)
minimum_product_reviews = st.slider(
    "Minimum product reviews for comparison",
    min_value=1,
    max_value=max(1, int(product_stats["Reviews"].max())),
    value=min(10, int(product_stats["Reviews"].max())),
    help="Filter low-volume products before comparing candidate issue rates.",
)
eligible_product_stats = product_stats.loc[
    product_stats["Reviews"].ge(minimum_product_reviews)
]
if eligible_product_stats.empty:
    st.info("No products meet this review-count threshold.")
    eligible_product_stats = product_stats
left, right = st.columns(2)
left.plotly_chart(
    px.scatter(
        eligible_product_stats.sample(
            min(5000, len(eligible_product_stats)),
            random_state=42,
        ),
        x="Reviews",
        y="Issue_Rate",
        color="Average_Sentiment",
        hover_name="product_id",
        title="Product issue rate vs. review volume",
        labels={
            "Issue_Rate": "Share of reviews with candidate issue tags",
            "Average_Sentiment": "Mean out-of-time sentiment score",
        },
    ),
    width="stretch",
)
right.plotly_chart(
    px.bar(
        eligible_product_stats.nlargest(15, "Candidate_Issue_Reviews").sort_values(
            "Candidate_Issue_Reviews"
        ),
        x="Candidate_Issue_Reviews",
        y="product_id",
        orientation="h",
        title="Products with most issue-tagged reviews",
        labels={
            "Candidate_Issue_Reviews": "Tagged reviews",
            "product_id": "Product",
        },
    ),
    width="stretch",
)

st.subheader("Review-level investigation")
if not product_stats.empty:
    chosen_product = st.text_input(
        "Enter a product ID to inspect",
        help="Copy a product ID from the product comparison table.",
        key="review_detail_product",
    )
    if chosen_product.strip():
        selected_reviews = filtered.loc[
            filtered["product_id"].astype(str).eq(chosen_product.strip())
        ]
        if selected_reviews.empty:
            st.info("No matching product ID in the filtered review set.")
        else:
            display_columns = [
                "review_time",
                "rating",
                "sentiment_label",
                "issue_tags",
                "reviewer_id",
                "title",
                "review_text",
            ]
            st.dataframe(
                selected_reviews[display_columns]
                .sort_values("review_time", ascending=False)
                .head(100),
                hide_index=True,
                width="stretch",
            )
