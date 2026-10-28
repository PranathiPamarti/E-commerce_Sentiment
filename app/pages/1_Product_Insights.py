import streamlit as st
import plotly.express as px
from pathlib import Path
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from src.review_data import load_review_data, review_data_version


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@st.cache_data(show_spinner=True)
def load_data(data_version):
    return load_review_data(PROJECT_ROOT)


st.title("Product insights · synthetic")
st.caption(
    "Review records are synthetic. Text sentiment is predicted by a TF-IDF classifier "
    "trained on labeled UCI Amazon product reviews; the sample review data is not real customer feedback."
)
try:
    df, products = load_data(review_data_version(PROJECT_ROOT))
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

categories = sorted(df["ProductCategory"].dropna().unique())
selected_categories = st.multiselect(
    "Product categories",
    categories,
    default=categories,
    help="Choose one or more categories to update every chart.",
)
filtered = df[df["ProductCategory"].isin(selected_categories)].copy()

if filtered.empty:
    st.info("Select at least one category to display insights.")
    st.stop()

total_reviews = len(filtered)
average_rating = filtered["Score"].mean()
average_sentiment = filtered["avg_sentiment_score"].mean()
positive_share = filtered["sentiment_value"].mean() * 100
unique_products = filtered["ProductId"].nunique()
unique_users = filtered["UserId"].nunique()

metric_columns = st.columns(5)
metric_columns[0].metric("Reviews", f"{total_reviews:,}")
metric_columns[1].metric("Products", f"{unique_products:,}")
metric_columns[2].metric("Reviewers", f"{unique_users:,}")
metric_columns[3].metric("Average rating", f"{average_rating:.2f} / 5")
metric_columns[4].metric("Positive text sentiment", f"{positive_share:.1f}%")

st.caption(f"Average review sentiment: {average_sentiment:+.2f} on a -1 to +1 scale.")

left, right = st.columns(2)
with left:
    st.subheader("Ratings")
    rating_counts = (
        filtered["Score"]
        .value_counts()
        .reindex([1, 2, 3, 4, 5], fill_value=0)
        .rename_axis("Rating")
        .reset_index(name="Reviews")
    )
    rating_chart = px.bar(
        rating_counts,
        x="Rating",
        y="Reviews",
        title="Review count by star rating",
        color_discrete_sequence=["#68766C"],
    )
    st.plotly_chart(rating_chart, use_container_width=True)

with right:
    st.subheader("Category comparison")
    category_stats = (
        filtered.groupby("ProductCategory", as_index=False)
        .agg(
            Average_Rating=("Score", "mean"),
            Reviews=("Score", "size"),
            Average_Sentiment=("avg_sentiment_score", "mean"),
        )
        .sort_values("Average_Rating", ascending=True)
    )
    category_chart = px.bar(
        category_stats,
        x="Average_Rating",
        y="ProductCategory",
        orientation="h",
        color="Average_Sentiment",
        color_continuous_scale="Greys",
        hover_data={"Reviews": True, "Average_Rating": ":.2f", "Average_Sentiment": ":.2f"},
        title="Average rating by category",
        labels={"Average_Rating": "Average rating", "ProductCategory": ""},
    )
    category_chart.update_xaxes(range=[0, 5])
    st.plotly_chart(category_chart, use_container_width=True)

if "Time" in filtered and filtered["Time"].notna().any():
    st.subheader("Review activity over time")
    monthly = (
        filtered.dropna(subset=["Time"])
        .assign(
            Month=lambda frame: frame["Time"]
            .dt.tz_convert(None)
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("Month", as_index=False)
        .agg(
            Reviews=("Score", "size"),
            Average_Rating=("Score", "mean"),
            Average_Sentiment=("avg_sentiment_score", "mean"),
        )
    )
    timeline = px.line(
        monthly,
        x="Month",
        y="Reviews",
        markers=True,
        hover_data={"Average_Rating": ":.2f", "Average_Sentiment": ":.2f"},
        title="Monthly review volume",
    )
    st.plotly_chart(timeline, use_container_width=True)

product_stats = (
    filtered.groupby("ProductId", as_index=False)
    .agg(
        Reviews=("Score", "size"),
        Average_Rating=("Score", "mean"),
        Average_Sentiment=("avg_sentiment_score", "mean"),
    )
    .merge(
        products[["ProductId", "product_title", "brand", "product_category", "price"]],
        on="ProductId",
        how="left",
        validate="one_to_one",
    )
)

left, right = st.columns(2)
with left:
    st.subheader("Most reviewed products")
    popular = product_stats.nlargest(10, "Reviews").sort_values("Reviews")
    popular_chart = px.bar(
        popular,
        x="Reviews",
        y="product_title",
        orientation="h",
        color="Average_Rating",
        color_continuous_scale="Greys",
        hover_data={"brand": True, "price": ":$.2f", "Average_Rating": ":.2f"},
        title="Top 10 by review volume",
    )
    st.plotly_chart(popular_chart, use_container_width=True)

with right:
    st.subheader("Product performance")
    minimum_reviews = st.slider(
        "Minimum reviews per product",
        min_value=1,
        max_value=max(1, int(product_stats["Reviews"].max())),
        value=min(20, int(product_stats["Reviews"].max())),
        help="Hide products with too few reviews for a useful comparison.",
    )
    rank_direction = st.radio(
        "Products to compare",
        ["Highest rated", "Lowest rated"],
        horizontal=True,
    )
    eligible_products = product_stats[product_stats["Reviews"] >= minimum_reviews].copy()
    if eligible_products.empty:
        st.info("No products meet this review-count threshold.")
    else:
        prior_weight = 20
        overall_rating = filtered["Score"].mean()
        eligible_products["Adjusted_Rating"] = (
            eligible_products["Reviews"] * eligible_products["Average_Rating"]
            + prior_weight * overall_rating
        ) / (eligible_products["Reviews"] + prior_weight)
        ranked_products = eligible_products.nlargest(
            10, "Adjusted_Rating"
        ) if rank_direction == "Highest rated" else eligible_products.nsmallest(
            10, "Adjusted_Rating"
        )
        ranked_products = ranked_products.sort_values("Adjusted_Rating")
        performance_chart = px.bar(
            ranked_products,
            x="Adjusted_Rating",
            y="product_title",
            orientation="h",
            color="Average_Sentiment",
            color_continuous_scale="Greys",
            range_color=[-1, 1],
            hover_data={
                "brand": True,
                "Reviews": True,
                "Average_Rating": ":.2f",
                "Adjusted_Rating": ":.2f",
                "Average_Sentiment": ":.2f",
                "price": ":$.2f",
            },
            title=f"Top 10 {rank_direction.lower()} products (review-count adjusted)",
            labels={
                "Adjusted_Rating": "Adjusted rating (out of 5)",
                "product_title": "",
                "Average_Sentiment": "Text sentiment",
            },
        )
        performance_chart.update_xaxes(range=[0, 5])
        performance_chart.update_layout(yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(performance_chart, use_container_width=True)
        st.caption(
            "Adjusted rating balances each product's average with the overall average "
            "using a 20-review prior. Hover for the raw rating, review count, and text sentiment."
        )

st.subheader("Review text")
text_column = "clean_text" if "clean_text" in filtered.columns else "Text"
review_text = filtered[text_column].dropna().astype(str).str.strip()
review_text = review_text[review_text.ne("")]
if review_text.empty:
    st.info("There is no review text available for the selected categories.")
else:
    sample_text = " ".join(review_text.sample(min(2000, len(review_text)), random_state=42))
    wordcloud = WordCloud(
        width=1000,
        height=450,
        background_color="white",
        colormap="Greys",
        max_words=120,
    ).generate(sample_text)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wordcloud.to_image(), interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.subheader("Product detail")
product_options = product_stats.sort_values("product_title")
selected_product = st.selectbox(
    "Choose a product",
    product_options["ProductId"].tolist(),
    format_func=lambda product_id: product_options.loc[
        product_options["ProductId"] == product_id, "product_title"
    ].iloc[0],
)
selected_reviews = filtered[filtered["ProductId"] == selected_product]
selected_product_info = products.loc[products["ProductId"] == selected_product].iloc[0]
st.markdown(
    f"**{selected_product_info['product_title']}** · "
    f"{selected_product_info['brand']} · "
    f"{selected_product_info['product_category'].title()} · "
    f"${selected_product_info['price']:.2f}"
)
st.dataframe(
    selected_reviews[
        ["Time", "Score", "Summary", "Text", "VerifiedPurchase", "HelpfulVotes"]
    ].sort_values("Time", ascending=False).head(10),
    hide_index=True,
    use_container_width=True,
)
