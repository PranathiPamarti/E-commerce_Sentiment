import json
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning
import warnings

from src.train_review_sentiment import (
    DATA_PATH,
    LABELS,
    METRICS_PATH,
    MODEL_PATH,
    SCORED_DATA_PATH,
    rating_sentiment,
    remove_duplicate_review_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SMALL_METRICS_PATH = PROJECT_ROOT / "models" / "sentiment_metrics.json"
SMALL_MODEL_PATH = PROJECT_ROOT / "models" / "sentiment_classifier.joblib"

st.title("Sentiment model")
st.caption("What is trained, how it is evaluated, and what the saved scores mean.")

if not METRICS_PATH.is_file() or not MODEL_PATH.is_file() or not SCORED_DATA_PATH.is_file():
    st.warning(
        "Main sentiment artifacts are missing. Run "
        "`python -m src.train_review_sentiment` after preparing the dataset."
    )
    st.stop()

try:
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except (OSError, ValueError) as error:
    st.error(f"Could not read sentiment metrics: {error}")
    st.stop()

test = metrics["held_out_test_metrics"]
baseline = metrics["majority_class_baseline_test_metrics"]
neutral_f1 = test["classification_report"]["neutral"]["f1-score"]
summary = st.columns(4)
summary[0].metric("Test rows", f"{metrics['test_rows']:,}")
summary[1].metric("Test accuracy", f"{test['accuracy']:.1%}")
summary[2].metric("Test macro-F1", f"{test['macro_f1']:.3f}")
summary[3].metric("Neutral-class F1", f"{neutral_f1:.3f}")

st.warning(
    "Target labels come from star ratings (1–2 negative, 3 neutral, 4–5 positive). "
    "The scores measure agreement with rating-derived weak labels, not human sentiment truth."
)

st.subheader("Model workflow")
st.markdown(
    """
1. Combine review title and body; remove normalized exact-text duplicates, keeping the earliest review.
2. Sort chronologically and split 60% train / 10% tuning / 10% out-of-time validation / 20% final test.
3. Compare two TF-IDF + balanced SGD configurations (`alpha=0.0001` and `0.001`) on the tuning period using macro-F1.
4. Reuse the selected model to score the separate validation period; refit it on the first 80% and evaluate once on the final 20%.
5. Save one final pipeline, metrics JSON, and a scored Parquet file. Only validation/test-period reviews receive dashboard sentiment scores.
"""
)

split_rows = [
    {"Period": "Train", "Rows": metrics["train_rows"], "Purpose": "Fit candidate text models"},
    {
        "Period": "Tuning validation",
        "Rows": metrics["hyperparameter_selection_rows"],
        "Purpose": "Select regularization strength",
    },
    {
        "Period": "Out-of-time validation",
        "Rows": metrics["validation_rows"],
        "Purpose": "Measure later-period model behavior and dashboard scores",
    },
    {"Period": "Held-out test", "Rows": metrics["test_rows"], "Purpose": "Final evaluation"},
]
if DATA_PATH.is_file():
    source_stat = DATA_PATH.stat()

    @st.cache_data(show_spinner="Calculating time boundaries...")
    def split_date_ranges(data_version: tuple[int, int]) -> dict[str, tuple[str, str]]:
        if not data_version:
            raise ValueError("A prepared-data version is required to calculate split dates.")
        frame = pd.read_parquet(
            DATA_PATH,
            columns=["review_id", "rating", "review_time", "title", "review_text"],
        )
        frame["review_time"] = pd.to_datetime(
            frame["review_time"], utc=True, errors="coerce"
        )
        frame["label"] = rating_sentiment(frame["rating"])
        frame = frame.dropna(subset=["label", "review_time", "review_text"])
        frame["label"] = frame["label"].astype(int)
        frame["model_text"] = (
            frame["title"].fillna("").astype(str).str.strip()
            + ". "
            + frame["review_text"].fillna("").astype(str).str.strip()
        ).str.strip()
        frame = frame.loc[frame["model_text"].str.len().gt(0)]
        frame, _ = remove_duplicate_review_text(frame)
        boundaries = (0, 0.6, 0.7, 0.8, 1)
        names = (
            "Train",
            "Tuning validation",
            "Out-of-time validation",
            "Held-out test",
        )
        dates = {}
        for name, lower, upper in zip(names, boundaries[:-1], boundaries[1:]):
            start = int(len(frame) * lower)
            end = int(len(frame) * upper)
            period = frame.iloc[start:end]["review_time"]
            dates[name] = (
                period.min().strftime("%Y-%m-%d"),
                period.max().strftime("%Y-%m-%d"),
            )
        return dates

    try:
        split_dates = split_date_ranges((source_stat.st_mtime_ns, source_stat.st_size))
        for row in split_rows:
            row["Date range"] = " to ".join(split_dates[row["Period"]])
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Could not calculate chronological split dates: {error}")
st.dataframe(pd.DataFrame(split_rows), hide_index=True, width="stretch")
st.caption(
    f"{metrics['duplicate_text_rows_removed_before_split']:,} normalized exact-text "
    "duplicate rows were removed before splitting. The saved prediction coverage is "
    f"{metrics['out_of_time_prediction_counts']['total']:,} out-of-time reviews."
)

st.subheader("What is the model?")
model_cols = st.columns(2)
with model_cols[0]:
    st.markdown("**Text features**")
    st.write(
        "TF-IDF word unigrams and bigrams; lowercase text, Unicode accent "
        "normalization, minimum document frequency 2, and a 100,000-feature cap."
    )
with model_cols[1]:
    st.markdown("**Classifier**")
    st.write(
        "SGDClassifier with logistic loss and balanced class weights. Logistic "
        "loss provides class probabilities; weights help address the five-star "
        "label imbalance."
    )

tuning = metrics["validation_hyperparameter_candidates"]["metrics_by_alpha"]
tuning_frame = pd.DataFrame.from_dict(tuning, orient="index")
tuning_frame.index.name = "SGD alpha"
tuning_frame["selected"] = tuning_frame.index == str(
    metrics["validation_hyperparameter_candidates"]["selected_alpha"]
)
st.markdown("**Tuning candidates**")
st.dataframe(
    tuning_frame[["accuracy", "macro_f1", "selected"]].style.format(
        {"accuracy": "{:.1%}", "macro_f1": "{:.3f}"}
    ),
    width="stretch",
)

st.subheader("Held-out comparison")
comparison = pd.DataFrame(
    {
        "Model": ["TF-IDF + balanced SGD", "Majority-class baseline"],
        "Accuracy": [test["accuracy"], baseline["accuracy"]],
        "Macro-F1": [test["macro_f1"], baseline["macro_f1"]],
    }
)
left, right = st.columns(2)
left.plotly_chart(
    px.bar(
        comparison,
        x="Model",
        y="Accuracy",
        title="Accuracy",
        text_auto=".1%",
        template="plotly_white",
        color="Model",
        color_discrete_sequence=["#66756B", "#C7CBC7"],
    ).update_layout(showlegend=False),
    width="stretch",
)
right.plotly_chart(
    px.bar(
        comparison,
        x="Model",
        y="Macro-F1",
        title="Macro-F1",
        text_auto=".3f",
        template="plotly_white",
        color="Model",
        color_discrete_sequence=["#66756B", "#C7CBC7"],
    ).update_layout(showlegend=False),
    width="stretch",
)

report = pd.DataFrame(test["classification_report"]).T
st.markdown("**Per-class test metrics**")
st.dataframe(
    report.loc[["negative", "neutral", "positive"], ["precision", "recall", "f1-score", "support"]]
    .rename_axis("Rating-derived class")
    .style.format(
        {"precision": "{:.1%}", "recall": "{:.1%}", "f1-score": "{:.3f}", "support": "{:,.0f}"}
    ),
    width="stretch",
)

confusion = test["confusion_matrix_negative_neutral_positive"]
figure = ff.create_annotated_heatmap(
    z=confusion,
    x=["Negative", "Neutral", "Positive"],
    y=["Negative", "Neutral", "Positive"],
    colorscale=[[0, "#F3F4F2"], [1, "#66756B"]],
    showscale=True,
)
figure.update_layout(
    title="Held-out test confusion matrix",
    xaxis_title="Predicted label",
    yaxis_title="Rating-derived label",
    template="plotly_white",
)
st.plotly_chart(figure, width="stretch")

st.subheader("Two saved sentiment pipelines—not one combined model")
st.write(
    "The main All_Beauty model above powers the review-intelligence analysis. "
    "A separate binary TF-IDF + logistic-regression model is retained for the "
    "synthetic demo screens; it was trained on 990 deduplicated labeled UCI "
    "cell-phone review sentences, and its held-out test has 198 examples."
)
if SMALL_METRICS_PATH.is_file():
    small_metrics = json.loads(SMALL_METRICS_PATH.read_text(encoding="utf-8"))
    small_test = small_metrics["held_out_test_metrics"]
    small_columns = st.columns(3)
    small_columns[0].metric("Legacy demo test accuracy", f"{small_test['accuracy']:.1%}")
    small_columns[1].metric("Legacy demo macro-F1", f"{small_test['macro_f1']:.3f}")
    small_columns[2].metric("Legacy test size", f"{small_metrics['split_rows']['test']:,}")
    if SMALL_MODEL_PATH.is_file():
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", InconsistentVersionWarning)
            joblib.load(SMALL_MODEL_PATH)
        if any(issubclass(warning.category, InconsistentVersionWarning) for warning in caught):
            st.warning(
                "The saved legacy model was created with a different scikit-learn "
                "version. Retrain it with `python -m src.train_sentiment_model` "
                "before relying on its predictions."
            )

st.subheader("Try a review")
st.caption(
    "This demo uses the saved All_Beauty model. Its displayed label is a model "
    "prediction against rating-derived training targets, not a human verdict."
)
review_text = st.text_area(
    "Review text",
    placeholder="Paste a short review to inspect the model's prediction.",
    height=120,
)
if st.button("Predict rating-derived sentiment", disabled=not review_text.strip()):
    try:
        sentiment_model = joblib.load(MODEL_PATH)
        probabilities = sentiment_model.predict_proba([review_text.strip()])[0]
        predicted_class = int(sentiment_model.classes_[probabilities.argmax()])
        predicted_probability = float(probabilities.max())
        result_columns = st.columns(2)
        result_columns[0].metric("Predicted class", LABELS[predicted_class])
        result_columns[1].metric("Predicted class probability", f"{predicted_probability:.1%}")
        probability_frame = pd.DataFrame(
            {
                "Class": [LABELS[int(class_id)] for class_id in sentiment_model.classes_],
                "Probability": probabilities,
            }
        )
        st.plotly_chart(
            px.bar(
                probability_frame,
                x="Class",
                y="Probability",
                title="Model class probabilities",
                template="plotly_white",
                color_discrete_sequence=["#68766C"],
            ),
            width="stretch",
        )
    except (OSError, ValueError, ImportError) as error:
        st.error(f"Could not load or run the saved sentiment model: {error}")

with st.expander("Further validation needed"):
    st.write(
        "The next model-quality step is to label a representative sentiment sample "
        "independently of stars, then report human-reference per-class metrics. "
        "The neutral F1 of 0.358 indicates this class needs particular attention. "
        "Do not interpret high overall accuracy without the baseline, macro-F1, "
        "and class-level results."
    )
