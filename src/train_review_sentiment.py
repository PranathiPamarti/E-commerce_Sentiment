"""Train a three-class review-text sentiment model from Amazon star ratings."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "review_intelligence.parquet"
SCORED_DATA_PATH = PROJECT_ROOT / "data" / "review_intelligence_scored.parquet"
MODEL_PATH = PROJECT_ROOT / "models" / "review_sentiment_pipeline.joblib"
METRICS_PATH = PROJECT_ROOT / "models" / "review_sentiment_metrics.json"
RANDOM_STATE = 42
LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}


def build_pipeline(alpha: float = 1e-4) -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=100_000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    alpha=alpha,
                    max_iter=25,
                    tol=1e-3,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def rating_sentiment(ratings: pd.Series) -> pd.Series:
    """Map star ratings to weak training labels, excluding invalid values."""
    numeric = pd.to_numeric(ratings, errors="coerce")
    labels = numeric.map(
        lambda rating: (
            0 if rating <= 2 else 1 if rating == 3 else 2
        )
        if pd.notna(rating) and 1 <= rating <= 5
        else None
    )
    return pd.Series(labels, index=ratings.index, dtype="Int8")


def _classification_metrics(labels: pd.Series, predictions) -> dict:
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix_negative_neutral_positive": confusion_matrix(
            labels, predictions, labels=[0, 1, 2]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1, 2],
            target_names=["negative", "neutral", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }


def _normalize_model_text(text: pd.Series) -> pd.Series:
    return text.str.casefold().str.replace(r"\s+", " ", regex=True).str.strip()


def remove_duplicate_review_text(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep the earliest normalized exact title+body text and report removed rows."""
    ordered = frame.sort_values(["review_time", "review_id"]).copy()
    text_key = _normalize_model_text(ordered["model_text"])
    duplicate_count = int(text_key.duplicated(keep="first").sum())
    ordered["model_text_key"] = text_key
    deduplicated = ordered.drop_duplicates("model_text_key", keep="first")
    return deduplicated, duplicate_count


def train_and_score(
    data_path: Path = DATA_PATH,
    scored_data_path: Path = SCORED_DATA_PATH,
) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Prepared reviews not found at {data_path}. "
            "Run `python -m src.prepare_review_intelligence` first."
        )

    frame = pd.read_parquet(data_path)
    required = {"review_id", "rating", "review_time", "title", "review_text"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Prepared review data is missing columns: {', '.join(sorted(missing))}.")
    frame["label"] = rating_sentiment(pd.to_numeric(frame["rating"], errors="coerce"))
    frame = frame.dropna(subset=["label", "review_time", "review_text"])
    frame["label"] = frame["label"].astype(int)
    frame["model_text"] = (
        frame["title"].fillna("").astype(str).str.strip()
        + ". "
        + frame["review_text"].fillna("").astype(str).str.strip()
    ).str.strip()
    frame = frame.loc[frame["model_text"].str.len().gt(0)]
    frame, duplicate_text_rows = remove_duplicate_review_text(frame)
    train_end = int(len(frame) * 0.6)
    selection_end = int(len(frame) * 0.7)
    validation_end = int(len(frame) * 0.8)
    train = frame.iloc[:train_end]
    selection_validation = frame.iloc[train_end:selection_end]
    validation = frame.iloc[selection_end:validation_end]
    train_validation = frame.iloc[:validation_end]
    test = frame.iloc[validation_end:]
    if any(
        split["label"].nunique() != 3
        for split in (train, selection_validation, validation, test)
    ):
        raise ValueError(
            "All chronological train, tuning, validation, and test splits must "
            "contain all three rating labels."
        )

    candidate_alphas = (1e-4, 1e-3)
    validation_metrics = {}
    validation_model = None
    selected_alpha = None
    best_validation_key = None
    for alpha in candidate_alphas:
        candidate = build_pipeline(alpha=alpha)
        candidate.fit(train["model_text"], train["label"])
        validation_predictions = candidate.predict(selection_validation["model_text"])
        validation_metrics[str(alpha)] = _classification_metrics(
            selection_validation["label"],
            validation_predictions,
        )
        validation_key = (
            validation_metrics[str(alpha)]["macro_f1"],
            validation_metrics[str(alpha)]["accuracy"],
        )
        if best_validation_key is None or validation_key > best_validation_key:
            validation_model = candidate
            selected_alpha = alpha
            best_validation_key = validation_key

    if validation_model is None or selected_alpha is None:
        raise RuntimeError("Hyperparameter selection did not produce a model.")
    validation_predictions = validation_model.predict(validation["model_text"])
    out_of_time_validation_metrics = _classification_metrics(
        validation["label"],
        validation_predictions,
    )
    model = build_pipeline(alpha=selected_alpha)
    model.fit(train_validation["model_text"], train_validation["label"])
    test_predictions = model.predict(test["model_text"])
    metrics = _classification_metrics(test["label"], test_predictions)
    majority_prediction = int(train_validation["label"].value_counts().idxmax())
    majority_predictions = pd.Series(majority_prediction, index=test.index)
    majority_baseline = _classification_metrics(test["label"], majority_predictions)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    validation_ids = set(validation["review_id"].astype(int))
    test_ids = set(test["review_id"].astype(int))
    training_text_keys = set(train["model_text_key"])
    train_validation_text_keys = set(train_validation["model_text_key"])

    scored_data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = scored_data_path.with_suffix(scored_data_path.suffix + ".part")
    parquet_file = pq.ParquetFile(data_path)
    writer: pq.ParquetWriter | None = None
    out_of_time_score_counts = {
        "validation": 0,
        "test": 0,
        "validation_duplicate_texts_excluded": 0,
        "test_duplicate_texts_excluded": 0,
    }
    try:
        class_indices = {int(label): index for index, label in enumerate(model.classes_)}
        negative_index = class_indices[0]
        positive_index = class_indices[2]
        for batch in parquet_file.iter_batches(batch_size=10_000):
            batch_frame = batch.to_pandas()
            texts = (
                batch_frame["title"].fillna("").astype(str).str.strip()
                + ". "
                + batch_frame["review_text"].fillna("").astype(str).str.strip()
            ).str.strip()
            text_keys = _normalize_model_text(texts)
            batch_ids = batch_frame["review_id"].astype(int)
            validation_mask = batch_ids.isin(validation_ids).to_numpy()
            test_mask = batch_ids.isin(test_ids).to_numpy()
            validation_positions = np.flatnonzero(validation_mask)
            test_positions = np.flatnonzero(test_mask)
            prediction_split = pd.Series(
                ["not_out_of_time"] * len(batch_frame),
                index=batch_frame.index,
                dtype="object",
            )
            if validation_positions.size:
                validation_is_duplicate = np.asarray(
                    [
                        text_keys.iloc[position] in training_text_keys
                        for position in validation_positions
                    ]
                )
                duplicate_positions = validation_positions[validation_is_duplicate]
                out_of_time_score_counts[
                    "validation_duplicate_texts_excluded"
                ] += len(duplicate_positions)
                prediction_split.iloc[duplicate_positions] = (
                    "duplicate_of_training_text"
                )
                validation_positions = validation_positions[~validation_is_duplicate]
                out_of_time_score_counts["validation"] += len(validation_positions)
            if test_positions.size:
                test_is_duplicate = np.asarray(
                    [
                        text_keys.iloc[position] in train_validation_text_keys
                        for position in test_positions
                    ]
                )
                duplicate_positions = test_positions[test_is_duplicate]
                out_of_time_score_counts["test_duplicate_texts_excluded"] += len(
                    duplicate_positions
                )
                prediction_split.iloc[duplicate_positions] = (
                    "duplicate_of_training_text"
                )
                test_positions = test_positions[~test_is_duplicate]
                out_of_time_score_counts["test"] += len(test_positions)

            sentiment_labels = pd.Series(
                pd.NA,
                index=batch_frame.index,
                dtype="string",
            )
            sentiment_scores = pd.Series(
                float("nan"),
                index=batch_frame.index,
                dtype="float32",
            )
            if validation_positions.size:
                validation_texts = texts.iloc[validation_positions]
                validation_probabilities = validation_model.predict_proba(
                    validation_texts
                )
                validation_predictions = validation_model.predict(validation_texts)
                selected_index = batch_frame.index[validation_positions]
                sentiment_labels.loc[selected_index] = [
                    LABELS[int(prediction)] for prediction in validation_predictions
                ]
                sentiment_scores.loc[selected_index] = (
                    validation_probabilities[:, positive_index]
                    - validation_probabilities[:, negative_index]
                ).astype("float32")
                prediction_split.loc[selected_index] = "validation_oos"
            if test_positions.size:
                test_texts = texts.iloc[test_positions]
                test_probabilities = model.predict_proba(test_texts)
                test_batch_predictions = model.predict(test_texts)
                selected_index = batch_frame.index[test_positions]
                sentiment_labels.loc[selected_index] = [
                    LABELS[int(prediction)] for prediction in test_batch_predictions
                ]
                sentiment_scores.loc[selected_index] = (
                    test_probabilities[:, positive_index]
                    - test_probabilities[:, negative_index]
                ).astype("float32")
                prediction_split.loc[selected_index] = "test_oos"

            batch_frame["sentiment_label"] = sentiment_labels
            batch_frame["sentiment_score"] = sentiment_scores
            batch_frame["sentiment_prediction_split"] = prediction_split
            batch_frame["sentiment_is_out_of_sample"] = prediction_split.isin(
                ["validation_oos", "test_oos"]
            )
            output_table = pa.Table.from_pandas(batch_frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output,
                    output_table.schema,
                    compression="zstd",
                )
            writer.write_table(output_table)
        if writer is not None:
            writer.close()
            writer = None
        os.replace(temporary_output, scored_data_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    metrics_document = {
        "dataset": "Amazon Reviews 2023 / All_Beauty",
        "dataset_source": "https://amazon-reviews-2023.github.io/",
        "target": "star-rating-derived sentiment: 1-2 negative, 3 neutral, 4-5 positive",
        "split_method": (
            "chronological 60% train / 10% hyperparameter-selection validation / "
            "10% out-of-time validation / 20% held-out test"
        ),
        "duplicate_text_handling": (
            "normalized exact title+review-text duplicates removed before split, "
            "keeping earliest timestamp"
        ),
        "duplicate_text_rows_removed_before_split": duplicate_text_rows,
        "train_rows": len(train),
        "hyperparameter_selection_rows": len(selection_validation),
        "validation_rows": len(validation),
        "test_rows": len(test),
        "validation_hyperparameter_candidates": {
            "sgd_alpha": list(candidate_alphas),
            "metrics_by_alpha": validation_metrics,
            "selected_alpha": selected_alpha,
            "selection_metric": (
                "hyperparameter-selection validation macro-F1, accuracy as tie-breaker"
            ),
        },
        "out_of_time_validation_metrics": out_of_time_validation_metrics,
        "out_of_time_prediction_counts": {
            **out_of_time_score_counts,
            "total": (
                out_of_time_score_counts["validation"]
                + out_of_time_score_counts["test"]
            ),
        },
        "majority_class_baseline_test_metrics": majority_baseline,
        "model": "TF-IDF unigrams+bigrams with balanced SGDClassifier (log loss)",
        "dashboard_sentiment_predictions": (
            "out-of-time predictions for validation and test rows only; exact "
            "duplicate texts already present in the relevant training set are excluded"
        ),
        "metrics_are_against_rating_derived_labels": True,
        "held_out_test_metrics": metrics,
        "model_artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
        "scored_data": str(scored_data_path.relative_to(PROJECT_ROOT)),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics_document, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics_document


def main() -> None:
    result = train_and_score()
    test_metrics = result["held_out_test_metrics"]
    print(f"Trained model: {MODEL_PATH}")
    print(f"Held-out chronological test rows: {result['test_rows']:,}")
    print(f"Rating-label accuracy: {test_metrics['accuracy']:.3f}")
    print(f"Rating-label macro-F1: {test_metrics['macro_f1']:.3f}")
    print(f"Scored reviews: {SCORED_DATA_PATH}")
    print("Interpret these as rating-label prediction metrics, not human sentiment accuracy.")


if __name__ == "__main__":
    main()
