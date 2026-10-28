"""Train and evaluate text-based sentiment classifiers on UCI Amazon reviews."""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/331/"
    "sentiment+labelled+sentences.zip"
)
DATASET_PATH = DATA_DIR / "amazon_cells_labelled.txt"
MODEL_PATH = MODELS_DIR / "sentiment_classifier.joblib"
METRICS_PATH = MODELS_DIR / "sentiment_metrics.json"
RANDOM_STATE = 42


def ensure_dataset() -> Path:
    """Download the UCI archive if needed and retain only its Amazon review file."""
    if DATASET_PATH.is_file():
        return DATASET_PATH

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "E-commerce-Sentiment-Analysis/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        source_name = next(
            (
                name
                for name in archive.namelist()
                if name.endswith("amazon_cells_labelled.txt")
            ),
            None,
        )
        if source_name is None:
            raise ValueError("The UCI archive does not contain the Amazon review data.")
        DATASET_PATH.write_bytes(archive.read(source_name))

    return DATASET_PATH


def load_dataset(path: Path) -> pd.DataFrame:
    dataset = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["text", "label"],
        dtype={"text": "string", "label": "int64"},
    )
    if dataset.empty or dataset["text"].isna().any():
        raise ValueError("The sentiment dataset must contain non-empty review text.")
    if set(dataset["label"].unique()) != {0, 1}:
        raise ValueError("The sentiment dataset must contain both labels 0 and 1.")

    dataset["text"] = dataset["text"].str.strip()
    dataset = dataset.loc[dataset["text"].ne("")].drop_duplicates(
        subset="text", keep="first"
    )
    if dataset["label"].nunique() != 2:
        raise ValueError("Both sentiment classes must remain after removing duplicates.")
    return dataset.reset_index(drop=True)


def build_pipeline(classifier: str) -> Pipeline:
    if classifier == "logistic_regression":
        estimator = LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
    elif classifier == "multinomial_nb":
        estimator = MultinomialNB(alpha=0.3)
    else:
        raise ValueError(f"Unknown sentiment classifier: {classifier}")

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    sublinear_tf=True,
                ),
            ),
            ("classifier", estimator),
        ]
    )


def evaluate(model: Pipeline, texts: pd.Series, labels: pd.Series) -> dict:
    predictions = model.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro")),
        "confusion_matrix_labels_0_then_1": confusion_matrix(
            labels, predictions, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=["negative", "positive"],
            output_dict=True,
            zero_division=0,
        ),
    }


def train_and_evaluate(dataset_path: Path | None = None) -> dict:
    source_path = dataset_path or ensure_dataset()
    dataset = load_dataset(source_path)
    train_validation, test = train_test_split(
        dataset,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=dataset["label"],
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=train_validation["label"],
    )

    candidate_names = ("logistic_regression", "multinomial_nb")
    validation_metrics = {}
    for name in candidate_names:
        model = build_pipeline(name)
        model.fit(train["text"], train["label"])
        validation_metrics[name] = evaluate(
            model, validation["text"], validation["label"]
        )

    selected_name = max(
        candidate_names,
        key=lambda name: (
            validation_metrics[name]["macro_f1"],
            validation_metrics[name]["accuracy"],
        ),
    )
    final_model = build_pipeline(selected_name)
    final_model.fit(train_validation["text"], train_validation["label"])
    test_metrics = evaluate(final_model, test["text"], test["label"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)
    metrics = {
        "dataset": "UCI Sentiment Labelled Sentences: Amazon cell-phone reviews",
        "dataset_url": "https://doi.org/10.24432/C57604",
        "dataset_license": "CC BY 4.0",
        "dataset_rows_after_exact_text_deduplication": len(dataset),
        "label_mapping": {"0": "negative", "1": "positive"},
        "split_rows": {
            "train": len(train),
            "validation": len(validation),
            "test": len(test),
        },
        "selection_metric": "validation macro F1, with accuracy as tie-breaker",
        "validation_metrics": validation_metrics,
        "selected_model": selected_name,
        "held_out_test_metrics": test_metrics,
        "random_state": RANDOM_STATE,
        "model_artifact": str(MODEL_PATH.relative_to(PROJECT_ROOT)),
    }
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    metrics = train_and_evaluate()
    selected_test = metrics["held_out_test_metrics"]
    print(f"Selected model: {metrics['selected_model']}")
    print(f"Held-out test accuracy: {selected_test['accuracy']:.3f}")
    print(f"Held-out test macro F1: {selected_test['macro_f1']:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved evaluation: {METRICS_PATH}")


if __name__ == "__main__":
    main()
