"""Shared loading and preparation for review data used by Streamlit pages."""

from pathlib import Path

import joblib
import pandas as pd


def review_data_version(project_root: Path) -> tuple[int, ...]:
    paths = (
        project_root / "data" / "sentiment_reviews.csv",
        project_root / "data" / "product_text.csv",
        project_root / "models" / "sentiment_classifier.joblib",
    )
    version: list[int] = []
    for path in paths:
        if path.name == "sentiment_classifier.joblib" and not path.is_file():
            raise FileNotFoundError(
                "The trained sentiment model is missing. From the project root, run "
                "`python -m src.train_sentiment_model`."
            )
        stat = path.stat()
        version.extend((stat.st_mtime_ns, stat.st_size))
    return tuple(version)


def load_review_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = project_root / "data" / "sentiment_reviews.csv"
    product_path = project_root / "data" / "product_text.csv"
    model_path = project_root / "models" / "sentiment_classifier.joblib"
    if not data_path.is_file() or not product_path.is_file():
        raise FileNotFoundError(
            "Review data is missing. From the project root, run "
            "`python -m src.generate_synthetic_data`."
        )
    if not model_path.is_file():
        raise FileNotFoundError(
            "The trained sentiment model is missing. From the project root, run "
            "`python -m src.train_sentiment_model`."
        )

    reviews = pd.read_csv(data_path)
    products = pd.read_csv(product_path)
    required_review_columns = {"UserId", "ProductId", "Score", "Text"}
    missing_review_columns = required_review_columns.difference(reviews.columns)
    if missing_review_columns:
        raise ValueError(
            f"Review data is missing required columns: "
            f"{', '.join(sorted(missing_review_columns))}."
        )

    required_product_columns = {
        "ProductId",
        "product_title",
        "brand",
        "product_category",
        "price",
    }
    missing_product_columns = required_product_columns.difference(products.columns)
    if missing_product_columns:
        raise ValueError(
            f"Product catalog is missing required columns: "
            f"{', '.join(sorted(missing_product_columns))}."
        )
    if products["ProductId"].duplicated().any():
        raise ValueError("Product catalog ProductId values must be unique.")

    categories = products[["ProductId", "product_category"]].rename(
        columns={"product_category": "CatalogCategory"}
    )
    reviews = reviews.merge(
        categories,
        on="ProductId",
        how="left",
        validate="many_to_one",
    )
    if "ProductCategory" in reviews:
        mismatches = (
            reviews["ProductCategory"].notna()
            & reviews["CatalogCategory"].notna()
            & reviews["ProductCategory"].ne(reviews["CatalogCategory"])
        )
        if mismatches.any():
            raise ValueError("Review categories do not match the product catalog.")
    reviews["ProductCategory"] = reviews["CatalogCategory"]
    reviews = reviews.drop(columns="CatalogCategory")

    model = joblib.load(model_path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise ValueError("The saved sentiment model must support predict and predict_proba.")
    classes = list(model.classes_)
    if set(classes) != {0, 1}:
        raise ValueError("The saved sentiment model must have negative/positive classes 0 and 1.")

    positive_class_index = classes.index(1)
    positive_probability = model.predict_proba(reviews["Text"].fillna("").astype(str))[
        :, positive_class_index
    ]
    if not pd.Series(positive_probability).between(0, 1).all():
        raise ValueError("The sentiment model returned invalid probabilities.")
    reviews["sentiment_value"] = pd.Series(
        model.predict(reviews["Text"].fillna("").astype(str)),
        index=reviews.index,
    ).astype(int)
    reviews["avg_sentiment_score"] = 2 * positive_probability - 1
    reviews["sentiment_label"] = reviews["sentiment_value"].map(
        {1: "Positive", 0: "Negative"}
    )
    if "Time" in reviews:
        reviews["Time"] = pd.to_datetime(
            reviews["Time"], unit="s", errors="coerce", utc=True
        )
    return reviews, products
