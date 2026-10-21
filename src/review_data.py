"""Shared loading and preparation for review data used by Streamlit pages."""

from pathlib import Path

import pandas as pd


def review_data_version(project_root: Path) -> tuple[int, ...]:
    paths = (
        project_root / "data" / "sentiment_reviews.csv",
        project_root / "data" / "product_text.csv",
    )
    version: list[int] = []
    for path in paths:
        stat = path.stat()
        version.extend((stat.st_mtime_ns, stat.st_size))
    return tuple(version)


def load_review_data(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_path = project_root / "data" / "sentiment_reviews.csv"
    product_path = project_root / "data" / "product_text.csv"
    if not data_path.is_file() or not product_path.is_file():
        raise FileNotFoundError(
            "Review data is missing. From the project root, run "
            "`python -m src.generate_synthetic_data`."
        )

    reviews = pd.read_csv(data_path)
    products = pd.read_csv(product_path)
    required_review_columns = {"UserId", "ProductId", "Score", "avg_sentiment_score"}
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

    reviews["sentiment_value"] = (
        pd.to_numeric(reviews["avg_sentiment_score"], errors="coerce") > 0
    ).astype(int)
    reviews["sentiment_label"] = reviews["sentiment_value"].map(
        {1: "Positive", 0: "Neutral or negative"}
    )
    if "Time" in reviews:
        reviews["Time"] = pd.to_datetime(
            reviews["Time"], unit="s", errors="coerce", utc=True
        )
    return reviews, products
