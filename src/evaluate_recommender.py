"""Evaluate the demo recommender against a popularity baseline."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    score_range = np.ptp(scores)
    if score_range == 0:
        return np.full(scores.shape, 0.5)
    return (scores - scores.min()) / score_range


def ranking_metrics(
    recommendations: list[str], relevant: set[str], top_k: int
) -> tuple[float, float, float]:
    ranked = recommendations[:top_k]
    hits = [1 if product_id in relevant else 0 for product_id in ranked]
    precision = sum(hits) / top_k
    recall = sum(hits) / len(relevant)
    dcg = sum(hit / np.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = min(len(relevant), top_k)
    ideal_dcg = sum(1 / np.log2(index + 2) for index in range(ideal_hits))
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return precision, recall, ndcg


def evaluate(
    reviews: pd.DataFrame,
    model: dict,
    top_k: int = 10,
    holdout_fraction: float = 0.2,
) -> tuple[pd.DataFrame, dict[str, float]]:
    required = {
        "UserId",
        "ProductId",
        "Score",
        "Time",
        "user_index",
        "product_index",
        "avg_sentiment_score",
    }
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"Review data is missing columns: {', '.join(sorted(missing))}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")
    if not 0 < holdout_fraction < 1:
        raise ValueError("holdout_fraction must be between 0 and 1.")

    factors_p = np.asarray(model["P"], dtype=float)
    factors_q = np.asarray(model["Q"], dtype=float)
    product_text = model["product_text"]
    if not isinstance(product_text, pd.DataFrame) or "ProductId" not in product_text:
        raise ValueError("The model product catalog must be a table with ProductId values.")
    if factors_p.ndim != 2 or factors_q.ndim != 2 or factors_p.shape[1] != factors_q.shape[1]:
        raise ValueError("The user and product factor matrices have incompatible shapes.")

    data = reviews.dropna(subset=["UserId", "ProductId", "Score", "Time"]).copy()
    data["Time"] = pd.to_numeric(data["Time"], errors="coerce")
    data = data.dropna(subset=["Time"]).sort_values(["UserId", "Time"])
    product_catalog = product_text["ProductId"].drop_duplicates().tolist()
    if not product_catalog or product_text["ProductId"].duplicated().any():
        raise ValueError("The model product catalog is empty.")
    product_mapping = data[["ProductId", "product_index"]].dropna().drop_duplicates()
    if product_mapping["ProductId"].duplicated().any():
        raise ValueError("Review data maps a product to multiple model indices.")
    product_index = product_mapping.set_index("ProductId")["product_index"]
    if not np.equal(
        product_index.to_numpy(dtype=float),
        product_index.to_numpy(dtype=int),
    ).all():
        raise ValueError("Product indices must be integers.")
    product_index = product_index.astype(int)
    if (product_index < 0).any() or (product_index >= factors_q.shape[0]).any():
        raise ValueError("A product index is outside the model.")

    metric_rows: list[dict[str, float | str]] = []
    recommended_sets: dict[str, set[str]] = {"Popularity": set(), "Hybrid": set()}
    evaluated_users = 0

    for user_id, user_reviews in data.groupby("UserId", sort=True):
        if len(user_reviews) < 5:
            continue
        holdout_size = max(1, int(np.ceil(len(user_reviews) * holdout_fraction)))
        train = user_reviews.iloc[:-holdout_size]
        test = user_reviews.iloc[-holdout_size:]
        relevant = set(test.loc[test["Score"] >= 4, "ProductId"])
        if not relevant:
            continue

        user_indices = train.get("user_index", pd.Series(dtype=float)).dropna().unique()
        if len(user_indices) != 1:
            raise ValueError(f"Expected one model index for user {user_id!r}.")
        user_index = int(user_indices[0])
        if float(user_indices[0]) != user_index:
            raise ValueError(f"User index for {user_id!r} is not an integer.")
        if user_index < 0 or user_index >= factors_p.shape[0]:
            raise ValueError(f"User index for {user_id!r} is outside the model.")

        seen = set(train["ProductId"])
        candidates = [product_id for product_id in product_catalog if product_id not in seen]
        if not candidates:
            continue
        candidate_indices = product_index.reindex(candidates)
        if candidate_indices.isna().any():
            raise ValueError("A catalog product is missing a model index.")
        candidate_indices = candidate_indices.to_numpy(dtype=int)
        if (candidate_indices < 0).any() or (candidate_indices >= factors_q.shape[0]).any():
            raise ValueError("A product index is outside the model.")

        collaborative_scores = factors_p[user_index] @ factors_q[candidate_indices].T
        collaborative_scores = normalize_scores(collaborative_scores)
        train_sentiment = train.groupby("ProductId")["avg_sentiment_score"].mean()
        overall_sentiment = float(train["avg_sentiment_score"].mean())
        sentiment = train_sentiment.reindex(candidates).fillna(overall_sentiment)
        sentiment_scores = ((sentiment.to_numpy(dtype=float) + 1) / 2).clip(0, 1)
        hybrid_scores = 0.75 * collaborative_scores + 0.25 * sentiment_scores

        popularity = train["ProductId"].value_counts()
        popularity_ranked = sorted(
            candidates, key=lambda product_id: (-int(popularity.get(product_id, 0)), product_id)
        )
        hybrid_ranked = sorted(
            zip(candidates, hybrid_scores),
            key=lambda item: (-float(item[1]), item[0]),
        )
        ranked_items = {
            "Popularity": popularity_ranked,
            "Hybrid": [product_id for product_id, _ in hybrid_ranked],
        }
        evaluated_users += 1
        for method, recommendations in ranked_items.items():
            recommended_sets[method].update(recommendations[:top_k])
            precision, recall, ndcg = ranking_metrics(recommendations, relevant, top_k)
            metric_rows.append(
                {
                    "Method": method,
                    "Precision@K": precision,
                    "Recall@K": recall,
                    "NDCG@K": ndcg,
                }
            )

    if evaluated_users == 0:
        raise ValueError(
            "No users had both enough history and at least one held-out 4+ star review."
        )

    metrics = pd.DataFrame(metric_rows).groupby("Method").mean(numeric_only=True)
    coverage = {
        method: len(products) / len(product_catalog)
        for method, products in recommended_sets.items()
    }
    return metrics, {
        "evaluated_users": float(evaluated_users),
        "catalog_coverage_popularity": coverage["Popularity"],
        "catalog_coverage_hybrid": coverage["Hybrid"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    args = parser.parse_args()

    data_path = PROJECT_ROOT / "data" / "sentiment_reviews.csv"
    model_path = PROJECT_ROOT / "models" / "hybrid_model.pkl"
    if not data_path.exists() or not model_path.exists():
        raise SystemExit(
            "Review data or model bundle not found. Run "
            "'python -m src.generate_synthetic_data' first."
        )

    reviews = pd.read_csv(data_path)
    with model_path.open("rb") as model_file:
        model = pickle.load(model_file)
    metrics, summary = evaluate(
        reviews, model, top_k=args.top_k, holdout_fraction=args.holdout_fraction
    )
    metrics.index.name = "Method"
    print(f"Temporal holdout evaluation (K={args.top_k}):")
    print(metrics.to_string(float_format=lambda value: f"{value:.4f}"))
    print(
        f"\nEvaluated users: {int(summary['evaluated_users'])}\n"
        f"Catalog coverage: popularity={summary['catalog_coverage_popularity']:.1%}, "
        f"hybrid={summary['catalog_coverage_hybrid']:.1%}"
    )
    print(
        "\nSynthetic-data metrics are a sanity check for this demo, not evidence "
        "of real-world recommendation quality."
    )


if __name__ == "__main__":
    main()
