"""Generate a deterministic synthetic dataset and model bundle for the demo app."""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = [
    "coffee",
    "tea",
    "snacks",
    "cereal",
    "sauces",
    "chocolate",
    "beverages",
    "pantry",
]
POSITIVE_TEXT = [
    "Excellent quality and great taste",
    "Very enjoyable and worth buying",
    "Fresh, delicious, and highly recommended",
    "Works exactly as expected; I would buy again",
]
NEUTRAL_TEXT = [
    "The product is okay for the price",
    "Average quality and as described",
    "It is usable but nothing special",
    "A reasonable product with ordinary results",
]
NEGATIVE_TEXT = [
    "Disappointing quality and not worth the price",
    "The taste was poor and I would not buy again",
    "Did not meet expectations",
    "Low quality and difficult to use",
]
CATEGORY_DETAILS = {
    "coffee": [
        "The roast has a smooth aroma and makes a good morning cup.",
        "It has a rich flavor that works well as part of my daily routine.",
    ],
    "tea": [
        "The flavor is easy to enjoy during a quiet evening.",
        "It makes a comforting cup with a pleasant taste.",
    ],
    "snacks": [
        "It is a convenient snack with a satisfying bite.",
        "The flavor makes it an enjoyable option between meals.",
    ],
    "cereal": [
        "It makes a quick breakfast and has a pleasant taste.",
        "The texture works well for an easy start to the day.",
    ],
    "sauces": [
        "The flavor adds a nice touch to everyday meals.",
        "It is easy to use when preparing a quick meal.",
    ],
    "chocolate": [
        "The rich flavor makes it a pleasant treat.",
        "It has a nice taste for an occasional sweet snack.",
    ],
    "beverages": [
        "It has a refreshing taste that is easy to enjoy.",
        "The flavor makes it a pleasant drink during the day.",
    ],
    "pantry": [
        "It is a useful staple for simple everyday meals.",
        "It fits easily into my regular cooking routine.",
    ],
}


def product_similarity(product_text: pd.DataFrame) -> np.ndarray:
    """Build cosine similarities from the generated product-description tokens."""
    token_sets = [
        set(re.findall(r"[a-z]+", text.lower()))
        for text in product_text["product_text"]
    ]
    vocabulary = sorted(set().union(*token_sets))
    token_index = {token: index for index, token in enumerate(vocabulary)}
    vectors = np.zeros((len(token_sets), len(vocabulary)), dtype=np.float32)

    for row, tokens in enumerate(token_sets):
        for token in tokens:
            vectors[row, token_index[token]] = 1

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.maximum(norms, 1e-12)
    return normalized @ normalized.T


def build_dataset(review_count: int, user_count: int, product_count: int, seed: int) -> None:
    if review_count < 1500:
        raise ValueError("review_count must be at least 1500 for the Product Insights word cloud.")
    if user_count < 2 or product_count < 2:
        raise ValueError("user_count and product_count must both be at least 2.")
    if review_count > user_count * product_count:
        raise ValueError("review_count cannot exceed user_count * product_count.")

    rng = np.random.default_rng(seed)
    user_ids = [f"USER{i:05d}" for i in range(user_count)]
    product_ids = [f"SYN{i:06d}" for i in range(product_count)]
    product_categories = [CATEGORIES[i % len(CATEGORIES)] for i in range(product_count)]
    product_qualities = rng.uniform(-0.8, 0.8, size=product_count)
    user_preferences = rng.normal(0, 0.65, size=user_count)
    latent_dimensions = 8
    user_factors = rng.uniform(0.05, 0.45, size=(user_count, latent_dimensions))
    product_factors = rng.uniform(0.05, 0.45, size=(product_count, latent_dimensions))

    product_rows = []
    for index, (product_id, category) in enumerate(zip(product_ids, product_categories)):
        flavor = ["classic", "light", "bold", "organic"][index % 4]
        description = (
            f"{flavor} {category} product with consistent quality, "
            f"convenient packaging, and dependable everyday use"
        )
        product_rows.append(
            {
                "ProductId": product_id,
                "product_title": f"{flavor.title()} {category.title()} {index + 1}",
                "product_category": category,
                "product_text": description,
            }
        )
    product_text = pd.DataFrame(product_rows)

    all_pairs = np.array(
        [(user_index, product_index) for user_index in range(user_count)
         for product_index in range(product_count)],
        dtype=np.int32,
    )
    selected = rng.choice(len(all_pairs), size=review_count, replace=False)
    pairs = all_pairs[selected]

    review_rows = []
    start_time = pd.Timestamp("2023-01-01", tz="UTC")
    for row_number, (user_index, product_index) in enumerate(pairs):
        preference = user_preferences[user_index]
        quality = product_qualities[product_index]
        rating_signal = preference + quality + rng.normal(0, 0.75)
        score = int(np.clip(np.rint(3 + rating_signal), 1, 5))
        sentiment = float(
            np.clip((score - 3) / 2 + rng.normal(0, 0.12), -1, 1)
        )
        if sentiment > 0.2:
            review = POSITIVE_TEXT[int(rng.integers(len(POSITIVE_TEXT)))]
        elif sentiment < -0.2:
            review = NEGATIVE_TEXT[int(rng.integers(len(NEGATIVE_TEXT)))]
        else:
            review = NEUTRAL_TEXT[int(rng.integers(len(NEUTRAL_TEXT)))]

        category = product_categories[product_index]
        detail = CATEGORY_DETAILS[category][int(rng.integers(2))]
        review_text = f"{review}. {detail}"
        clean_text = re.sub(r"[^a-z0-9\s]", "", review_text.lower())
        timestamp = start_time + pd.Timedelta(
            days=int(rng.integers(0, 1000)),
            seconds=int(rng.integers(0, 86400)),
        )
        review_rows.append(
            {
                "UserId": user_ids[user_index],
                "ProductId": product_ids[product_index],
                "Score": score,
                "Time": int(timestamp.timestamp()),
                "Summary": review.split(".")[0][:80],
                "Text": review_text,
                "avg_sentiment_score": sentiment,
                "clean_text": clean_text,
                "user_index": user_index,
                "product_index": product_index,
            }
        )

    reviews = pd.DataFrame(review_rows)
    sentiment_map = reviews.groupby("ProductId")["avg_sentiment_score"].mean().to_dict()
    cosine_sim = product_similarity(product_text)

    data_dir = PROJECT_ROOT / "data"
    models_dir = PROJECT_ROOT / "models"
    data_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reviews.to_csv(data_dir / "sentiment_reviews.csv", index=False)
    product_text.to_csv(data_dir / "product_text.csv", index=False)

    user_map = {user_id: index for index, user_id in enumerate(user_ids)}
    product_map = {product_id: index for index, product_id in enumerate(product_ids)}
    np.save(models_dir / "P.npy", user_factors.astype(np.float32))
    np.save(models_dir / "Q.npy", product_factors.astype(np.float32))
    np.save(models_dir / "cosine_sim.npy", cosine_sim)
    for filename, mapping in (
        ("user_map.pkl", user_map),
        ("product_map.pkl", product_map),
    ):
        with (models_dir / filename).open("wb") as mapping_file:
            pickle.dump(mapping, mapping_file)

    hybrid_model = {
        "P": user_factors.astype(np.float32),
        "Q": product_factors.astype(np.float32),
        "cosine_sim": cosine_sim,
        "sentiment_map": sentiment_map,
        "product_text": product_text,
    }
    with (models_dir / "hybrid_model.pkl").open("wb") as model_file:
        pickle.dump(hybrid_model, model_file)

    print(f"Created {len(reviews):,} synthetic reviews for {user_count:,} users and {product_count:,} products.")
    print(f"Reviews: {data_dir / 'sentiment_reviews.csv'}")
    print(f"Product text: {data_dir / 'product_text.csv'}")
    print(f"Model bundle: {models_dir / 'hybrid_model.pkl'}")
    print(f"Model components: {models_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviews", type=int, default=3000)
    parser.add_argument("--users", type=int, default=100)
    parser.add_argument("--products", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build_dataset(args.reviews, args.users, args.products, args.seed)


if __name__ == "__main__":
    main()
