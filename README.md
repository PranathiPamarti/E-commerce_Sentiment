# E-Commerce Sentiment Analysis

A Streamlit demo for exploring product reviews, sentiment summaries, and a
hybrid product-recommendation prototype.

## Run locally

The repository includes a small, generated synthetic dataset and matching
model artifacts so the app works after cloning. To regenerate them locally:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/app.py
```

To regenerate the included demo data and model artifacts before running the
app, run:

```powershell
python -m src.generate_synthetic_data
```

The generator writes `data/sentiment_reviews.csv`,
`data/product_text.csv`, and the model files under `models/`. Its defaults
create 8,000 synthetic reviews, 200 users, and 120 products, with product
categories, brands, prices, varied review text, verified-purchase flags, and
helpful-vote counts. Re-running it replaces those generated files. You can
adjust the dataset size and random seed, for example:

```powershell
python -m src.generate_synthetic_data --reviews 8000 --users 200 --products 120 --seed 42
```

The included data and model artifacts are only for demonstrating the app. They
are synthetic, not real customer feedback, and should not be used to draw
conclusions about products or model quality. If you delete the files, run the
generator again.

## Evaluate the recommendation prototype

After generating the data and model files, run:

```powershell
python -m src.evaluate_recommender --top-k 10
```

This holds out each eligible user's latest 20% of reviews, ranks unseen
products, and compares the hybrid prototype with a training-set popularity
baseline using Precision@K, Recall@K, and NDCG@K. The model factors are
generated from the synthetic data simulator, so these results are a pipeline
sanity check—not an independent or real-world quality claim.

## Pages

- **Product Insights**: category filters, review and sentiment KPIs, interactive
  rating/category/time charts, product comparisons, and a review word cloud.
- **Hybrid Recommendation Demo**: a prototype recommendation list for a
  selected user and optional reference product.
- **Product Explorer**: product details and recent positive and negative
  reviews.
- **Dataset Explorer**: preview and filter the review data.

## Project layout

```text
app/        Streamlit app and pages
data/       Synthetic demo datasets
models/     Matching synthetic demo model artifacts
notebooks/  Analysis and recommendation experiments
src/        Data generation and shared loading helpers
```

## Current limitations

- The default setup uses deterministic synthetic data, not a production
  reviews dataset.
- Offline ranking metrics compare this synthetic prototype with popularity;
  they do not establish real-world performance. The generated factors encode
  the simulator's preferences and are not a trained model for arbitrary data.
- The app uses precomputed sentiment scores from its dataset; it does not run a
  transformer model at app startup.
