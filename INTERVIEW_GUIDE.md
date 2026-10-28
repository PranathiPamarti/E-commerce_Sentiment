# Interview Study Guide

## 1. Project in one sentence

I built an end-to-end product-review analytics app that profiles a public Amazon review sample, predicts star-rating-derived sentiment out of time, surfaces rule-based issue candidates, and supports product/reviewer investigation with an explicit human-audit workflow.

## 2. Dataset and scope

- Source: Amazon Reviews 2023, **All_Beauty** category.
- Local deterministic sample: **399,596 valid reviews**, **87,139 products**, **373,031 reviewers**; 404 sampled records failed required-field checks and were dropped.
- Sample spans 2000-11-01 through 2023-09-09 UTC; star ratings are 1–5. The local sample and source-line counts are described in `data/review_intelligence_stats.json`.
- The sample is not the full category. Do not describe the 399,596 as the full source population.
- Text and model artifacts are local and ignored by Git.

## 3. What the model predicts

The public category data has ratings but no human sentiment annotation. The target is therefore a weak proxy:

| Rating | Training target |
|---|---|
| 1–2 stars | Negative |
| 3 stars | Neutral |
| 4–5 stars | Positive |

This model predicts the rating-derived class from the review title and text. It does not establish human sentiment, customer intent, or emotional tone independently of ratings.

## 4. Main sentiment model and evaluation

**Representation:** word TF-IDF unigrams and bigrams, lowercase text, accent normalization, `min_df=2`, maximum 100,000 features. Bigrams can represent short local phrases such as “not effective.”

**Classifier:** `SGDClassifier` with logistic loss and balanced class weights. Logistic loss provides class probabilities; class weighting responds to the imbalanced rating-derived classes.

**Leakage controls and split:**

1. Normalize title-plus-review text by casefolding, whitespace normalization, and trimming.
2. Remove exact normalized duplicate rows before the split, keeping the earliest timestamp. **15,586 rows** were removed in the current run.
3. Sort chronologically and use 60% train / 10% hyperparameter tuning / 10% later validation / 20% final test:

| Period | Rows | Purpose |
|---|---:|---|
| Train | 230,406 | Fit the candidate models |
| Tuning validation | 38,401 | Choose regularization by macro-F1 |
| Out-of-time validation | 38,401 | Evaluate the selected model on later data |
| Held-out test | 76,802 | Final comparison |

The training run compares two `alpha` values, `0.0001` and `0.001`, then refits the selected configuration on the first 80%. That is **three fit calls in one run, but one final saved main model**. The independent 10% validation prediction uses the selected candidate trained on the first 60%; the test prediction uses the final refit.

### Current held-out results

| Measure | Main model | Majority baseline |
|---|---:|---:|
| Accuracy | 85.12% | 64.70% |
| Macro-F1 | 0.6997 | 0.2619 |

Class F1: negative **0.819**, neutral **0.358**, positive **0.922**. The low neutral F1 and substantial class imbalance are important caveats. Accuracy alone would overstate what the model does well.

The app stores **38,401 out-of-time validation** and **76,802 held-out test** sentiment predictions (115,203 total) across **37,068 products**. Earlier training rows are not assigned in-sample sentiment values. Volume and rule-based issue analysis still use the full prepared sample.

## 5. How many models are run?

There are two distinct final sentiment artifacts, not one model trained repeatedly by the dashboard:

1. **Primary three-class All_Beauty TF-IDF + SGD model.** Its training command fits two hyperparameter candidates and one final refit (three fit calls) and saves one final pipeline.
2. **Separate binary synthetic-demo TF-IDF model.** Its training command compares logistic regression with Multinomial Naive Bayes and refits the selected winner (three fit calls), saving one binary pipeline. It was trained on 990 deduplicated UCI Amazon cell-phone review sentences; its 198-row test has 81.82% accuracy and 0.818 macro-F1. It only supports the legacy synthetic app screens.

Loading either dashboard page does not train either model. The main dashboard loads persisted scored Parquet and metrics. The sentiment page loads the saved main model only when the interactive text prediction is requested.

The recommendation page is not a third learned model. Its latent factors are produced by the synthetic-data generator, and its content similarity is a prototype component. The 32 issue tags are regular expressions, not ML classifications.

## 6. Issue tags, reviewer features, and dashboard

- **Issue candidates:** 32 readable regex patterns. All 32 meet the current descriptive threshold of at least 25 matched reviews across at least 3 products. This is coverage, not accuracy or precision.
- **Audit:** the random stratified sample contains 400 reviews. Candidate-tagged records are oversampled; inverse-probability weights restore population estimates. The audit currently has **0/400 primary labels**, so there is no measured issue precision or recall and no evidence for a 91% claim.
- **Reviewer signals:** volume, activity rate, out-of-time negative share, and repeated text. These are exploratory signals, not proof of fraud or coordination. Reviewers without out-of-time sentiment scores have a missing negative share, not an assumed positive/negative result.
- **UI:** project overview, data profile/quality, sentiment model explanation, issue and product analytics, reviewer signals, blinded issue audit, and a segregated synthetic demo.

## 7. A defendable interview response

> “I used a public All_Beauty review sample of about 400,000 records. Ratings were the only available sentiment supervision, so I mapped them into three weak classes and made that limitation explicit. I deduplicated normalized exact text before chronological splitting, tuned regularization on an earlier validation period, and reserved later periods for validation and final testing. The TF-IDF + balanced SGD model got 85.12% accuracy and 0.6997 macro-F1 versus a 64.70% / 0.2619 majority baseline. Its neutral F1 is only 0.358, so I would not claim human sentiment accuracy. I built 32 interpretable issue candidates and a stratified human audit, but the audit is unfinished, so I do not claim a measured 91% precision.”

## 8. Limitations and next steps

1. Finish blinded human issue annotation; double-code a subset and report weighted precision, recall, and confidence intervals.
2. Human-label a representative sentiment sample independently of star ratings; re-evaluate per-class performance, especially neutral.
3. Validate temporal and reviewer signals with product/context review before drawing conclusions.
4. If the project needs a real recommender claim, fit collaborative/content models on observed interactions with time-based holdout and compare against popularity. Current factor matrices are synthetic.

## 9. Re-run commands

```powershell
python -m src.prepare_review_intelligence
python -m src.train_review_sentiment
python -m streamlit run app/app.py
```

For notebook presentation, open `notebooks/1_data_cleaning_eda.ipynb` with the project kernel and run all cells. Full training is off by default; its explicit switch is `RUN_FULL_TRAINING = True`.
