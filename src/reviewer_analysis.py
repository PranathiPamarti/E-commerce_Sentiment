"""Descriptive reviewer features for review-pattern investigation."""

from __future__ import annotations

import pandas as pd


def build_reviewer_features(reviews: pd.DataFrame) -> pd.DataFrame:
    required = {
        "reviewer_id",
        "review_time",
        "review_text",
        "sentiment_label",
        "sentiment_is_out_of_sample",
    }
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"Review data is missing columns: {', '.join(sorted(missing))}.")
    if reviews.empty:
        raise ValueError("Reviewer features require at least one review.")

    review_count = reviews.groupby("reviewer_id").size().rename("review_count")
    review_dates = reviews.groupby("reviewer_id")["review_time"].agg(
        first_review="min",
        last_review="max",
        active_days="nunique",
    )
    active_months = (
        reviews.assign(
            review_month=reviews["review_time"]
            .dt.tz_convert(None)
            .dt.to_period("M")
            .astype(str)
        )
        .groupby("reviewer_id")["review_month"]
        .nunique()
        .rename("active_months")
    )

    out_of_time_reviews = reviews.loc[reviews["sentiment_is_out_of_sample"]]
    negative_share = (
        out_of_time_reviews.assign(
            is_negative=out_of_time_reviews["sentiment_label"].eq("Negative")
        )
        .groupby("reviewer_id")["is_negative"]
        .mean()
        .rename("negative_share")
    )

    normalized_text = (
        reviews["review_text"]
        .fillna("")
        .astype(str)
        .str.casefold()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    reviewers_per_text = (
        pd.DataFrame(
            {"reviewer_id": reviews["reviewer_id"], "review_text_key": normalized_text}
        )
        .groupby("review_text_key")["reviewer_id"]
        .nunique()
    )
    reused_text = normalized_text.map(reviewers_per_text).ge(3)
    reused_text_share = (
        reviews.assign(reused_text=reused_text)
        .groupby("reviewer_id")["reused_text"]
        .mean()
        .rename("reused_text_share")
    )

    result = pd.concat(
        [review_count, review_dates, active_months, negative_share, reused_text_share],
        axis=1,
    )
    result["review_rate_per_active_month"] = (
        result["review_count"] / result["active_months"].clip(lower=1)
    )
    result["high_volume_flag"] = result["review_count"].ge(
        result["review_count"].quantile(0.99)
    )

    if out_of_time_reviews.empty:
        negative_baseline = None
    else:
        negative_baseline = float(
            out_of_time_reviews["sentiment_label"].eq("Negative").mean()
        )
    result["unusually_negative_flag"] = (
        result["review_count"].ge(10)
        & result["negative_share"].gt(
            min(1.0, negative_baseline + 0.4)
            if negative_baseline is not None
            else 1.0
        )
    )

    eligible_velocity = result.loc[
        result["review_count"].ge(10),
        "review_rate_per_active_month",
    ]
    velocity_threshold = (
        float(eligible_velocity.quantile(0.99))
        if not eligible_velocity.empty
        else float("inf")
    )
    result["burst_activity_flag"] = (
        result["review_count"].ge(10)
        & result["review_rate_per_active_month"].gt(velocity_threshold)
    )
    result["reused_text_flag"] = (
        result["review_count"].ge(3)
        & result["reused_text_share"].gt(0.5)
    )
    result["pattern_flags"] = result[
        [
            "high_volume_flag",
            "unusually_negative_flag",
            "burst_activity_flag",
            "reused_text_flag",
        ]
    ].sum(axis=1)
    return result.reset_index()
