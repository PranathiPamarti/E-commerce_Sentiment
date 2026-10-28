"""Reproducible stratified annotation samples and weighted tag evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.issue_tagging import ISSUE_PATTERNS, has_issue_tags, normalize_tags, parse_tags


ANNOTATION_COLUMNS = [
    "review_id",
    "product_id",
    "reviewer_id",
    "review_time",
    "rating",
    "title",
    "review_text",
    "predicted_tags",
    "human_tags",
    "annotator_2_tags",
    "sampling_stratum",
    "stratum_population",
    "stratum_sample",
    "sampling_weight",
]
STRATUM_ORDER = ("candidate_tag", "no_candidate_tag")


def create_annotation_sample(
    reviews: pd.DataFrame,
    sample_size: int = 400,
    seed: int = 42,
) -> pd.DataFrame:
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2 for stratified sampling.")
    required = set(ANNOTATION_COLUMNS[:7]) | {"issue_tags"}
    missing = required.difference(reviews.columns)
    if missing:
        raise ValueError(f"Review data is missing columns: {', '.join(sorted(missing))}.")
    if reviews["review_id"].duplicated().any():
        raise ValueError("Review IDs must be unique before annotation sampling.")

    frame = reviews.copy()
    frame["issue_tags"] = frame["issue_tags"].map(normalize_tags)
    frame["sampling_stratum"] = np.where(
        frame["issue_tags"].map(has_issue_tags),
        STRATUM_ORDER[0],
        STRATUM_ORDER[1],
    )
    stratum_sizes = frame["sampling_stratum"].value_counts().to_dict()
    available_strata = [stratum for stratum in STRATUM_ORDER if stratum_sizes.get(stratum, 0)]
    if not available_strata:
        raise ValueError("No reviews are available for the annotation sample.")

    target_allocation = {
        STRATUM_ORDER[0]: max(1, round(sample_size * 0.8)),
        STRATUM_ORDER[1]: sample_size - max(1, round(sample_size * 0.8)),
    }
    if len(available_strata) == 1:
        target_allocation[available_strata[0]] = sample_size
    for stratum in available_strata:
        target_allocation[stratum] = min(
            target_allocation[stratum],
            int(stratum_sizes[stratum]),
        )
    unallocated = sample_size - sum(
        target_allocation[stratum] for stratum in available_strata
    )
    while unallocated:
        eligible = [
            stratum
            for stratum in available_strata
            if target_allocation[stratum] < stratum_sizes[stratum]
        ]
        if not eligible:
            raise ValueError("Requested annotation sample exceeds the review population.")
        for stratum in eligible:
            if not unallocated:
                break
            target_allocation[stratum] += 1
            unallocated -= 1

    sampled_frames = []
    for offset, stratum in enumerate(available_strata):
        population_size = int(stratum_sizes[stratum])
        sample_count = int(target_allocation[stratum])
        selected = frame.loc[frame["sampling_stratum"].eq(stratum)].sample(
            n=sample_count,
            random_state=seed + offset,
            replace=False,
        )
        selected = selected.copy()
        selected["stratum_population"] = population_size
        selected["stratum_sample"] = sample_count
        selected["sampling_weight"] = population_size / sample_count
        sampled_frames.append(selected)

    result = pd.concat(sampled_frames, ignore_index=True)
    result["predicted_tags"] = result["issue_tags"].map(
        lambda tags: "|".join(tags) if tags else "none"
    )
    result["human_tags"] = ""
    result["annotator_2_tags"] = ""
    return (
        result[ANNOTATION_COLUMNS]
        .sort_values("review_id")
        .reset_index(drop=True)
    )


def _micro_metrics(
    predictions: list[set[str]],
    truths: list[set[str]],
    weights: np.ndarray,
) -> dict[str, float]:
    true_positive = 0.0
    false_positive = 0.0
    false_negative = 0.0
    for predicted, truth, weight in zip(predictions, truths, weights):
        true_positive += len(predicted & truth) * weight
        false_positive += len(predicted - truth) * weight
        false_negative += len(truth - predicted) * weight

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive_tags": true_positive,
        "false_positive_tags": false_positive,
        "false_negative_tags": false_negative,
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
    }


def evaluate_annotations(
    annotations: pd.DataFrame,
    bootstrap_iterations: int = 1000,
    seed: int = 42,
) -> dict:
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be at least 1.")
    required = {
        "predicted_tags",
        "human_tags",
        "sampling_stratum",
        "stratum_population",
        "stratum_sample",
    }
    missing = required.difference(annotations.columns)
    if missing:
        raise ValueError(f"Annotation file is missing columns: {', '.join(sorted(missing))}.")
    if annotations.empty:
        raise ValueError("The annotation file contains no reviews.")
    if annotations["review_id"].duplicated().any():
        raise ValueError("Review IDs must be unique in the annotation file.")
    unreviewed = annotations["human_tags"].fillna("").astype(str).str.strip().eq("")
    if unreviewed.any():
        raise ValueError(
            f"All sampled rows must be annotated before estimating precision; "
            f"{int(unreviewed.sum())} remain."
        )

    predictions = [parse_tags(value) for value in annotations["predicted_tags"]]
    truths = [parse_tags(value) for value in annotations["human_tags"]]
    sample_weights = pd.to_numeric(
        annotations["sampling_weight"],
        errors="coerce",
    ).to_numpy(dtype=float)
    if (
        not np.isfinite(sample_weights).all()
        or (sample_weights <= 0).any()
        or (annotations["stratum_population"].astype(float) <= 0).any()
        or (annotations["stratum_sample"].astype(float) <= 0).any()
    ):
        raise ValueError("Sampling weights and stratum sizes must be positive and finite.")
    expected_weights = (
        annotations["stratum_population"].astype(float)
        / annotations["stratum_sample"].astype(float)
    ).to_numpy()
    if not np.allclose(sample_weights, expected_weights):
        raise ValueError("Sampling weights do not match the recorded stratum sizes.")
    if annotations.groupby("sampling_stratum")["stratum_sample"].nunique().gt(1).any():
        raise ValueError("Every row in a stratum must use the same sample size.")
    if annotations.groupby("sampling_stratum")["stratum_population"].nunique().gt(1).any():
        raise ValueError("Every row in a stratum must use the same population size.")
    if (
        annotations.groupby("sampling_stratum").size()
        != annotations.groupby("sampling_stratum")["stratum_sample"].first()
    ).any():
        raise ValueError("Every sampled row must be annotated before estimation.")

    estimates = _micro_metrics(predictions, truths, sample_weights)
    rng = np.random.default_rng(seed)
    strata = {
        stratum: np.flatnonzero(
            annotations["sampling_stratum"].to_numpy() == stratum
        )
        for stratum in annotations["sampling_stratum"].unique()
    }
    bootstrap_metrics = {key: [] for key in ("micro_precision", "micro_recall", "micro_f1")}
    for _ in range(bootstrap_iterations):
        sampled_indices = np.concatenate(
            [
                rng.choice(indices, size=len(indices), replace=True)
                for indices in strata.values()
            ]
        )
        replicate = _micro_metrics(
            [predictions[index] for index in sampled_indices],
            [truths[index] for index in sampled_indices],
            sample_weights[sampled_indices],
        )
        for key in bootstrap_metrics:
            bootstrap_metrics[key].append(replicate[key])

    for key, values in bootstrap_metrics.items():
        estimates[f"{key}_ci_95"] = [
            float(np.quantile(values, 0.025)),
            float(np.quantile(values, 0.975)),
        ]

    per_tag = {}
    for tag in ISSUE_PATTERNS:
        true_positive = sum(
            weight for predicted, truth, weight in zip(predictions, truths, sample_weights)
            if tag in predicted and tag in truth
        )
        false_positive = sum(
            weight for predicted, truth, weight in zip(predictions, truths, sample_weights)
            if tag in predicted and tag not in truth
        )
        false_negative = sum(
            weight for predicted, truth, weight in zip(predictions, truths, sample_weights)
            if tag not in predicted and tag in truth
        )
        denominator = true_positive + false_positive
        if denominator:
            per_tag[tag] = {
                "estimated_true_positive": true_positive,
                "estimated_false_positive": false_positive,
                "estimated_false_negative": false_negative,
                "estimated_precision": true_positive / denominator,
                "sample_predicted_reviews": sum(tag in value for value in predictions),
                "sample_human_reviews": sum(tag in value for value in truths),
            }

    second_column = annotations.get("annotator_2_tags")
    double_coded = (
        annotations.loc[
            second_column.fillna("").astype(str).str.strip().ne("")
            & annotations["human_tags"].fillna("").astype(str).str.strip().ne("")
        ]
        if second_column is not None
        else annotations.iloc[0:0]
    )
    agreement: dict[str, object] = {
        "double_coded_reviews": len(double_coded),
        "exact_set_agreement": None,
        "mean_per_tag_cohen_kappa": None,
        "per_tag_cohen_kappa": {},
    }
    if not double_coded.empty:
        primary_tags = [parse_tags(value) for value in double_coded["human_tags"]]
        secondary_tags = [
            parse_tags(value) for value in double_coded["annotator_2_tags"]
        ]
        agreement["exact_set_agreement"] = float(
            np.mean(
                [
                    primary == secondary
                    for primary, secondary in zip(primary_tags, secondary_tags)
                ]
            )
        )
        kappas: dict[str, float] = {}
        for tag in ISSUE_PATTERNS:
            primary_binary = np.array([tag in labels for labels in primary_tags])
            secondary_binary = np.array([tag in labels for labels in secondary_tags])
            observed = float(np.mean(primary_binary == secondary_binary))
            primary_rate = float(primary_binary.mean())
            secondary_rate = float(secondary_binary.mean())
            expected = (
                primary_rate * secondary_rate
                + (1 - primary_rate) * (1 - secondary_rate)
            )
            if expected < 1:
                kappas[tag] = (observed - expected) / (1 - expected)
        agreement["mean_per_tag_cohen_kappa"] = (
            float(np.mean(list(kappas.values()))) if kappas else None
        )
        agreement["per_tag_cohen_kappa"] = kappas

    return {
        "sampling_method": "stratified random sample by presence of any candidate issue tag",
        "confidence_interval_method": (
            f"{bootstrap_iterations}-replicate percentile bootstrap within sampling strata"
        ),
        "human-label-reference": "primary annotator; second-annotator agreement is reported separately",
        "annotated_reviews": len(annotations),
        "strata": {
            stratum: {
                "sample_rows": int(len(indices)),
                "population_rows": int(
                    annotations.loc[
                        annotations["sampling_stratum"].eq(stratum),
                        "stratum_population",
                    ].iloc[0]
                ),
                "sampling_weight": float(
                    annotations.loc[
                        annotations["sampling_stratum"].eq(stratum),
                        "sampling_weight",
                    ].iloc[0]
                ),
            }
            for stratum, indices in strata.items()
        },
        **estimates,
        "per_tag": per_tag,
        "inter_annotator_agreement": agreement,
    }
