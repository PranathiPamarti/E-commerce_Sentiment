import unittest

import pandas as pd

from src.issue_audit import create_annotation_sample, evaluate_annotations
from src.issue_tagging import ISSUE_PATTERNS, parse_tags, tag_review
from src.reviewer_analysis import build_reviewer_features
from src.train_review_sentiment import (
    rating_sentiment,
    remove_duplicate_review_text,
)


class IssueTaggingTests(unittest.TestCase):
    def test_detects_multiple_candidate_issue_tags(self):
        tags = tag_review(
            "The bottle leaked in transit and caused a rash. The scent was too strong."
        )
        self.assertIn("leakage", tags)
        self.assertIn("rash_or_irritation", tags)
        self.assertIn("strong_fragrance", tags)

    def test_empty_text_does_not_create_tags(self):
        self.assertEqual(tag_review(""), [])
        self.assertEqual(tag_review(None), [])

    def test_annotation_parser_handles_none_and_rejects_unknown_tags(self):
        self.assertEqual(parse_tags("none"), set())
        with self.assertRaises(ValueError):
            parse_tags("not_a_real_tag")

    def test_precision_is_measured_only_on_completed_human_labels(self):
        annotations = pd.DataFrame(
            {
                "review_id": [1, 2, 3, 4],
                "sampling_stratum": [
                    "candidate_tag",
                    "candidate_tag",
                    "no_candidate_tag",
                    "no_candidate_tag",
                ],
                "stratum_population": [100, 100, 900, 900],
                "stratum_sample": [2, 2, 2, 2],
                "sampling_weight": [50, 50, 450, 450],
                "predicted_tags": ["leakage", "poor_value", "none", "none"],
                "human_tags": ["leakage", "none", "leakage", "none"],
                "annotator_2_tags": ["leakage", "none", "", ""],
            }
        )
        metrics = evaluate_annotations(annotations, bootstrap_iterations=50)
        self.assertEqual(metrics["annotated_reviews"], 4)
        self.assertAlmostEqual(metrics["micro_precision"], 0.5)
        self.assertAlmostEqual(metrics["micro_recall"], 0.1)
        self.assertEqual(metrics["true_positive_tags"], 50)
        self.assertEqual(metrics["false_positive_tags"], 50)
        self.assertEqual(metrics["false_negative_tags"], 450)
        self.assertEqual(len(metrics["micro_precision_ci_95"]), 2)
        self.assertEqual(
            metrics["inter_annotator_agreement"]["double_coded_reviews"],
            2,
        )
        self.assertEqual(
            metrics["inter_annotator_agreement"]["exact_set_agreement"],
            1.0,
        )

    def test_partial_annotation_does_not_report_population_metrics(self):
        annotations = pd.DataFrame(
            {
                "review_id": [1],
                "sampling_stratum": ["candidate_tag"],
                "stratum_population": [10],
                "stratum_sample": [1],
                "sampling_weight": [10],
                "predicted_tags": ["leakage"],
                "human_tags": [""],
            }
        )
        with self.assertRaisesRegex(ValueError, "All sampled rows"):
            evaluate_annotations(annotations, bootstrap_iterations=10)

    def test_annotation_sample_includes_tagged_and_untagged_rows(self):
        reviews = pd.DataFrame(
            {
                "review_id": [1, 2, 3, 4],
                "product_id": ["p1", "p2", "p3", "p4"],
                "reviewer_id": ["u1", "u2", "u3", "u4"],
                "review_time": pd.to_datetime(
                    ["2023-01-01"] * 4,
                    utc=True,
                ),
                "rating": [1, 2, 5, 4],
                "title": ["a", "b", "c", "d"],
                "review_text": ["leaked", "rash", "fine", "fine"],
                "issue_tags": [["leakage"], ["rash_or_irritation"], [], []],
            }
        )
        sample = create_annotation_sample(reviews, sample_size=4)
        self.assertEqual(len(sample), 4)
        self.assertEqual(sample["human_tags"].tolist(), ["", "", "", ""])
        self.assertEqual(sample["annotator_2_tags"].tolist(), ["", "", "", ""])
        self.assertIn("none", set(sample["predicted_tags"]))
        self.assertEqual(
            sample["sampling_stratum"].value_counts().to_dict(),
            {"candidate_tag": 2, "no_candidate_tag": 2},
        )
        self.assertTrue(
            (
                sample["sampling_weight"]
                == sample["stratum_population"] / sample["stratum_sample"]
            ).all()
        )

    def test_rating_labels_keep_neutral_separate(self):
        labels = rating_sentiment(pd.Series([1, 2, 3, 4, 5]))
        self.assertEqual(labels.tolist(), [0, 0, 1, 2, 2])

    def test_issue_taxonomy_has_at_least_25_tags(self):
        self.assertGreaterEqual(len(ISSUE_PATTERNS), 25)

    def test_duplicate_text_removal_is_case_and_whitespace_insensitive(self):
        frame = pd.DataFrame(
            {
                "review_id": [1, 2, 3],
                "review_time": pd.to_datetime(
                    ["2023-01-01", "2023-01-02", "2023-01-03"],
                    utc=True,
                ),
                "model_text": ["Fine product", " fine   product ", "Another review"],
            }
        )
        deduplicated, removed = remove_duplicate_review_text(frame)
        self.assertEqual(removed, 1)
        self.assertEqual(deduplicated["review_id"].tolist(), [1, 3])

    def test_reviewer_negative_share_ignores_unscored_training_rows(self):
        reviews = pd.DataFrame(
            {
                "reviewer_id": ["u1", "u1", "u2"],
                "review_time": pd.to_datetime(
                    ["2023-01-01", "2023-02-01", "2023-02-02"],
                    utc=True,
                ),
                "review_text": ["shared text", "negative text", "shared text"],
                "sentiment_label": [None, "Negative", None],
                "sentiment_is_out_of_sample": [False, True, False],
            }
        )
        features = build_reviewer_features(reviews).set_index("reviewer_id")
        self.assertEqual(features.loc["u1", "negative_share"], 1.0)
        self.assertTrue(pd.isna(features.loc["u2", "negative_share"]))

    def test_reviewer_features_handle_empty_out_of_time_sentiment(self):
        reviews = pd.DataFrame(
            {
                "reviewer_id": ["u1", "u2"],
                "review_time": pd.to_datetime(
                    ["2023-01-01", "2023-01-02"],
                    utc=True,
                ),
                "review_text": ["a", "b"],
                "sentiment_label": [None, None],
                "sentiment_is_out_of_sample": [False, False],
            }
        )
        features = build_reviewer_features(reviews)
        self.assertTrue(features["negative_share"].isna().all())
        self.assertFalse(features["unusually_negative_flag"].any())


if __name__ == "__main__":
    unittest.main()
