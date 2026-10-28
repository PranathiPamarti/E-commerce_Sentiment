from pathlib import Path

import pandas as pd
import streamlit as st

from src.issue_audit import create_annotation_sample, evaluate_annotations
from src.issue_tagging import ISSUE_PATTERNS, TAG_DESCRIPTIONS
from src.train_review_sentiment import SCORED_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANNOTATION_PATH = PROJECT_ROOT / "data" / "issue_tag_annotations.csv"


@st.cache_data(show_spinner="Loading candidate reviews...")
def load_reviews(data_version):
    if not data_version:
        raise ValueError("A data version is required to load candidate reviews.")
    return pd.read_parquet(SCORED_DATA_PATH)


st.title("Issue Tag Review & Precision Audit")
st.caption(
    "Review candidate tags, record human labels, and measure precision on the "
    "annotated sample. Do not treat unreviewed candidates as ground truth."
)

if not SCORED_DATA_PATH.is_file():
    st.info(
        "Prepare and score the public dataset first. Run "
        "`python -m src.prepare_review_intelligence`, then "
        "`python -m src.train_review_sentiment`."
    )
    st.stop()

data_stat = SCORED_DATA_PATH.stat()
data_version = (data_stat.st_mtime_ns, data_stat.st_size)
try:
    reviews = load_reviews(data_version)
except (OSError, ValueError, ImportError) as error:
    st.error(f"Could not load prepared reviews: {error}")
    st.stop()

with st.expander("Issue-tag reference", expanded=False):
    st.dataframe(
        pd.DataFrame(
            [
                {"Tag": tag, "Description": TAG_DESCRIPTIONS[tag]}
                for tag in ISSUE_PATTERNS
            ]
        ),
        hide_index=True,
        width="stretch",
    )

if not ANNOTATION_PATH.is_file():
    sample_size = st.number_input(
        "Reviews to include in the human audit sample",
        min_value=50,
        max_value=2000,
        value=400,
        step=50,
    )
    if st.button("Create balanced annotation sample", type="primary"):
        try:
            sample = create_annotation_sample(reviews, int(sample_size))
            ANNOTATION_PATH.parent.mkdir(parents=True, exist_ok=True)
            sample.to_csv(ANNOTATION_PATH, index=False)
            st.rerun()
        except (ValueError, OSError) as error:
            st.error(f"Could not create annotation sample: {error}")
            st.stop()
    st.info(
        "The stratified random sample includes candidate-tagged and untagged "
        "reviews, oversampling candidates and weighting both groups back to their "
        "population sizes. Human labels are required before precision is calculated."
    )
    st.stop()

try:
    annotations = pd.read_csv(
        ANNOTATION_PATH,
        dtype={"human_tags": "string", "predicted_tags": "string"},
        keep_default_na=False,
    )
except (OSError, ValueError, pd.errors.ParserError) as error:
    st.error(f"Could not read the annotation file: {error}")
    st.stop()

required_columns = {
    "review_id",
    "product_id",
    "reviewer_id",
    "review_time",
    "rating",
    "title",
    "review_text",
    "predicted_tags",
    "human_tags",
}
missing_columns = required_columns.difference(annotations.columns)
if missing_columns:
    st.error(
        "The annotation file is missing required columns: "
        f"{', '.join(sorted(missing_columns))}."
    )
    st.stop()
if "annotator_2_tags" not in annotations:
    annotations["annotator_2_tags"] = ""

primary_reviewed = annotations["human_tags"].astype(str).str.strip().ne("")
primary_count = int(primary_reviewed.sum())
st.progress(primary_count / max(1, len(annotations)))
st.caption(f"{primary_count:,} of {len(annotations):,} reviews have primary human labels.")

if primary_count == len(annotations):
    try:
        metrics = evaluate_annotations(annotations)
    except ValueError as error:
        st.error(f"Could not evaluate annotations: {error}")
        st.stop()
    metrics_columns = st.columns(4)
    metrics_columns[0].metric("Annotated reviews", f"{metrics['annotated_reviews']:,}")
    metrics_columns[1].metric("Issue-tag precision", f"{metrics['micro_precision']:.1%}")
    metrics_columns[2].metric("Issue-tag recall", f"{metrics['micro_recall']:.1%}")
    metrics_columns[3].metric("Micro F1", f"{metrics['micro_f1']:.1%}")
    precision_low, precision_high = metrics["micro_precision_ci_95"]
    st.caption(
        "Population-weighted estimate from the documented stratified sample. "
        f"Approximate 95% within-stratum bootstrap CI for precision: "
        f"{precision_low:.1%}–{precision_high:.1%}."
    )
    per_tag = metrics["per_tag"]
    if per_tag:
        per_tag_frame = pd.DataFrame.from_dict(per_tag, orient="index")
        per_tag_frame.index.name = "issue_tag"
        st.dataframe(
            per_tag_frame.reset_index().sort_values("estimated_precision"),
            hide_index=True,
            width="stretch",
        )
    agreement = metrics["inter_annotator_agreement"]
    if agreement["double_coded_reviews"]:
        agreement_columns = st.columns(2)
        agreement_columns[0].metric(
            "Double-coded reviews",
            f"{agreement['double_coded_reviews']:,}",
        )
        agreement_columns[1].metric(
            "Exact annotator agreement",
            f"{agreement['exact_set_agreement']:.1%}",
        )
        if agreement["mean_per_tag_cohen_kappa"] is not None:
            st.caption(
                "Mean Cohen's kappa across issue tags with non-degenerate labels: "
                f"{agreement['mean_per_tag_cohen_kappa']:.3f}."
            )
else:
    st.info(
        "Population-weighted precision and recall are shown only after every "
        "randomly sampled review has a completed human label."
    )

annotation_role = st.radio(
    "Annotation pass",
    ["Primary annotator (blind)", "Independent second annotator (blind)"],
    horizontal=True,
)
target_column = (
    "human_tags"
    if annotation_role == "Primary annotator (blind)"
    else "annotator_2_tags"
)
role_reviewed = annotations[target_column].astype(str).str.strip().ne("")
unreviewed_positions = annotations.index[~role_reviewed].tolist()
positions = unreviewed_positions or annotations.index.tolist()
position = st.selectbox(
    "Review to annotate",
    positions,
    format_func=lambda index: (
        f"Review {annotations.loc[index, 'review_id']} · "
        f"{'unreviewed' if index in unreviewed_positions else 'already labeled'}"
    ),
)
record = annotations.loc[position]
st.write(f"**Product:** {record['product_id']} · **Rating:** {record['rating']}")
st.write(f"**Reviewer:** {record['reviewer_id']} · **Date:** {record['review_time']}")
st.markdown(f"**{record['title']}**")
st.write(record["review_text"])

if annotation_role == "Independent second annotator (blind)":
    st.caption(
        "Primary labels and automated tags are hidden during this independent pass."
    )
elif str(record[target_column]).strip():
    predicted_tags = [
        tag
        for tag in record["predicted_tags"].split("|")
        if tag and tag.lower() != "none"
    ]
    with st.expander("Reveal candidate tags after primary annotation"):
        st.write(
            "**Automated candidate tags:** "
            + (
                ", ".join(f"`{tag}`" for tag in predicted_tags)
                if predicted_tags
                else "No candidate tag"
            )
        )

existing = {
    tag.strip()
    for tag in str(record[target_column]).split("|")
    if tag.strip() and tag.strip().lower() != "none"
}
human_tags = st.multiselect(
    "Human-confirmed issue tags",
    options=list(ISSUE_PATTERNS),
    default=sorted(existing),
    format_func=lambda tag: f"{tag} — {TAG_DESCRIPTIONS[tag]}",
    key=f"human_tags_{target_column}_{record['review_id']}",
)
no_issue = st.checkbox(
    "No recurring issue is present in this review",
    value=str(record[target_column]).strip().lower() == "none",
    key=f"no_issue_{target_column}_{record['review_id']}",
)
if st.button("Save human annotation", type="primary"):
    if bool(human_tags) == no_issue:
        st.error("Select one or more issue tags, or mark the review as having no issue.")
    else:
        annotations.loc[position, target_column] = (
            "|".join(sorted(human_tags)) if human_tags else "none"
        )
        temporary_path = ANNOTATION_PATH.with_suffix(".csv.part")
        try:
            annotations.to_csv(temporary_path, index=False)
            temporary_path.replace(ANNOTATION_PATH)
        except OSError as error:
            st.error(f"Could not save the human annotation: {error}")
            st.stop()
        st.rerun()

st.download_button(
    "Download current human-annotation data",
    annotations.to_csv(index=False).encode("utf-8"),
    file_name="issue_tag_annotations.csv",
    mime="text/csv",
)
