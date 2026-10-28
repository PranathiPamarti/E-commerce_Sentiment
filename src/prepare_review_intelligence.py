"""Download and normalize the Amazon Reviews 2023 All_Beauty data."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import shutil
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from src.issue_tagging import tag_review


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARCHIVE_PATH = DATA_DIR / "Amazon_Reviews_2023_All_Beauty.jsonl"
PARQUET_PATH = DATA_DIR / "review_intelligence.parquet"
STATS_PATH = DATA_DIR / "review_intelligence_stats.json"
DATASET_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/"
    "resolve/main/raw/review_categories/All_Beauty.jsonl?download=true"
)
BATCH_SIZE = 10_000
SCHEMA = pa.schema(
    [
        ("review_id", pa.int64()),
        ("product_id", pa.string()),
        ("reviewer_id", pa.string()),
        ("rating", pa.float32()),
        ("review_time", pa.timestamp("s", tz="UTC")),
        ("title", pa.string()),
        ("review_text", pa.string()),
        ("helpful_votes", pa.int32()),
        ("verified_purchase", pa.bool_()),
        ("issue_tags", pa.list_(pa.string())),
    ]
)


def download_archive(path: Path = ARCHIVE_PATH) -> Path:
    if path.is_file() and path.stat().st_size:
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".part")
    request = urllib.request.Request(
        DATASET_URL,
        headers={"User-Agent": "E-commerce-Review-Intelligence/1.0"},
    )
    try:
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                with temporary_path.open("wb") as target:
                    while chunk := response.read(1024 * 1024):
                        target.write(chunk)
        except urllib.error.URLError:
            curl = shutil.which("curl.exe") or shutil.which("curl")
            if curl is None:
                raise
            subprocess.run(
                [
                    curl,
                    "--fail",
                    "--location",
                    "--retry",
                    "3",
                    "--output",
                    str(temporary_path),
                    DATASET_URL,
                ],
                check=True,
                timeout=1800,
            )
        with temporary_path.open("rb") as source:
            if source.read(2) == b"\x1f\x8b":
                with gzip.open(temporary_path, "rb") as compressed:
                    compressed.read(1)
            else:
                source.seek(0)
                json.loads(source.readline())
        os.replace(temporary_path, path)
    except Exception:
        if temporary_path.exists():
            temporary_path.unlink()
        raise
    return path


def _review_datetime(timestamp: Any) -> datetime | None:
    if not isinstance(timestamp, (int, float)):
        return None
    seconds = timestamp / 1000 if timestamp > 10_000_000_000 else timestamp
    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _normalize_record(record: dict[str, Any], review_id: int) -> dict[str, Any] | None:
    product_id = record.get("parent_asin") or record.get("asin")
    reviewer_id = record.get("user_id")
    review_text = (record.get("text") or "").strip()
    title = (record.get("title") or "").strip()
    rating = record.get("rating")
    review_time = _review_datetime(record.get("timestamp"))
    if (
        not product_id
        or not reviewer_id
        or not review_text
        or review_time is None
        or not isinstance(rating, (int, float))
        or not 1 <= rating <= 5
    ):
        return None

    return {
        "review_id": review_id,
        "product_id": str(product_id),
        "reviewer_id": str(reviewer_id),
        "rating": float(rating),
        "review_time": review_time,
        "title": title,
        "review_text": review_text,
        "helpful_votes": max(0, int(record.get("helpful_vote") or 0)),
        "verified_purchase": record.get("verified_purchase"),
        "issue_tags": tag_review(f"{title}. {review_text}"),
    }


def prepare_dataset(
    archive_path: Path = ARCHIVE_PATH,
    output_path: Path = PARQUET_PATH,
    min_reviews: int = 350_000,
    min_products: int = 50_000,
    max_reviews: int | None = 400_000,
    seed: int = 42,
) -> dict[str, int | str]:
    """Prepare a reproducible random sample in bounded batches."""
    if not archive_path.is_file():
        raise FileNotFoundError(f"Review archive not found: {archive_path}")
    if max_reviews is not None and max_reviews < min_reviews:
        raise ValueError("max_reviews must be at least min_reviews.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".part")
    product_ids: set[str] = set()
    reviewer_ids: set[str] = set()
    review_count = 0
    dropped_count = 0
    writer: pq.ParquetWriter | None = None

    try:
        with archive_path.open("r", encoding="utf-8") as source:
            source_line_count = sum(1 for _ in source)
        selected_line_numbers = (
            set(
                random.Random(seed).sample(
                    range(1, source_line_count + 1),
                    min(max_reviews, source_line_count),
                )
            )
            if max_reviews is not None and max_reviews < source_line_count
            else None
        )

        with archive_path.open("rb") as binary_source:
            is_gzip = binary_source.read(2) == b"\x1f\x8b"
        opener = gzip.open if is_gzip else open
        with opener(archive_path, "rt", encoding="utf-8") as source:
            batch: list[dict[str, Any]] = []
            for line_number, line in enumerate(source, start=1):
                if not line.strip():
                    continue
                if (
                    selected_line_numbers is not None
                    and line_number not in selected_line_numbers
                ):
                    continue
                record = json.loads(line)
                normalized = _normalize_record(record, line_number)
                if normalized is None:
                    dropped_count += 1
                    continue
                batch.append(normalized)
                if len(batch) >= BATCH_SIZE:
                    table = pa.Table.from_pylist(batch, schema=SCHEMA)
                    if writer is None:
                        writer = pq.ParquetWriter(temporary_output, SCHEMA, compression="zstd")
                    writer.write_table(table)
                    product_ids.update(row["product_id"] for row in batch)
                    reviewer_ids.update(row["reviewer_id"] for row in batch)
                    review_count += len(batch)
                    batch.clear()
                    if review_count % 100_000 == 0:
                        print(f"Processed {review_count:,} reviews...")

            if batch:
                table = pa.Table.from_pylist(batch, schema=SCHEMA)
                if writer is None:
                    writer = pq.ParquetWriter(temporary_output, SCHEMA, compression="zstd")
                writer.write_table(table)
                product_ids.update(row["product_id"] for row in batch)
                reviewer_ids.update(row["reviewer_id"] for row in batch)
                review_count += len(batch)
        if writer is not None:
            writer.close()
            writer = None

        if review_count < min_reviews or len(product_ids) < min_products:
            raise ValueError(
                f"Prepared {review_count:,} reviews across {len(product_ids):,} products; "
                f"need at least {min_reviews:,} reviews and {min_products:,} products."
            )
        os.replace(temporary_output, output_path)
    except Exception:
        if writer is not None:
            writer.close()
        if temporary_output.exists():
            temporary_output.unlink()
        raise

    stats: dict[str, int | str] = {
        "dataset": "Amazon Reviews 2023 / All_Beauty",
        "source_url": "https://amazon-reviews-2023.github.io/",
        "reviews": review_count,
        "source_rows": source_line_count,
        "random_sample_seed": seed if selected_line_numbers is not None else "full dataset",
        "products": len(product_ids),
        "reviewers": len(reviewer_ids),
        "dropped_invalid_rows": dropped_count,
        "minimum_reviews_required": min_reviews,
        "minimum_products_required": min_products,
        "parquet_path": str(output_path.relative_to(PROJECT_ROOT)),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ARCHIVE_PATH)
    parser.add_argument("--output", type=Path, default=PARQUET_PATH)
    parser.add_argument("--min-reviews", type=int, default=350_000)
    parser.add_argument("--min-products", type=int, default=50_000)
    parser.add_argument(
        "--max-reviews",
        type=int,
        default=400_000,
        help="Randomly sample up to this many source rows (default: 400000).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    archive_path = args.archive if args.archive.is_file() else download_archive(args.archive)
    stats = prepare_dataset(
        archive_path,
        args.output,
        min_reviews=args.min_reviews,
        min_products=args.min_products,
        max_reviews=args.max_reviews,
        seed=args.seed,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
