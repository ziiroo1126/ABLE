#!/usr/bin/env python3
"""Rebuild the paper's 1,200-example probe corpus without redistributing GPQA.

The repository publishes 1,000 non-GPQA examples and a content-free manifest
of the 200 GPQA source indices used by the paper. Users must obtain GPQA from
its official gated Hugging Face repository after accepting its access terms.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "selected_data"
DEFAULT_PUBLIC_DATA = DATA_DIR / "ABLE_dataset_public_1000.jsonl"
DEFAULT_MANIFEST = DATA_DIR / "gpqa_selection_manifest.json"
DEFAULT_OUTPUT = DATA_DIR / "ABLE_dataset_1200.jsonl"

GPQA_PROMPT = (
    "Given the following question and four candidate choices, choose the best one. "
    "Your response should end with one of A, B, C or D.\n"
    "Question: {question}\n"
    "Choices:\n"
    "A. {correct}\n"
    "B. {incorrect_1}\n"
    "C. {incorrect_2}\n"
    "D. {incorrect_3}\n"
    "Answer:"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-data", type=Path, default=DEFAULT_PUBLIC_DATA)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--gpqa-source",
        type=Path,
        default=None,
        help=(
            "Optional authorized local GPQA .csv or .jsonl file. When omitted, "
            "the script loads the official gated Hugging Face dataset."
        ),
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Override the Hugging Face dataset revision recorded in the manifest.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_local_gpqa(path: Path) -> Dict[int, Dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    elif path.suffix.lower() == ".jsonl":
        rows = read_jsonl(path)
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload["data"]
    else:
        raise ValueError("--gpqa-source must be a .csv, .json, or .jsonl file")

    indexed: Dict[int, Dict[str, Any]] = {}
    for position, row in enumerate(rows):
        source_index = int(row.get("original_idx", position))
        indexed[source_index] = row
    return indexed


def load_hub_gpqa(manifest: Dict[str, Any], revision: str | None) -> Dict[int, Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for gated download. Install it "
            "with 'pip install datasets' or pass --gpqa-source."
        ) from exc

    source = manifest["source"]
    resolved_revision = revision or source.get("revision")
    try:
        dataset = load_dataset(
            source["repository"],
            source["configuration"],
            split=source["split"],
            revision=resolved_revision,
        )
    except Exception as exc:
        raise RuntimeError(
            "Unable to access the official GPQA dataset. Visit the dataset page, "
            "accept its conditions, authenticate with 'hf auth login', and retry."
        ) from exc
    return {index: dict(row) for index, row in enumerate(dataset)}


def make_gpqa_record(selection: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    required = (
        "Question",
        "Correct Answer",
        "Incorrect Answer 1",
        "Incorrect Answer 2",
        "Incorrect Answer 3",
    )
    missing = [field for field in required if field not in source]
    if missing:
        raise KeyError(f"GPQA source record is missing fields: {', '.join(missing)}")

    return {
        "index": selection["index"],
        "split": selection["split"],
        "original_idx": selection["original_idx"],
        "resource": "gpqa",
        "question": GPQA_PROMPT.format(
            question=source["Question"],
            correct=source["Correct Answer"],
            incorrect_1=source["Incorrect Answer 1"],
            incorrect_2=source["Incorrect Answer 2"],
            incorrect_3=source["Incorrect Answer 3"],
        ),
        "choices": ["A", "B", "C", "D"],
        "ans_idx": 0,
    }


def canonical_sha256(records: Iterable[Dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        canonical = json.dumps(
            record, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
        digest.update(canonical.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_public_records(records: list[Dict[str, Any]]) -> None:
    if len(records) != 1000:
        raise ValueError(f"Expected 1,000 public records, found {len(records)}")
    if any(record.get("resource") == "gpqa" for record in records):
        raise ValueError("Public data unexpectedly contains GPQA records")
    if [record["index"] for record in records] != list(range(1000)):
        raise ValueError("Public record indices must be contiguous from 0 to 999")
    full_indices = [record.get("full_index") for record in records]
    if len(set(full_indices)) != 1000 or any(index is None for index in full_indices):
        raise ValueError("Public records must contain unique full_index values")


def write_jsonl_atomic(records: list[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, delete=False
    ) as handle:
        temporary_path = Path(handle.name)
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    os.replace(temporary_path, output)


def main() -> int:
    args = parse_args()
    public_records = read_jsonl(args.public_data)
    validate_public_records(public_records)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    selections = manifest["records"]
    if len(selections) != 200:
        raise ValueError(f"Expected 200 GPQA selections, found {len(selections)}")

    if args.gpqa_source is not None:
        gpqa_by_index = load_local_gpqa(args.gpqa_source)
    else:
        gpqa_by_index = load_hub_gpqa(manifest, args.revision)

    gpqa_records = []
    for selection in selections:
        source_index = int(selection["original_idx"])
        if source_index not in gpqa_by_index:
            raise KeyError(f"GPQA source does not contain index {source_index}")
        gpqa_records.append(make_gpqa_record(selection, gpqa_by_index[source_index]))

    full_records = []
    for public_record in public_records:
        restored = dict(public_record)
        restored["index"] = restored.pop("full_index")
        full_records.append(restored)
    full_records.extend(gpqa_records)
    full_records.sort(key=lambda record: record["index"])

    if [record["index"] for record in full_records] != list(range(1200)):
        raise ValueError("Reconstructed full indices must be contiguous from 0 to 1199")

    actual_hash = canonical_sha256(full_records)
    expected_hash = manifest["expected_canonical_sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            "Reconstructed data do not match the corpus used in the paper. "
            f"Expected canonical SHA-256 {expected_hash}, got {actual_hash}. "
            "Check the GPQA source revision and do not use the mismatched output."
        )

    write_jsonl_atomic(full_records, args.output)
    print(f"Wrote {len(full_records)} records to {args.output}")
    print(f"Canonical SHA-256: {actual_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
