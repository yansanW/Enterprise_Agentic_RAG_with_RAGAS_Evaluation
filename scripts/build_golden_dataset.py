#!/usr/bin/env python3
"""Convert an Open RAG Benchmark slice to this project's golden schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("data/openrag_slice/slice_manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("data/golden_dataset.json"))
    args = parser.parse_args()

    with args.manifest.open(encoding="utf-8") as handle:
        manifest = json.load(handle)

    golden = [
        {"question": record["question"], "ground_truth": record["golden_answer"]}
        for record in manifest["records"]
    ]
    report = {
        "output_schema": ["question", "ground_truth"],
        "mapped": {"question": "question", "golden_answer": "ground_truth"},
        "not_mapped_to_golden_schema": [
            "query_id",
            "query_type",
            "source_type",
            "doc_id",
            "section_id",
        ],
        "note": "Unmapped provenance remains in slice_manifest.json; no values were guessed or discarded there.",
        "record_count": len(golden),
    }
    write_json(args.output, golden)
    write_json(args.output.with_name("golden_dataset_mapping_report.json"), report)
    print(f"Wrote {len(golden)} records to {args.output}")


if __name__ == "__main__":
    main()
