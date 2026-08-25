#!/usr/bin/env python3
"""Prepare a small, deterministic slice of vectara/open_ragbench."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from datasets import DownloadConfig, DownloadManager

REVISION = "63f6b052ff83508b08e242db42263ee708815c26"
REPO_BASE = f"https://huggingface.co/datasets/vectara/open_ragbench/resolve/{REVISION}/pdf/arxiv"
SOURCE_ORDER = ("text", "text-image", "text-table", "text-table-image")


def read_json(path: str | Path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def select_balanced(queries: dict, qrels: dict, answers: dict, size: int) -> list[dict]:
    if size < len(SOURCE_ORDER):
        raise ValueError(f"--size must be at least {len(SOURCE_ORDER)}")

    candidates: dict[str, list[str]] = {source: [] for source in SOURCE_ORDER}
    for query_id in sorted(set(queries) & set(qrels) & set(answers)):
        source = queries[query_id].get("source")
        if source in candidates:
            candidates[source].append(query_id)

    base, remainder = divmod(size, len(SOURCE_ORDER))
    selected: list[str] = []
    for index, source in enumerate(SOURCE_ORDER):
        quota = base + (1 if index < remainder else 0)
        if len(candidates[source]) < quota:
            raise ValueError(f"Only {len(candidates[source])} candidates available for {source}")
        selected.extend(candidates[source][:quota])

    # Stable interleaving avoids grouping the saved slice by modality.
    selected.sort()
    return [
        {
            "query_id": query_id,
            "question": queries[query_id]["query"],
            "golden_answer": answers[query_id],
            "query_type": queries[query_id].get("type"),
            "source_type": queries[query_id].get("source"),
            "doc_id": qrels[query_id]["doc_id"],
            "section_id": qrels[query_id]["section_id"],
        }
        for query_id in selected
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=Path("data/openrag_slice"))
    parser.add_argument("--skip-pdfs", action="store_true")
    args = parser.parse_args()

    raw_dir = args.output_dir / "raw"
    corpus_dir = args.output_dir / "corpus"
    documents_dir = args.output_dir / "documents"
    for directory in (raw_dir, corpus_dir, documents_dir):
        directory.mkdir(parents=True, exist_ok=True)

    manager = DownloadManager(
        download_config=DownloadConfig(cache_dir=raw_dir / ".cache")
    )
    metadata_urls = {
        name: f"{REPO_BASE}/{name}.json"
        for name in ("queries", "qrels", "answers", "pdf_urls")
    }
    downloaded = manager.download(metadata_urls)
    metadata = {name: read_json(path) for name, path in downloaded.items()}
    for name, payload in metadata.items():
        write_json(raw_dir / f"{name}.json", payload)

    records = select_balanced(
        metadata["queries"], metadata["qrels"], metadata["answers"], args.size
    )
    doc_ids = sorted({record["doc_id"] for record in records})

    corpus_urls = {
        doc_id: f"{REPO_BASE}/corpus/{doc_id}.json" for doc_id in doc_ids
    }
    corpus_paths = manager.download(corpus_urls)
    for doc_id, path in corpus_paths.items():
        shutil.copyfile(path, corpus_dir / f"{doc_id}.json")

    pdf_status: dict[str, dict] = {}
    if args.skip_pdfs:
        pdf_status = {doc_id: {"status": "skipped"} for doc_id in doc_ids}
    else:
        pdf_urls = {
            doc_id: metadata["pdf_urls"][doc_id]
            for doc_id in doc_ids
            if doc_id in metadata["pdf_urls"]
        }
        missing_urls = sorted(set(doc_ids) - set(pdf_urls))
        if missing_urls:
            raise KeyError(f"Missing PDF URLs for: {', '.join(missing_urls)}")
        pdf_paths = manager.download(pdf_urls)
        for doc_id, path in pdf_paths.items():
            destination = documents_dir / f"{doc_id}.pdf"
            shutil.copyfile(path, destination)
            pdf_status[doc_id] = {
                "status": "downloaded",
                "path": str(destination),
                "url": pdf_urls[doc_id],
            }

    manifest = {
        "dataset": "vectara/open_ragbench",
        "dataset_revision": REVISION,
        "selection": "lexicographically first queries, balanced across source_type",
        "requested_size": args.size,
        "qa_count": len(records),
        "document_count": len(doc_ids),
        "source_type_counts": dict(Counter(r["source_type"] for r in records)),
        "query_type_counts": dict(Counter(r["query_type"] for r in records)),
        "records": records,
        "pdfs": pdf_status,
    }
    write_json(args.output_dir / "slice_manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in ("qa_count", "document_count", "source_type_counts", "query_type_counts")}, indent=2))


if __name__ == "__main__":
    main()
