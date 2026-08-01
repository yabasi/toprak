# Copyright (c) 2026 Abbas Kandemir (@yabasi)
# Licensed under the Apache License, Version 2.0. See LICENSE file in the project root.

"""Toprak JSONL korpuslarında provenance ve kalite metadata denetimi."""

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.governance import SCHEMA_VERSION, content_sha256, utc_now_iso


REQUIRED_FIELDS = {
    "schema_version",
    "source",
    "dataset_id",
    "dataset_revision",
    "licenses",
    "license_status",
    "downloaded_at",
    "content_sha256",
    "quality_score",
    "quality_signals",
    "pii_redactions",
    "contamination_matches",
}


def iter_jsonl_files(path: str):
    if os.path.isfile(path):
        yield path
        return
    for root, _, filenames in os.walk(path):
        for filename in sorted(filenames):
            if filename.endswith(".jsonl"):
                yield os.path.join(root, filename)


def audit_corpus(path: str, verify_hashes: bool = True) -> dict:
    totals = Counter()
    sources = Counter()
    license_statuses = Counter()
    missing_fields = Counter()
    seen_hashes = set()
    files = []

    for file_path in iter_jsonl_files(path):
        file_totals = Counter()
        with open(file_path, "r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                totals["lines"] += 1
                file_totals["lines"] += 1
                try:
                    document = json.loads(line)
                except json.JSONDecodeError:
                    totals["invalid_json"] += 1
                    file_totals["invalid_json"] += 1
                    continue

                missing = REQUIRED_FIELDS - document.keys()
                for field in missing:
                    missing_fields[field] += 1
                if missing:
                    totals["missing_metadata"] += 1

                if document.get("schema_version") != SCHEMA_VERSION:
                    totals["schema_mismatch"] += 1
                sources[document.get("source", "unknown")] += 1
                status = document.get("license_status", "missing")
                license_statuses[status] += 1
                if status in {"review_required", "inherited_review_required"}:
                    totals["license_review_required"] += 1
                if document.get("pii_redactions"):
                    totals["pii_redacted_documents"] += 1
                if document.get("contamination_matches"):
                    totals["contaminated_documents"] += 1

                digest = document.get("content_sha256")
                if digest in seen_hashes:
                    totals["duplicate_hashes"] += 1
                elif digest:
                    seen_hashes.add(digest)
                if verify_hashes and digest and digest != content_sha256(document.get("text", "")):
                    totals["hash_mismatch"] += 1

        files.append({"path": file_path, **dict(file_totals)})

    return {
        "schema_version": SCHEMA_VERSION,
        "audited_at": utc_now_iso(),
        "root": os.path.abspath(path),
        "totals": dict(totals),
        "sources": dict(sorted(sources.items())),
        "license_statuses": dict(sorted(license_statuses.items())),
        "missing_fields": dict(sorted(missing_fields.items())),
        "files": files,
    }


def main():
    parser = argparse.ArgumentParser(description="Toprak korpus metadata denetimi")
    parser.add_argument("path", help="JSONL dosyası veya korpus dizini")
    parser.add_argument("--output", default=None, help="JSON rapor dosyası")
    parser.add_argument("--no-verify-hashes", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Eksik metadata, hash/şema hatası veya duplicate varsa hata koduyla çık",
    )
    args = parser.parse_args()

    report = audit_corpus(args.path, verify_hashes=not args.no_verify_hashes)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)

    blocking = sum(
        report["totals"].get(key, 0)
        for key in (
            "invalid_json",
            "missing_metadata",
            "schema_mismatch",
            "hash_mismatch",
            "duplicate_hashes",
        )
    )
    if args.strict and blocking:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
