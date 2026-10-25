#!/usr/bin/env python3
"""Copy compact, publication-relevant result artifacts into results_public.

This script is deliberately non-destructive: it never changes or removes the
original results tree.  It also writes SHA-256 hashes for the curated files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


DEFAULT_RUNS = (
    "madgraph_tth_validation",
    "madgraph_tth_pointwise_validation",
    "madgraph_tth_sobol_highstat",
    "madgraph_tth_sobol_ultrastat",
    "madgraph_tth_madevent_precision",
    "nf_vs_madevent_tth",
    "nf_vs_madevent_tth_highstat",
    "nf_vs_madevent_tth_exact_error_audit",
    "copula_bin_sweep",
    "copula_benchmark_sweep",
)

ALLOWED_SUFFIXES = {".csv", ".json", ".png", ".pdf", ".svg", ".txt", ".md"}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results_public"))
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS)
    parser.add_argument("--max-file-mb", type=float, default=20.0)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    skipped = []
    limit = int(args.max_file_mb * 1024 * 1024)

    for run in args.runs:
        source_dir = args.source / run
        if not source_dir.is_dir():
            skipped.append({"run": run, "reason": "missing directory"})
            continue
        for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
            relative = source.relative_to(args.source)
            if source.suffix.lower() not in ALLOWED_SUFFIXES:
                skipped.append({"file": str(relative), "reason": "suffix excluded"})
                continue
            if source.stat().st_size > limit:
                skipped.append({"file": str(relative), "reason": "larger than limit"})
                continue
            destination = args.output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            manifest.append({
                "file": str(relative),
                "bytes": destination.stat().st_size,
                "sha256": digest(destination),
            })

    record = {
        "source": str(args.source.resolve()),
        "output": str(args.output.resolve()),
        "runs": list(args.runs),
        "files": manifest,
        "skipped": skipped,
    }
    (args.output / "MANIFEST.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Curated {len(manifest)} files into {args.output.resolve()}")
    print(f"Skipped {len(skipped)} files; see MANIFEST.json")


if __name__ == "__main__":
    main()
