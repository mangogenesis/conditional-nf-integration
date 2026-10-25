#!/usr/bin/env python3
"""Retrofit MadEvent XERRUP values into an existing NF-vs-MadEvent benchmark."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

def empirical_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan


def banner_cross_section(process_dir: Path, run_name: str):
    banners = sorted((process_dir / "Events" / run_name).glob("*_banner.txt"))
    if not banners:
        return None
    text = banners[-1].read_text(encoding="utf-8", errors="replace")
    init = re.search(r"<init>(.*?)</init>", text, flags=re.IGNORECASE | re.DOTALL)
    if not init:
        match = re.search(
            r"Integrated weight \(pb\)\s*:\s*([+\-0-9.eEdD]+)",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return float(match.group(1).replace("D", "E").replace("d", "e")), math.nan
        return None
    lines = [line.strip() for line in init.group(1).splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    subprocesses = []
    for line in lines[1:]:
        fields = line.replace("D", "E").replace("d", "e").split()
        if len(fields) >= 4:
            try:
                subprocesses.append((float(fields[0]), float(fields[1])))
            except ValueError:
                pass
    if not subprocesses:
        return None
    return (
        sum(value for value, _ in subprocesses),
        math.sqrt(sum(error * error for _, error in subprocesses)),
    )


def channel_log_cross_section(process_dir: Path, run_name: str):
    """Sum preserved MadEvent integration-channel logs for one named run."""
    logs = sorted(
        process_dir.glob(f"SubProcesses/P*/G*/{run_name}_log.txt")
    )
    number = r"[+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][+\-]?\d+)?"
    contributions = []
    for path in logs:
        text = path.read_text(encoding="utf-8", errors="replace")
        cross_matches = re.findall(
            rf"Cross\s+sec\s*=\s*({number})", text, flags=re.IGNORECASE
        )
        error_matches = re.findall(
            rf"Std\s+dev\s*=\s*({number})", text, flags=re.IGNORECASE
        )
        if not cross_matches or not error_matches:
            continue
        cross = float(cross_matches[-1].replace("D", "E").replace("d", "e"))
        error = float(error_matches[-1].replace("D", "E").replace("d", "e"))
        contributions.append((cross, error, path))
    if not contributions:
        return None
    return (
        sum(item[0] for item in contributions),
        math.sqrt(sum(item[1] ** 2 for item in contributions)),
        len(contributions),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--process-dir", type=Path, default=Path("generated/gg_tth_full_madevent"))
    parser.add_argument("--results", type=Path, default=Path("results/nf_vs_madevent_tth"))
    args = parser.parse_args()

    raw_path = args.results / "raw_results.csv"
    frame = pd.read_csv(raw_path)
    for column in (
        "mg_logged_channel_cross_fb",
        "mg_logged_channel_count",
        "mg_logged_cross_coverage",
    ):
        frame[column] = math.nan
    recovered = 0
    for index, row in frame[frame["method"].eq("madevent")].iterrows():
        run_name = str(row["run_name"])
        log_result = channel_log_cross_section(args.process_dir, run_name)
        if log_result is not None:
            cross_pb, error_pb, channels = log_result
            original_cross_pb = float(row["estimate_fb"]) / 1000.0
            coverage = cross_pb / original_cross_pb if original_cross_pb else math.nan
            frame.loc[index, "mg_logged_channel_cross_fb"] = cross_pb * 1000.0
            frame.loc[index, "mg_logged_channel_count"] = channels
            frame.loc[index, "mg_logged_cross_coverage"] = coverage
            banner_result = banner_cross_section(args.process_dir, run_name)
            if banner_result is not None:
                banner_cross_pb, _ = banner_result
                relative = abs(cross_pb - banner_cross_pb) / max(abs(banner_cross_pb), 1e-300)
                if relative > 0.02:
                    print(
                        f"Warning: {run_name} channel-log sum differs from banner "
                        f"by {relative:.2%} ({channels} channels found)"
                    )
        else:
            result = banner_cross_section(args.process_dir, run_name)
            if result is not None:
                cross_pb, error_pb = result
            else:
                cross_pb = error_pb = None
        if cross_pb is None:
            print(f"No MadEvent result found for {row['run_name']}")
            continue
        # Keep the benchmark's banner-derived total cross section.  Historical
        # per-channel logs can be incomplete, as quantified by the coverage
        # column, but their Std dev fields remain useful diagnostics.
        frame.loc[index, "internal_stderr_fb"] = error_pb * 1000.0
        recovered += 1

    frame.to_csv(args.results / "raw_results_with_mg_errors.csv", index=False)
    summary = frame.groupby(["energy_gev", "method"], sort=True).agg(
        repeats=("repeat", "count"),
        mean_fb=("estimate_fb", "mean"),
        empirical_sem_fb=("estimate_fb", empirical_sem),
        mean_internal_stderr_fb=("internal_stderr_fb", "mean"),
        mean_normalized_ess=("normalized_ess", "mean"),
        evaluations_per_repeat=("evaluations", "mean"),
        mean_seconds=("seconds", "mean"),
        total_seconds=("seconds", "sum"),
        mean_mg_log_coverage=("mg_logged_cross_coverage", "mean"),
        minimum_mg_log_coverage=("mg_logged_cross_coverage", "min"),
    ).reset_index()
    summary.to_csv(args.results / "summary_with_mg_errors.csv", index=False)

    rows = []
    for energy, group in summary.groupby("energy_gev"):
        indexed = group.set_index("method")
        if not {"conditional_nf", "madevent"}.issubset(indexed.index):
            continue
        nf, mg = indexed.loc["conditional_nf"], indexed.loc["madevent"]
        difference = float(nf["mean_fb"] - mg["mean_fb"])
        empirical_combined = math.hypot(
            float(nf["empirical_sem_fb"]), float(mg["empirical_sem_fb"])
        )
        # Internal errors are per repeat; divide their quadrature-combined mean
        # by sqrt(repeats) for the uncertainty on each method's reported mean.
        internal_combined = math.hypot(
            float(nf["mean_internal_stderr_fb"]) / math.sqrt(float(nf["repeats"])),
            float(mg["mean_internal_stderr_fb"]) / math.sqrt(float(mg["repeats"])),
        )
        rows.append({
            "energy_gev": energy,
            "nf_mean_fb": nf["mean_fb"], "madevent_mean_fb": mg["mean_fb"],
            "difference_fb": difference,
            "relative_difference": difference / float(mg["mean_fb"]),
            "empirical_combined_error_fb": empirical_combined,
            "empirical_pull": difference / empirical_combined,
            "internal_combined_error_fb": internal_combined,
            "internal_pull": difference / internal_combined,
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(args.results / "validation_with_mg_errors.csv", index=False)
    print(f"Recovered MadEvent errors for {recovered} runs")
    print("\nSUMMARY WITH MADEVENT ERRORS")
    print(summary.to_string(index=False))
    print("\nVALIDATION")
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
