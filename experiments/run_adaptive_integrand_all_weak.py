#!/usr/bin/env python3
"""Adaptive integrand experiment restricted to the all-weak candidate.

The baseline implementation remains in
``run_adaptive_integrand_structure_search.py``.  This entry point deliberately
overrides only candidate selection so baseline and experimental results remain
directly comparable.
"""

import builtins
import sys
import warnings

import torch

import run_adaptive_integrand_structure_search as baseline


THOROUGH_ADAPTIVE_PROFILE = "thorough_adaptive"


def enable_adaptive_thorough_profile():
    """Keep thorough budgets but bypass its legacy fixed beta schedule."""
    original_parse_args = baseline.parse_args
    original_configure = baseline.configure_speed_profile

    def parse_args():
        args = original_parse_args()
        if args.speed_profile == "thorough":
            args.speed_profile = THOROUGH_ADAPTIVE_PROFILE
        return args

    def configure_speed_profile(profile):
        original_configure(
            "thorough" if profile == THOROUGH_ADAPTIVE_PROFILE else profile
        )

    baseline.parse_args = parse_args
    baseline.configure_speed_profile = configure_speed_profile


def search_all_weak_only(
    benchmark,
    weak_edges,
    current_graph,
    current_model,
    current_metrics,
    marginal_state,
    args,
    device,
    round_id,
):
    """Train and test exactly one candidate that removes every weak edge."""
    if not weak_edges:
        print("No weak edges found; all-weak candidate skipped.", flush=True)
        return None

    print()
    print("CANDIDATE SEARCH MODE: ALL WEAK ONLY", flush=True)
    candidate = baseline.train_candidate(
        benchmark=benchmark,
        name="all_weak",
        removals=weak_edges,
        current_graph=current_graph,
        current_model=current_model,
        current_metrics=current_metrics,
        marginal_state=marginal_state,
        args=args,
        device=device,
        round_id=round_id,
    )

    if candidate["pass"]:
        print("All-weak candidate accepted; no half-edge candidates tested.", flush=True)
        return candidate

    print("All-weak candidate rejected; baseline graph retained.", flush=True)
    del candidate["model"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return None


def main():
    warnings.filterwarnings(
        "ignore",
        message=r"networkx backend defined more than once:.*",
        category=RuntimeWarning,
    )
    baseline.search_candidates = search_all_weak_only
    enable_adaptive_thorough_profile()
    if not any(arg == "--output" or arg.startswith("--output=") for arg in sys.argv):
        sys.argv.extend(["--output", "results/adaptive_integrand_all_weak"])
    print("EXPERIMENT OVERRIDE: candidate search = all weak only", flush=True)
    original_print = builtins.print

    def all_weak_print(*values, **kwargs):
        corrected = tuple(
            (
                "candidate search        = all weak only"
                if isinstance(value, str)
                and value.startswith("candidate search        = ")
                else "speed profile           = thorough (adaptive temperature)"
                if isinstance(value, str)
                and value.startswith("speed profile           = thorough_adaptive")
                else value
            )
            for value in values
        )
        original_print(*corrected, **kwargs)

    builtins.print = all_weak_print
    try:
        baseline.main()
    finally:
        builtins.print = original_print


if __name__ == "__main__":
    main()
