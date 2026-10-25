#!/usr/bin/env python3
"""Compare a global conditional topology NF with its repaired specialist atlas.

The exact target sampler is used only for post-training visualization.  With no
explicit conditions, the script plots the specialist centers stored in the
atlas.  A flat ``--conditions c0 c1 c2 ...`` list may be supplied instead.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import LogNorm, TwoSlopeNorm


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_joint_topology_specialist_atlas_32d as audit  # noqa: E402
import plot_topology_target_vs_nf_32d as topology_plot  # noqa: E402
import run_topology_ladder_32d as topology  # noqa: E402
import train_joint_topology_specialist_atlas_32d as joint  # noqa: E402


COND_DIM = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atlas", type=Path, required=True)
    parser.add_argument(
        "--conditions", type=float, nargs="+",
        help="Optional flat list c0 c1 c2 repeated for every plotted condition.",
    )
    parser.add_argument("--pair", type=int, nargs=2, default=(0, 1))
    parser.add_argument("--samples", type=int, default=200_000)
    parser.add_argument(
        "--audit-samples", type=int, default=500_000,
        help="Importance samples per proposal, condition, and replicate.",
    )
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=10_000)
    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20261030)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.conditions is not None and len(args.conditions) % COND_DIM:
        parser.error("--conditions must contain groups of three numbers")
    if args.conditions is not None and any(
        value < 0.0 or value > 1.0 for value in args.conditions
    ):
        parser.error("conditions must lie in [0,1]^3")
    if min(args.samples, args.audit_samples) < 1_000:
        parser.error("sample counts must be at least 1000")
    if min(args.replicates, args.chunk) < 1:
        parser.error("--replicates and --chunk must be positive")
    if args.bins < 12:
        parser.error("--bins must be at least 12")
    if any(index < 0 or index >= topology.DIMENSION for index in args.pair):
        parser.error("--pair coordinates are outside the benchmark dimension")
    return args


def histogram2d(values, pair, edges):
    counts, _, _ = np.histogram2d(
        values[:, pair[0]], values[:, pair[1]], bins=(edges, edges)
    )
    return counts


def probability(counts):
    return counts / max(float(counts.sum()), np.finfo(float).tiny)


def total_variation(first, second):
    return 0.5 * float(np.abs(first - second).sum())


def support_metrics(values, target_p1):
    x0 = values[:, 0]
    branch0 = float(np.mean(x0 <= topology.BRANCH_WIDTH))
    branch1 = float(np.mean(x0 >= 1.0 - topology.BRANCH_WIDTH))
    gap = max(0.0, 1.0 - branch0 - branch1)
    return branch1, abs(branch1 - target_p1), gap


@torch.inference_mode()
def sample_proposals(
    adaptive, global_model, specialists, centers, gate_args,
    condition, samples,
):
    cond = condition.expand(samples, COND_DIM)
    global_values = adaptive.sample_from_model(global_model, samples, cond)
    repaired_values, _ = joint.sample_atlas(
        adaptive, global_model, specialists, cond, centers, gate_args
    )
    return global_values, repaired_values


@torch.inference_mode()
def audit_proposal(
    proposal, adaptive, global_model, specialists, centers, gate_args,
    benchmark, condition, samples, chunk,
):
    """Frozen importance-sampling audit with stable log accumulators."""
    logw_sum = None
    logw2_sum = None
    maximum_log_weight = -math.inf
    branch0 = branch1 = gap = 0
    remaining = samples
    while remaining:
        count = min(remaining, chunk)
        cond = condition.expand(count, COND_DIM)
        if proposal == "global":
            x = adaptive.sample_from_model(global_model, count, cond)
            logq = global_model.log_prob(x, cond).reshape(-1)
        elif proposal == "repaired":
            x, logq = joint.sample_atlas(
                adaptive, global_model, specialists, cond, centers, gate_args
            )
        else:
            raise ValueError(f"unknown proposal {proposal!r}")
        logf = topology.topology_log_integrand(
            benchmark, x, cond
        ).reshape(-1)
        logw = (logf - logq).double()
        part1 = torch.logsumexp(logw, dim=0).cpu()
        part2 = torch.logsumexp(2.0 * logw, dim=0).cpu()
        logw_sum = part1 if logw_sum is None else torch.logaddexp(logw_sum, part1)
        logw2_sum = (
            part2 if logw2_sum is None else torch.logaddexp(logw2_sum, part2)
        )
        maximum_log_weight = max(maximum_log_weight, float(logw.max()))
        in0 = x[:, 0] <= topology.BRANCH_WIDTH
        in1 = x[:, 0] >= 1.0 - topology.BRANCH_WIDTH
        branch0 += int(in0.sum())
        branch1 += int(in1.sum())
        gap += int((~in0 & ~in1).sum())
        remaining -= count

    logn = math.log(samples)
    integral = math.exp(float(logw_sum) - logn)
    m2 = math.exp(float(logw2_sum) - logn)
    maximum_weight = (
        math.exp(maximum_log_weight) if maximum_log_weight < 709.0 else math.inf
    )
    return {
        "integral_estimate": integral,
        "m2": m2,
        "normalized_ess": integral * integral / m2,
        "maximum_log_f_over_q": maximum_log_weight,
        "maximum_log10_f_over_q": maximum_log_weight / math.log(10.0),
        "maximum_f_over_q": maximum_weight,
        "proposal_branch_0_mass": branch0 / samples,
        "proposal_branch_1_mass": branch1 / samples,
        "proposal_gap_mass": gap / samples,
    }


def save_audit_figures(frame, conditions, output):
    """Save branch-mass and integration-quality comparisons."""
    labels = [f"c{i}" for i in range(len(conditions))]
    positions = np.arange(len(conditions), dtype=float)
    colors = {"global": "#6B7280", "repaired": "#07879B"}

    branch_figure, branch_axis = plt.subplots(
        figsize=(max(8.0, 1.35 * len(conditions)), 5.2), constrained_layout=True
    )
    target = (
        frame.groupby("condition_index")["target_branch_1_mass"].first()
        .reindex(range(len(conditions))).to_numpy()
    )
    branch_axis.plot(
        positions, target, "ko-", lw=2.0, ms=6, label="Exact target"
    )
    for offset, proposal in ((-0.08, "global"), (0.08, "repaired")):
        selected = frame[frame.proposal == proposal]
        grouped = selected.groupby("condition_index").proposal_branch_1_mass
        mean = grouped.mean().reindex(range(len(conditions))).to_numpy()
        std = grouped.std().fillna(0.0).reindex(range(len(conditions))).to_numpy()
        branch_axis.errorbar(
            positions + offset, mean, yerr=std, fmt="o", ms=7, capsize=3,
            color=colors[proposal], label=(
                "Global conditional NF" if proposal == "global"
                else "Repaired specialist atlas"
            ),
        )
    branch_axis.set_xticks(positions, labels)
    branch_axis.set_ylim(0.0, 1.0)
    branch_axis.set_ylabel("Probability mass on branch 1")
    branch_axis.set_xlabel("Condition (listed in comparison_metrics.csv)")
    branch_axis.set_title("Branch-mass comparison across conditions")
    branch_axis.grid(axis="y", alpha=0.25)
    branch_axis.legend(frameon=False, ncol=3)
    branch_figure.savefig(
        output / "branch_mass_comparison.png", dpi=220, bbox_inches="tight"
    )
    branch_figure.savefig(
        output / "branch_mass_comparison.pdf", bbox_inches="tight"
    )
    plt.close(branch_figure)

    integration_figure, integration_axes = plt.subplots(
        2, 2, figsize=(13.0, 9.0), constrained_layout=True
    )
    panels = (
        (integration_axes[0, 0], "integral_estimate", "Integral estimate", False),
        (integration_axes[0, 1], "m2", r"Second moment $M_2$", True),
        (integration_axes[1, 0], "normalized_ess", "Normalized ESS", True),
        (
            integration_axes[1, 1], "maximum_f_over_q",
            r"Largest observed $f/q$", True,
        ),
    )
    for axis, column, title, logarithmic in panels:
        for offset, proposal in ((-0.08, "global"), (0.08, "repaired")):
            selected = frame[frame.proposal == proposal]
            grouped = selected.groupby("condition_index")[column]
            center = grouped.median().reindex(range(len(conditions))).to_numpy()
            lower = grouped.min().reindex(range(len(conditions))).to_numpy()
            upper = grouped.max().reindex(range(len(conditions))).to_numpy()
            errors = np.vstack((center - lower, upper - center))
            axis.errorbar(
                positions + offset, center, yerr=errors, fmt="o", ms=7,
                capsize=3, color=colors[proposal], label=(
                    "Global NF" if proposal == "global" else "Repaired atlas"
                ),
            )
        if column == "integral_estimate":
            axis.axhline(1.0, color="black", ls="--", lw=1.5, label="Exact I=1")
        if logarithmic:
            axis.set_yscale("log")
        axis.set_xticks(positions, labels)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    integration_axes[0, 0].legend(frameon=False, ncol=3)
    integration_figure.suptitle(
        "Repeated frozen importance-sampling audit\n"
        "points show replicate medians; bars show replicate minima and maxima",
        fontsize=16,
    )
    integration_figure.savefig(
        output / "integration_weight_comparison.png", dpi=220,
        bbox_inches="tight",
    )
    integration_figure.savefig(
        output / "integration_weight_comparison.pdf", bbox_inches="tight"
    )
    plt.close(integration_figure)


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision("high")

    (
        adaptive, global_model, specialists, centers, gate_args, benchmark
    ) = audit.load_atlas(args.atlas, device)
    if args.conditions is None:
        conditions = centers.detach().cpu().numpy()
        condition_source = "specialist centers stored in the atlas"
    else:
        conditions = np.asarray(args.conditions, dtype=np.float32).reshape(-1, 3)
        condition_source = "user-supplied conditions"

    pair = tuple(args.pair)
    edges = np.linspace(0.0, 1.0, args.bins + 1)
    rows = len(conditions)
    figure, axes = plt.subplots(
        rows, 5, figsize=(21.0, 4.0 * rows), squeeze=False,
        constrained_layout=True,
    )
    metrics = []
    audit_rows = []

    for index, condition_np in enumerate(conditions):
        condition = torch.as_tensor(condition_np, device=device)
        target, _ = topology_plot.sample_exact_target(
            benchmark, condition, args.samples, device
        )
        global_values, repaired_values = sample_proposals(
            adaptive, global_model, specialists, centers, gate_args,
            condition, args.samples,
        )
        target = target.cpu().numpy()
        global_values = global_values.cpu().numpy()
        repaired_values = repaired_values.cpu().numpy()

        target_count = histogram2d(target, pair, edges)
        global_count = histogram2d(global_values, pair, edges)
        repaired_count = histogram2d(repaired_values, pair, edges)
        target_probability = probability(target_count)
        global_probability = probability(global_count)
        repaired_probability = probability(repaired_count)
        global_difference = global_probability - target_probability
        repaired_difference = repaired_probability - target_probability
        global_tv = total_variation(global_probability, target_probability)
        repaired_tv = total_variation(repaired_probability, target_probability)

        maximum = max(
            float(target_count.max()), float(global_count.max()),
            float(repaired_count.max()), 1.0,
        )
        density_norm = LogNorm(vmin=1.0, vmax=maximum)
        for column, counts, title in (
            (0, target_count, "Exact target"),
            (1, global_count, f"Global NF (TV={global_tv:.3f})"),
            (2, repaired_count, f"Repaired atlas (TV={repaired_tv:.3f})"),
        ):
            density_image = axes[index, column].imshow(
                counts.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="auto", cmap="viridis", norm=density_norm,
            )
            axes[index, column].set_title(title)

        difference_scale = max(
            float(np.abs(global_difference).max()),
            float(np.abs(repaired_difference).max()), 1e-8,
        )
        difference_norm = TwoSlopeNorm(
            vmin=-difference_scale, vcenter=0.0, vmax=difference_scale
        )
        for column, difference, title in (
            (3, global_difference, "Global NF minus target"),
            (4, repaired_difference, "Repaired atlas minus target"),
        ):
            difference_image = axes[index, column].imshow(
                difference.T, origin="lower", extent=(0, 1, 0, 1),
                aspect="auto", cmap="coolwarm", norm=difference_norm,
            )
            axes[index, column].set_title(title)

        for axis in axes[index]:
            axis.set_xlabel(fr"$x_{{{pair[0]}}}$")
            axis.set_ylabel(fr"$x_{{{pair[1]}}}$")
        axes[index, 0].text(
            0.0, 1.14,
            "condition = " + np.array2string(
                condition_np, precision=4, separator=", "
            ),
            transform=axes[index, 0].transAxes, fontsize=11,
            fontweight="bold", ha="left", va="bottom",
        )

        target_p1 = float(topology.branch_probabilities(
            benchmark, condition.reshape(1, COND_DIM)
        )[0, 1])
        global_p1, global_branch_error, global_gap = support_metrics(
            global_values, target_p1
        )
        repaired_p1, repaired_branch_error, repaired_gap = support_metrics(
            repaired_values, target_p1
        )
        gate = joint.specialist_gate(
            condition.reshape(1, COND_DIM), centers,
            gate_args.gate_length_scale, gate_args.global_fraction,
        ).reshape(-1)
        metrics.append({
            "condition_index": index,
            "c0": float(condition_np[0]),
            "c1": float(condition_np[1]),
            "c2": float(condition_np[2]),
            "pair_first": pair[0],
            "pair_second": pair[1],
            "global_pair_tv": global_tv,
            "repaired_pair_tv": repaired_tv,
            "pair_tv_change": repaired_tv - global_tv,
            "target_branch_1_mass": target_p1,
            "global_branch_1_mass": global_p1,
            "repaired_branch_1_mass": repaired_p1,
            "global_branch_mass_error": global_branch_error,
            "repaired_branch_mass_error": repaired_branch_error,
            "global_gap_mass": global_gap,
            "repaired_gap_mass": repaired_gap,
            "global_gate_weight": float(gate[0]),
            "largest_specialist_gate_weight": float(gate[1:].max()),
        })

        for replicate in range(args.replicates):
            for proposal_index, proposal in enumerate(("global", "repaired")):
                current_seed = (
                    args.seed + 100_003 * index + 1_009 * replicate
                    + 17 * proposal_index
                )
                torch.manual_seed(current_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(current_seed)
                result = audit_proposal(
                    proposal, adaptive, global_model, specialists, centers,
                    gate_args, benchmark, condition, args.audit_samples,
                    args.chunk,
                )
                result["absolute_branch_1_mass_error"] = abs(
                    result["proposal_branch_1_mass"] - target_p1
                )
                audit_rows.append({
                    "condition_index": index,
                    "c0": float(condition_np[0]),
                    "c1": float(condition_np[1]),
                    "c2": float(condition_np[2]),
                    "proposal": proposal,
                    "replicate": replicate,
                    "samples": args.audit_samples,
                    "target_branch_1_mass": target_p1,
                    **result,
                })
                print(
                    f"condition={index:02d} {proposal:8s} "
                    f"replicate={replicate + 1:02d}/{args.replicates:02d} "
                    f"I={result['integral_estimate']:.6f} "
                    f"M2={result['m2']:.6g} "
                    f"ESS={result['normalized_ess']:.4f} "
                    f"max(f/q)={result['maximum_f_over_q']:.6g}",
                    flush=True,
                )

    figure.colorbar(
        density_image, ax=axes[:, :3], label="samples per bin", shrink=0.72
    )
    figure.colorbar(
        difference_image, ax=axes[:, 3:],
        label="proposal minus target probability mass", shrink=0.72,
    )
    figure.suptitle(
        f"{benchmark}: global conditional NF versus repaired specialist atlas",
        fontsize=18,
    )
    png = args.output / "global_vs_repaired_atlas.png"
    pdf = args.output / "global_vs_repaired_atlas.pdf"
    figure.savefig(png, dpi=220, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)

    frame = pd.DataFrame(metrics)
    frame.to_csv(args.output / "comparison_metrics.csv", index=False)
    audit_frame = pd.DataFrame(audit_rows)
    audit_frame.to_csv(args.output / "importance_audit_replicates.csv", index=False)
    audit_summary = audit_frame.groupby(
        ["condition_index", "proposal"], as_index=False
    ).agg(
        replicates=("replicate", "size"),
        mean_integral=("integral_estimate", "mean"),
        maximum_absolute_integral_error=(
            "integral_estimate", lambda values: float(np.max(np.abs(values - 1.0)))
        ),
        median_m2=("m2", "median"),
        maximum_m2=("m2", "max"),
        median_normalized_ess=("normalized_ess", "median"),
        minimum_normalized_ess=("normalized_ess", "min"),
        maximum_f_over_q=("maximum_f_over_q", "max"),
        maximum_log_f_over_q=("maximum_log_f_over_q", "max"),
        mean_branch_1_mass=("proposal_branch_1_mass", "mean"),
        maximum_branch_1_mass_error=("absolute_branch_1_mass_error", "max"),
        maximum_gap_mass=("proposal_gap_mass", "max"),
    )
    audit_summary.to_csv(args.output / "importance_audit_summary.csv", index=False)
    save_audit_figures(audit_frame, conditions, args.output)
    summary = {
        "benchmark": benchmark,
        "atlas": str(args.atlas),
        "condition_source": condition_source,
        "conditions": conditions.tolist(),
        "pair": list(pair),
        "samples_per_distribution_per_condition": args.samples,
        "importance_audit_samples_per_proposal_condition_replicate": (
            args.audit_samples
        ),
        "importance_audit_replicates": args.replicates,
        "mean_global_pair_tv": float(frame.global_pair_tv.mean()),
        "mean_repaired_pair_tv": float(frame.repaired_pair_tv.mean()),
        "mean_pair_tv_change": float(frame.pair_tv_change.mean()),
        "mean_global_branch_mass_error": float(
            frame.global_branch_mass_error.mean()
        ),
        "mean_repaired_branch_mass_error": float(
            frame.repaired_branch_mass_error.mean()
        ),
        "maximum_global_gap_mass": float(frame.global_gap_mass.max()),
        "maximum_repaired_gap_mass": float(frame.repaired_gap_mass.max()),
        "exact_target_samples_used_for_training": False,
        "audit_summary": audit_summary.to_dict(orient="records"),
    }
    with (args.output / "comparison_summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nFigure: {png.resolve()}")


if __name__ == "__main__":
    main()
