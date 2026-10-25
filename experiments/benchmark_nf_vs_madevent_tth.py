#!/usr/bin/env python3
"""Repeated fixed-energy performance comparison of conditional NF and MadEvent."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp

import validate_madgraph_tth_integral as nfval
import validate_madgraph_tth_with_madevent as meval
from madgraph_tth_integrand import MadGraphTTHIntegrand


_ME_WORKER = None


def initialize_me_worker(standalone: str, width_window: float, alpha_s: float):
    global _ME_WORKER
    _ME_WORKER = MadGraphTTHIntegrand(
        Path(standalone), width_window=width_window, alpha_s=alpha_s
    )


def worker_log_integrand(x: np.ndarray, energy: np.ndarray) -> np.ndarray:
    if _ME_WORKER is None:
        raise RuntimeError("Matrix-element worker was not initialized")
    return _ME_WORKER.log_integrand(x, energy)


@torch.no_grad()
def nf_fixed_energy_parallel(model, energy_value, args, device, executor):
    z = (energy_value - args.sqrt_s_min) / (args.sqrt_s_max - args.sqrt_s_min)
    feature = torch.as_tensor(
        nfval.condition_features([z]), device=device, dtype=torch.float32
    )
    pending = []
    remaining = args.nf_evals
    started = time.perf_counter()
    while remaining:
        count = min(remaining, args.gpu_chunk)
        cond = feature.expand(count, nfval.COND_DIM)
        x, logq = nfval.adaptive.sample_from_proposal(
            proposal_model=model, n=count, cond=cond, device=device,
            uniform_fraction=args.uniform_fraction,
        )
        x_np = x.detach().cpu().double().numpy()
        energy = np.full(count, energy_value, dtype=np.float64)
        future = executor.submit(worker_log_integrand, x_np, energy)
        pending.append((future, logq.detach().cpu().double().numpy()))
        remaining -= count
    logw = np.concatenate([future.result() - logq for future, logq in pending])
    n = logw.size
    log_sum = float(logsumexp(logw))
    log_sum2 = float(logsumexp(2.0 * logw))
    estimate = math.exp(log_sum - math.log(n))
    second = math.exp(log_sum2 - math.log(n))
    stderr = math.sqrt(max(0.0, second - estimate * estimate) / n)
    ess = math.exp(2.0 * log_sum - math.log(n) - log_sum2)
    return {
        "estimate_fb": estimate * nfval.GEV2_TO_FB,
        "stderr_fb": stderr * nfval.GEV2_TO_FB,
        "normalized_ess": ess,
        "evaluations": n,
        "seconds": time.perf_counter() - started,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mg5", default=os.environ.get("MG5AMC", "mg5_aMC"))
    parser.add_argument("--process-dir", type=Path, default=Path("generated/gg_tth_full_madevent"))
    parser.add_argument("--standalone", type=Path, default=Path("generated/gg_tth_full_standalone"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("results/madgraph_tth_copula_thorough/madgraph_tth_20_final_model.pt"))
    parser.add_argument("--energies", type=float, nargs="+", default=(600, 800, 1000, 1400, 2000))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--nf-evals", type=int, default=100_000)
    parser.add_argument("--nevents", type=int, default=10_000)
    parser.add_argument("--sqrt-s-min", type=float, default=600.0)
    parser.add_argument("--sqrt-s-max", type=float, default=2000.0)
    parser.add_argument("--width-window", type=float, default=15.0)
    parser.add_argument("--alpha-s", type=float, default=0.118)
    parser.add_argument("--ren-scale", type=float, default=91.188)
    parser.add_argument("--required-accuracy", type=float, default=0.002)
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--me-chunk", type=int, default=512)
    parser.add_argument("--gpu-chunk", type=int, default=10_000)
    parser.add_argument("--me-workers", type=int, default=1,
                        help="parallel CPU processes for scalar MadGraph evaluations")
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--run-prefix", default="performance")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("results/nf_vs_madevent_tth"))
    return parser.parse_args()


def summarize(rows: pd.DataFrame) -> pd.DataFrame:
    def empirical_sem(values):
        values = np.asarray(values, dtype=float)
        return float(values.std(ddof=1) / math.sqrt(values.size)) if values.size > 1 else math.nan

    summary = rows.groupby(["energy_gev", "method"], sort=True).agg(
        repeats=("repeat", "count"),
        mean_fb=("estimate_fb", "mean"),
        empirical_sem_fb=("estimate_fb", empirical_sem),
        mean_internal_stderr_fb=("internal_stderr_fb", "mean"),
        mean_normalized_ess=("normalized_ess", "mean"),
        evaluations_per_repeat=("evaluations", "mean"),
        mean_seconds=("seconds", "mean"),
        total_seconds=("seconds", "sum"),
    ).reset_index()
    return summary


def main():
    args = parse_args()
    if args.repeats < 2:
        raise ValueError("Use at least two repeats to estimate empirical precision")
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    mg5 = meval.resolve_executable(args.mg5)
    meval.generate_process(mg5, args.process_dir)
    param_card = args.standalone / "Cards" / "param_card.dat"
    meval.install_param_card(param_card, args.process_dir)
    evaluator = MadGraphTTHIntegrand(
        args.standalone, width_window=args.width_window, alpha_s=args.alpha_s
    )
    model, _ = nfval.load_nf(args.checkpoint, device)

    executor = None
    if args.me_workers > 1:
        executor = ProcessPoolExecutor(
            max_workers=args.me_workers,
            mp_context=mp.get_context("spawn"),
            initializer=initialize_me_worker,
            initargs=(str(args.standalone.resolve()), args.width_window, args.alpha_s),
        )

    rows = []
    for energy in args.energies:
        for repeat in range(args.repeats):
            repeat_seed = args.seed + 100_000 * repeat + int(round(energy))
            torch.manual_seed(repeat_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(repeat_seed)
            if executor is None:
                nf = nfval.nf_fixed_energy(model, evaluator, energy, args, device)
            else:
                nf = nf_fixed_energy_parallel(model, energy, args, device, executor)
            rows.append({
                "energy_gev": energy, "repeat": repeat, "method": "conditional_nf",
                "estimate_fb": nf["estimate_fb"],
                "internal_stderr_fb": nf["stderr_fb"],
                "normalized_ess": nf["normalized_ess"],
                "evaluations": nf["evaluations"], "seconds": nf["seconds"],
            })
            pd.DataFrame(rows).to_csv(args.output / "partial_results.csv", index=False)
            print(f"NF {energy:g} GeV repeat {repeat + 1}/{args.repeats}: "
                  f"{nf['estimate_fb']:.6g} fb, {nf['seconds']:.2f} s", flush=True)

            run_args = SimpleNamespace(**vars(args))
            run_args.seed = args.seed + 100_000 * repeat
            run_args.run_prefix = f"{args.run_prefix}_r{repeat}"
            madevent_started = time.perf_counter()
            madevent = meval.run_energy(args.process_dir.resolve(), energy, run_args)
            madevent_seconds = madevent.get(
                "seconds", time.perf_counter() - madevent_started
            )
            rows.append({
                "energy_gev": energy, "repeat": repeat, "method": "madevent",
                "estimate_fb": madevent["madevent_fb"],
                "internal_stderr_fb": madevent["madevent_error_fb"],
                "normalized_ess": math.nan,
                # MadEvent does not expose a stable ME-call counter in its banner.
                "evaluations": math.nan, "seconds": madevent_seconds,
                "run_name": madevent["run_name"], "log": madevent["log"],
            })
            pd.DataFrame(rows).to_csv(args.output / "partial_results.csv", index=False)

    if executor is not None:
        executor.shutdown()

    detail = pd.DataFrame(rows)
    summary = summarize(detail)
    detail.to_csv(args.output / "raw_results.csv", index=False)
    summary.to_csv(args.output / "summary.csv", index=False)

    means = summary.pivot(index="energy_gev", columns="method", values="mean_fb")
    if {"conditional_nf", "madevent"}.issubset(means.columns):
        means["nf_minus_madevent_fb"] = means["conditional_nf"] - means["madevent"]
        means["relative_difference"] = means["nf_minus_madevent_fb"] / means["madevent"]
        means.reset_index().to_csv(args.output / "method_comparison.csv", index=False)

    config = vars(args).copy()
    config.update({key: str(value) for key, value in config.items() if isinstance(value, Path)})
    (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("\nPERFORMANCE SUMMARY")
    print(summary.to_string(index=False))
    print(f"\nResults: {args.output.resolve()}")


if __name__ == "__main__":
    main()
