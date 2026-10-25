import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from nf.conditional_cubic_1d import ConditionalCubicFlow1D

from run_copula_bin_sweep import (
    sample_target,
    target_logpdf,
    seed_everything,
)


# ============================================================
# Configuration
# ============================================================

DEFAULT_BENCHMARKS = [
    "single_gaussian",
    "sharp_broad",
    "curved_ridge",
    "xshape",
]

DEFAULT_DIMS = [8]
DEFAULT_SEEDS = [0]

MODELS = [
    "markov1",
    "markov2",
    "mi_top1",
    "mi_top2",
    "mi_top3",
    "full_ar",
]

COND_DIM = 3
N_BINS = 32
HIDDEN = 128

MARGINAL_STEPS = 1200
COPULA_STEPS = 1800

BATCH_SIZE = 4096

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4

GRAD_CLIP = 10.0

# MI graph discovery
N_GRAPH_CONDITIONS = 16
N_MI_PER_CONDITION = 25_000
MI_HIST_BINS = 32

# Evaluation
N_EVAL = 100_000
N_KL = 50_000

OUTPUT_DIR = Path(
    "results/mi_sparse_copula_benchmark"
)


TEST_CONDITIONS = torch.tensor(
    [
        [0.10, 0.10, 0.10],
        [0.10, 0.10, 0.90],
        [0.10, 0.90, 0.10],
        [0.10, 0.90, 0.90],

        [0.90, 0.10, 0.10],
        [0.90, 0.10, 0.90],
        [0.90, 0.90, 0.10],
        [0.90, 0.90, 0.90],

        [0.25, 0.25, 0.50],
        [0.25, 0.75, 0.50],
        [0.75, 0.25, 0.50],
        [0.75, 0.75, 0.50],

        [0.50, 0.25, 0.25],
        [0.50, 0.25, 0.75],
        [0.50, 0.75, 0.25],
        [0.50, 0.75, 0.75],
    ],
    dtype=torch.float32,
)


# ============================================================
# Utilities
# ============================================================

def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


def sample_conditions(n, device):
    return torch.rand(
        n,
        COND_DIM,
        device=device,
    )


def count_parameters(model):
    return sum(
        p.numel()
        for p in model.parameters()
    )


# ============================================================
# Histogram MI
# ============================================================

def estimate_pairwise_mi(
    x,
    y,
    n_bins=32,
):

    hist, _, _ = np.histogram2d(
        x,
        y,
        bins=n_bins,
        range=[
            [0.0, 1.0],
            [0.0, 1.0],
        ],
    )

    total = hist.sum()

    if total <= 0:
        return 0.0

    pxy = hist / total

    px = pxy.sum(
        axis=1,
        keepdims=True,
    )

    py = pxy.sum(
        axis=0,
        keepdims=True,
    )

    denom = px * py

    mask = (
        (pxy > 0)
        &
        (denom > 0)
    )

    mi = np.sum(
        pxy[mask]
        *
        np.log(
            pxy[mask]
            / denom[mask]
        )
    )

    return float(mi)


def estimate_mi_matrix(
    u,
    n_bins,
):

    d = u.shape[1]

    matrix = np.zeros(
        (d, d),
        dtype=np.float64,
    )

    for i in range(d):
        for j in range(
            i + 1,
            d,
        ):

            value = estimate_pairwise_mi(
                u[:, i],
                u[:, j],
                n_bins=n_bins,
            )

            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


# ============================================================
# Shared marginal model
# ============================================================

class MarginalBank(nn.Module):

    def __init__(
        self,
        dim,
        cond_dim,
        hidden,
        n_bins,
    ):

        super().__init__()

        self.dim = dim

        self.marginals = nn.ModuleList(
            [
                ConditionalCubicFlow1D(
                    cond_dim=cond_dim,
                    hidden=hidden,
                    n_bins=n_bins,
                )
                for _ in range(dim)
            ]
        )

    def x_to_u(
        self,
        x,
        cond,
    ):

        values = []

        for j in range(
            self.dim
        ):

            uj, _ = (
                self.marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            values.append(
                uj.clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            values,
            dim=1,
        )


# ============================================================
# Parent construction
# ============================================================

def build_markov_parents(
    dim,
    order,
):

    parents = {
        0: []
    }

    for j in range(
        1,
        dim,
    ):

        start = max(
            0,
            j - order,
        )

        parents[j] = list(
            range(
                start,
                j,
            )
        )

    return parents


def build_full_ar_parents(
    dim,
):

    return {
        j: list(range(j))
        for j in range(dim)
    }


def build_mi_topk_parents(
    mi_matrix,
    k,
):

    dim = mi_matrix.shape[0]

    parents = {
        0: []
    }

    for j in range(
        1,
        dim,
    ):

        candidates = [
            (
                i,
                mi_matrix[i, j],
            )
            for i in range(j)
        ]

        candidates.sort(
            key=lambda z: z[1],
            reverse=True,
        )

        selected = [
            i
            for i, _
            in candidates[:min(k, j)]
        ]

        selected.sort()

        parents[j] = selected

    return parents


# ============================================================
# Sparse copula flow
# ============================================================

class SparseCopulaFlow(nn.Module):

    def __init__(
        self,
        dim,
        cond_dim,
        hidden,
        n_bins,
        parents,
    ):

        super().__init__()

        self.dim = dim
        self.cond_dim = cond_dim
        self.parents = parents

        self.conditionals = nn.ModuleDict()

        for j in range(
            1,
            dim,
        ):

            n_parents = len(
                parents[j]
            )

            context_dim = (
                cond_dim
                + n_parents
            )

            self.conditionals[
                str(j)
            ] = ConditionalCubicFlow1D(
                cond_dim=context_dim,
                hidden=hidden,
                n_bins=n_bins,
            )

    def _context(
        self,
        u,
        cond,
        j,
    ):

        parent_idx = (
            self.parents[j]
        )

        if len(
            parent_idx
        ) == 0:

            return cond

        return torch.cat(
            [
                u[:, parent_idx],
                cond,
            ],
            dim=1,
        )

    def log_prob_u(
        self,
        u,
        cond,
    ):

        logq = torch.zeros(
            u.shape[0],
            device=u.device,
        )

        # u0 is uniform:
        # log q(u0) = 0

        for j in range(
            1,
            self.dim,
        ):

            context = self._context(
                u,
                cond,
                j,
            )

            _, forward_logdet = (
                self.conditionals[
                    str(j)
                ].inverse(
                    u[:, j:j+1],
                    context,
                )
            )

            # inverse() returns forward logdet,
            # so log density is -forward_logdet
            logq = (
                logq
                - forward_logdet.squeeze(-1)
            )

        return logq

    def sample_u(
        self,
        n,
        cond,
    ):

        if cond.shape[0] == 1:
            cond = cond.expand(
                n,
                -1,
            )

        u = torch.zeros(
            n,
            self.dim,
            device=cond.device,
        )

        u[:, 0] = torch.rand(
            n,
            device=cond.device,
        )

        for j in range(
            1,
            self.dim,
        ):

            context = self._context(
                u,
                cond,
                j,
            )

            base = torch.rand(
                n,
                1,
                device=cond.device,
            )

            uj, _ = (
                self.conditionals[
                    str(j)
                ].forward(
                    base,
                    context,
                )
            )

            u[:, j:j+1] = (
                uj.clamp(
                    0.0,
                    1.0,
                )
            )

        return u


# ============================================================
# Complete marginal + copula model
# ============================================================

class MarginalSparseCopulaFlow(nn.Module):

    def __init__(
        self,
        marginal_bank,
        parents,
        dim,
    ):

        super().__init__()

        self.dim = dim
        self.marginals = marginal_bank.marginals

        self.copula = SparseCopulaFlow(
            dim=dim,
            cond_dim=COND_DIM,
            hidden=HIDDEN,
            n_bins=N_BINS,
            parents=parents,
        )

    def x_to_u_and_log_marginal(
        self,
        x,
        cond,
    ):

        u_list = []

        log_marg = torch.zeros(
            x.shape[0],
            device=x.device,
        )

        for j in range(
            self.dim
        ):

            uj, forward_logdet = (
                self.marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            uj = uj.clamp(
                0.0,
                1.0,
            )

            u_list.append(
                uj
            )

            log_marg = (
                log_marg
                - forward_logdet.squeeze(-1)
            )

        u = torch.cat(
            u_list,
            dim=1,
        )

        return (
            u,
            log_marg,
        )

    def log_prob(
        self,
        x,
        cond,
    ):

        u, log_marg = (
            self.x_to_u_and_log_marginal(
                x,
                cond,
            )
        )

        log_copula = (
            self.copula.log_prob_u(
                u,
                cond,
            )
        )

        return (
            log_marg
            + log_copula
        )

    def sample(
        self,
        n,
        cond,
    ):

        if cond.shape[0] == 1:

            cond = cond.expand(
                n,
                -1,
            )

        u = self.copula.sample_u(
            n,
            cond,
        )

        x_list = []

        for j in range(
            self.dim
        ):

            xj, _ = (
                self.marginals[j]
                .forward(
                    u[:, j:j+1],
                    cond,
                )
            )

            x_list.append(
                xj.clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            x_list,
            dim=1,
        )


# ============================================================
# Marginal training
# ============================================================

def train_marginals(
    marginal_bank,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    optimizer = torch.optim.Adam(
        marginal_bank.parameters(),
        lr=LR_MARGINAL,
    )

    marginal_bank.train()

    recent = []

    for step in range(
        steps
    ):

        cond = sample_conditions(
            batch_size,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            dim,
        )

        loss = 0.0

        for j in range(
            dim
        ):

            _, forward_logdet = (
                marginal_bank
                .marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            loss += (
                forward_logdet.mean()
            )

        loss = loss / dim

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            marginal_bank.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"    marginals "
                f"{step+1:5d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Learn MI graph from training conditions only
# ============================================================

def discover_mi_graph(
    marginal_bank,
    benchmark,
    dim,
    device,
    n_conditions,
    n_per_condition,
    hist_bins,
):

    marginal_bank.eval()

    matrices = []

    graph_conditions = torch.rand(
        n_conditions,
        COND_DIM,
        device=device,
    )

    print(
        "\nDiscovering MI structure..."
    )

    for idx in range(
        n_conditions
    ):

        cond = (
            graph_conditions[idx]
            .reshape(1, -1)
            .expand(
                n_per_condition,
                -1,
            )
        )

        with torch.no_grad():

            x = sample_target(
                benchmark,
                cond,
                dim,
            )

            u = (
                marginal_bank
                .x_to_u(
                    x,
                    cond,
                )
                .cpu()
                .numpy()
            )

        matrix = estimate_mi_matrix(
            u,
            n_bins=hist_bins,
        )

        matrices.append(
            matrix
        )

    mean_matrix = np.mean(
        np.stack(
            matrices,
            axis=0,
        ),
        axis=0,
    )

    return (
        mean_matrix,
        graph_conditions
        .detach()
        .cpu()
        .numpy(),
    )


# ============================================================
# Copula training
# ============================================================

def train_copula(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):

    for marginal in (
        model.marginals
    ):

        for p in (
            marginal.parameters()
        ):
            p.requires_grad = False

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    model.train()

    recent = []

    for step in range(
        steps
    ):

        cond = sample_conditions(
            batch_size,
            device,
        )

        with torch.no_grad():

            x = sample_target(
                benchmark,
                cond,
                dim,
            )

            u, _ = (
                model
                .x_to_u_and_log_marginal(
                    x,
                    cond,
                )
            )

        logq_u = (
            model.copula
            .log_prob_u(
                u,
                cond,
            )
        )

        loss = (
            -logq_u.mean()
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copula.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        recent.append(
            loss.item()
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"      copula "
                f"{step+1:5d}/{steps} "
                f"loss="
                f"{np.mean(recent[-100:]):.6f}"
            )


# ============================================================
# Evaluation
# ============================================================

def estimate_forward_kl(
    model,
    benchmark,
    cond,
    dim,
    n,
):

    cond_batch = cond.expand(
        n,
        -1,
    )

    with torch.no_grad():

        x = sample_target(
            benchmark,
            cond_batch,
            dim,
        )

        logp = target_logpdf(
            benchmark,
            x,
            cond_batch,
        )

        logq = model.log_prob(
            x,
            cond_batch,
        )

        values = (
            logp - logq
        )

    return (
        values.mean().item(),
        (
            values.std(
                unbiased=True
            )
            / np.sqrt(n)
        ).item(),
    )


def evaluate_is(
    model,
    benchmark,
    cond,
    dim,
    n,
):

    cond_batch = cond.expand(
        n,
        -1,
    )

    with torch.no_grad():

        x = model.sample(
            n,
            cond,
        )

        logq = model.log_prob(
            x,
            cond_batch,
        )

        logp = target_logpdf(
            benchmark,
            x,
            cond_batch,
        )

        weights = torch.exp(
            logp - logq
        )

        estimate = (
            weights.mean()
        )

        variance = (
            weights.var(
                unbiased=True
            )
            / n
        )

        ess = (
            weights.sum() ** 2
            /
            (
                weights.square().sum()
                * n
            )
        )

        relative_error = (
            torch.abs(
                estimate - 1.0
            )
        )

    return {
        "estimate":
            estimate.item(),

        "relative_error":
            relative_error.item(),

        "variance":
            variance.item(),

        "ess":
            ess.item(),
    }


# ============================================================
# Main
# ============================================================

def run(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph_dir = (
        output_dir
        / "graphs"
    )

    graph_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"\nDevice: {device}"
    )

    raw_rows = []

    total_cases = (
        len(args.benchmarks)
        * len(args.dims)
        * len(args.seeds)
    )

    case_index = 0

    for benchmark in (
        args.benchmarks
    ):

        for dim in (
            args.dims
        ):

            for seed in (
                args.seeds
            ):

                case_index += 1

                print()
                print(
                    "=" * 80
                )

                print(
                    f"[{case_index}/"
                    f"{total_cases}] "
                    f"{benchmark} "
                    f"| d={dim} "
                    f"| seed={seed}"
                )

                print(
                    "=" * 80
                )

                seed_everything(
                    seed
                )

                # --------------------------------------------
                # Shared marginals
                # --------------------------------------------

                marginal_bank = (
                    MarginalBank(
                        dim=dim,
                        cond_dim=COND_DIM,
                        hidden=HIDDEN,
                        n_bins=N_BINS,
                    )
                    .to(device)
                )

                synchronize(device)
                t0 = time.perf_counter()

                train_marginals(
                    marginal_bank,
                    benchmark,
                    dim,
                    device,
                    args.marginal_steps,
                    args.batch_size,
                )

                synchronize(device)

                marginal_time = (
                    time.perf_counter()
                    - t0
                )

                shared_state = {
                    k:
                    v.detach()
                    .cpu()
                    .clone()

                    for k, v
                    in marginal_bank
                    .state_dict()
                    .items()
                }

                # --------------------------------------------
                # MI discovery
                # --------------------------------------------

                mi_matrix, graph_conditions = (
                    discover_mi_graph(
                        marginal_bank,
                        benchmark,
                        dim,
                        device,
                        args.graph_conditions,
                        args.mi_samples,
                        args.mi_bins,
                    )
                )

                np.save(
                    graph_dir
                    /
                    (
                        f"{benchmark}"
                        f"_d{dim}"
                        f"_seed{seed}"
                        f"_mi.npy"
                    ),
                    mi_matrix,
                )

                np.save(
                    graph_dir
                    /
                    (
                        f"{benchmark}"
                        f"_d{dim}"
                        f"_seed{seed}"
                        f"_conditions.npy"
                    ),
                    graph_conditions,
                )

                # --------------------------------------------
                # Parent structures
                # --------------------------------------------

                structures = {
                    "markov1":
                        build_markov_parents(
                            dim,
                            1,
                        ),

                    "markov2":
                        build_markov_parents(
                            dim,
                            2,
                        ),

                    "mi_top1":
                        build_mi_topk_parents(
                            mi_matrix,
                            1,
                        ),

                    "mi_top2":
                        build_mi_topk_parents(
                            mi_matrix,
                            2,
                        ),

                    "mi_top3":
                        build_mi_topk_parents(
                            mi_matrix,
                            3,
                        ),

                    "full_ar":
                        build_full_ar_parents(
                            dim
                        ),
                }

                graph_json = {}

                for (
                    name,
                    parents,
                ) in structures.items():

                    graph_json[name] = {
                        str(k): v
                        for k, v
                        in parents.items()
                    }

                with open(
                    graph_dir
                    /
                    (
                        f"{benchmark}"
                        f"_d{dim}"
                        f"_seed{seed}"
                        f"_parents.json"
                    ),
                    "w",
                ) as f:

                    json.dump(
                        graph_json,
                        f,
                        indent=2,
                    )

                # --------------------------------------------
                # Train/evaluate each structure
                # --------------------------------------------

                for model_name in (
                    args.models
                ):

                    print()
                    print(
                        f"  MODEL: "
                        f"{model_name}"
                    )

                    local_bank = (
                        MarginalBank(
                            dim=dim,
                            cond_dim=COND_DIM,
                            hidden=HIDDEN,
                            n_bins=N_BINS,
                        )
                        .to(device)
                    )

                    local_bank.load_state_dict(
                        shared_state
                    )

                    model = (
                        MarginalSparseCopulaFlow(
                            marginal_bank=
                                local_bank,
                            parents=
                                structures[
                                    model_name
                                ],
                            dim=dim,
                        )
                        .to(device)
                    )

                    seed_everything(
                        seed
                    )

                    synchronize(device)
                    t0 = time.perf_counter()

                    train_copula(
                        model,
                        benchmark,
                        dim,
                        device,
                        args.copula_steps,
                        args.batch_size,
                    )

                    synchronize(device)

                    copula_time = (
                        time.perf_counter()
                        - t0
                    )

                    parameter_count = (
                        count_parameters(
                            model
                        )
                    )

                    # ------------------------------------
                    # Held-out evaluation
                    # ------------------------------------

                    for (
                        condition_id,
                        cond,
                    ) in enumerate(
                        TEST_CONDITIONS
                    ):

                        cond = (
                            cond
                            .to(device)
                            .reshape(
                                1,
                                -1,
                            )
                        )

                        kl, kl_se = (
                            estimate_forward_kl(
                                model,
                                benchmark,
                                cond,
                                dim,
                                args.n_kl,
                            )
                        )

                        is_result = (
                            evaluate_is(
                                model,
                                benchmark,
                                cond,
                                dim,
                                args.n_eval,
                            )
                        )

                        row = {
                            "benchmark":
                                benchmark,

                            "model":
                                model_name,

                            "dimension":
                                dim,

                            "seed":
                                seed,

                            "condition_id":
                                condition_id,

                            "forward_kl":
                                kl,

                            "forward_kl_se":
                                kl_se,

                            "ess":
                                is_result[
                                    "ess"
                                ],

                            "variance":
                                is_result[
                                    "variance"
                                ],

                            "estimate":
                                is_result[
                                    "estimate"
                                ],

                            "relative_error":
                                is_result[
                                    "relative_error"
                                ],

                            "parameter_count":
                                parameter_count,

                            "marginal_train_time":
                                marginal_time,

                            "copula_train_time":
                                copula_time,

                            "total_train_time":
                                (
                                    marginal_time
                                    +
                                    copula_time
                                ),
                        }

                        raw_rows.append(
                            row
                        )

                    pd.DataFrame(
                        raw_rows
                    ).to_csv(
                        output_dir
                        / "raw_results.csv",
                        index=False,
                    )

                    del model
                    del local_bank

                    if (
                        torch.cuda.is_available()
                    ):
                        torch.cuda.empty_cache()

                del marginal_bank

                if (
                    torch.cuda.is_available()
                ):
                    torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================

    df = pd.DataFrame(
        raw_rows
    )

    summary = (
        df
        .groupby(
            [
                "benchmark",
                "model",
                "dimension",
            ]
        )
        .agg(
            ess_mean=(
                "ess",
                "mean",
            ),

            forward_kl_mean=(
                "forward_kl",
                "mean",
            ),

            variance_mean=(
                "variance",
                "mean",
            ),

            relative_error_mean=(
                "relative_error",
                "mean",
            ),

            parameter_count_mean=(
                "parameter_count",
                "mean",
            ),

            total_train_time_mean=(
                "total_train_time",
                "mean",
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        output_dir
        / "summary.csv",
        index=False,
    )

    print()
    print(
        "=" * 80
    )

    print(
        "MI-SPARSE COPULA BENCHMARK COMPLETE"
    )

    print(
        "=" * 80
    )

    print()
    print(
        summary.to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--benchmarks",
        nargs="+",
        default=
            DEFAULT_BENCHMARKS,
    )

    parser.add_argument(
        "--dims",
        nargs="+",
        type=int,
        default=
            DEFAULT_DIMS,
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=
            DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=
            MODELS,
    )

    parser.add_argument(
        "--marginal-steps",
        type=int,
        default=
            MARGINAL_STEPS,
    )

    parser.add_argument(
        "--copula-steps",
        type=int,
        default=
            COPULA_STEPS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=
            BATCH_SIZE,
    )

    parser.add_argument(
        "--graph-conditions",
        type=int,
        default=
            N_GRAPH_CONDITIONS,
    )

    parser.add_argument(
        "--mi-samples",
        type=int,
        default=
            N_MI_PER_CONDITION,
    )

    parser.add_argument(
        "--mi-bins",
        type=int,
        default=
            MI_HIST_BINS,
    )

    parser.add_argument(
        "--n-eval",
        type=int,
        default=
            N_EVAL,
    )

    parser.add_argument(
        "--n-kl",
        type=int,
        default=
            N_KL,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=
            str(
                OUTPUT_DIR
            ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    if args.smoke:

        args.benchmarks = [
            "curved_ridge",
            "sharp_broad",
        ]

        args.dims = [8]
        args.seeds = [0]

        args.models = [
            "markov1",
            "mi_top1",
            "mi_top2",
            "full_ar",
        ]

        args.marginal_steps = 100
        args.copula_steps = 150

        args.batch_size = 1024

        args.graph_conditions = 4
        args.mi_samples = 5000
        args.mi_bins = 16

        args.n_eval = 10_000
        args.n_kl = 5_000

        args.output = (
            "results/"
            "mi_sparse_copula_benchmark_smoke"
        )

    return args


if __name__ == "__main__":

    args = parse_args()
    run(args)