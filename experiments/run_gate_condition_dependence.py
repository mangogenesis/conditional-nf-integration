import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
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

COND_DIM = 3
DIM = 8

N_BINS = 32
HIDDEN = 128
GATE_HIDDEN = 64

MARGINAL_STEPS = 1200
COPULA_STEPS = 2500

BATCH_SIZE = 4096

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4

GRAD_CLIP = 10.0

PRIOR_EDGE_PROB = 0.20

TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.20

BETA_WARMUP_FRAC = 0.20

# benchmark-specific beta choices
DEFAULT_BETAS = {
    "curved_ridge": 0.02,
    "sharp_broad": 0.005,
}

# Dense random condition sample used to measure gate variation
N_DIAGNOSTIC_CONDITIONS = 5000

# One-dimensional condition sweeps
N_CONDITION_GRID = 101

N_EVAL = 100_000
N_KL = 50_000

OUTPUT_DIR = Path(
    "results/gate_condition_dependence"
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


def linear_schedule(
    step,
    total_steps,
    start,
    end,
):
    if total_steps <= 1:
        return end

    frac = step / (total_steps - 1)

    return (
        start
        + frac * (end - start)
    )


def beta_schedule(
    step,
    total_steps,
    beta_final,
):
    warmup_steps = int(
        BETA_WARMUP_FRAC
        * total_steps
    )

    if step < warmup_steps:
        return 0.0

    remaining = max(
        total_steps - warmup_steps,
        1,
    )

    frac = (
        step - warmup_steps
    ) / remaining

    frac = np.clip(
        frac,
        0.0,
        1.0,
    )

    return (
        beta_final * frac
    )


# ============================================================
# Marginals
# ============================================================

class MarginalBank(nn.Module):

    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.marginals = nn.ModuleList(
            [
                ConditionalCubicFlow1D(
                    cond_dim=COND_DIM,
                    hidden=HIDDEN,
                    n_bins=N_BINS,
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
# Gate network
# ============================================================

class GateNetwork(nn.Module):

    def __init__(
        self,
        dim,
        conditional,
        prior_edge_prob,
    ):
        super().__init__()

        self.dim = dim
        self.conditional = conditional

        self.n_edges = (
            dim * (dim - 1) // 2
        )

        self.edge_pairs = []

        for j in range(
            1,
            dim,
        ):
            for i in range(j):
                self.edge_pairs.append(
                    (i, j)
                )

        prior_logit = math.log(
            prior_edge_prob
            /
            (
                1.0
                - prior_edge_prob
            )
        )

        if conditional:

            self.net = nn.Sequential(
                nn.Linear(
                    COND_DIM,
                    GATE_HIDDEN,
                ),
                nn.Tanh(),

                nn.Linear(
                    GATE_HIDDEN,
                    GATE_HIDDEN,
                ),
                nn.Tanh(),

                nn.Linear(
                    GATE_HIDDEN,
                    self.n_edges,
                ),
            )

            # Start exactly at prior
            final_layer = (
                self.net[-1]
            )

            nn.init.zeros_(
                final_layer.weight
            )

            nn.init.constant_(
                final_layer.bias,
                prior_logit,
            )

        else:

            self.logits = nn.Parameter(
                torch.full(
                    (
                        self.n_edges,
                    ),
                    prior_logit,
                    dtype=torch.float32,
                )
            )

    def forward(
        self,
        cond,
    ):
        if self.conditional:
            return self.net(
                cond
            )

        return (
            self.logits
            .unsqueeze(0)
            .expand(
                cond.shape[0],
                -1,
            )
        )


# ============================================================
# Binary Concrete / KL
# ============================================================

def sample_binary_concrete(
    logits,
    temperature,
):
    eps = 1e-6

    uniform = (
        torch.rand_like(
            logits
        )
        .clamp(
            eps,
            1.0 - eps,
        )
    )

    logistic_noise = (
        torch.log(
            uniform
        )
        - torch.log1p(
            -uniform
        )
    )

    return torch.sigmoid(
        (
            logits
            + logistic_noise
        )
        / temperature
    )


def bernoulli_kl(
    probs,
    prior_prob,
):
    eps = 1e-7

    probs = probs.clamp(
        eps,
        1.0 - eps,
    )

    prior = torch.tensor(
        prior_prob,
        dtype=probs.dtype,
        device=probs.device,
    )

    return (
        probs
        * (
            torch.log(
                probs
            )
            - torch.log(
                prior
            )
        )
        +
        (
            1.0 - probs
        )
        * (
            torch.log(
                1.0 - probs
            )
            - torch.log(
                1.0 - prior
            )
        )
    )


# ============================================================
# Bayesian gated copula
# ============================================================

class BayesianGatedCopula(nn.Module):

    def __init__(
        self,
        dim,
        conditional,
        prior_edge_prob,
    ):
        super().__init__()

        self.dim = dim

        self.gates = GateNetwork(
            dim=dim,
            conditional=conditional,
            prior_edge_prob=
                prior_edge_prob,
        )

        self.conditionals = (
            nn.ModuleDict()
        )

        self.edge_index = {}

        edge_counter = 0

        for j in range(
            1,
            dim,
        ):

            indices = []

            for _ in range(j):
                indices.append(
                    edge_counter
                )

                edge_counter += 1

            self.edge_index[
                j
            ] = indices

            context_dim = (
                2 * j
                + COND_DIM
            )

            self.conditionals[
                str(j)
            ] = ConditionalCubicFlow1D(
                cond_dim=context_dim,
                hidden=HIDDEN,
                n_bins=N_BINS,
            )

    def gate_logits(
        self,
        cond,
    ):
        return self.gates(
            cond
        )

    def gate_probabilities(
        self,
        cond,
    ):
        return torch.sigmoid(
            self.gate_logits(
                cond
            )
        )

    def _context(
        self,
        u,
        cond,
        gates,
        j,
    ):
        idx = (
            self.edge_index[j]
        )

        gate_j = gates[
            :,
            idx
        ]

        previous_u = (
            u[:, :j]
        )

        centered_u = (
            previous_u
            - 0.5
        )

        gated_u = (
            gate_j
            * centered_u
        )

        return torch.cat(
            [
                gated_u,
                gate_j,
                cond,
            ],
            dim=1,
        )

    def log_prob_u(
        self,
        u,
        cond,
        gates,
    ):
        logq = torch.zeros(
            u.shape[0],
            device=u.device,
        )

        for j in range(
            1,
            self.dim,
        ):

            context = (
                self._context(
                    u,
                    cond,
                    gates,
                    j,
                )
            )

            _, forward_logdet = (
                self.conditionals[
                    str(j)
                ].inverse(
                    u[:, j:j+1],
                    context,
                )
            )

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

        gates = (
            self.gate_probabilities(
                cond
            )
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

            context = (
                self._context(
                    u,
                    cond,
                    gates,
                    j,
                )
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
# Combined model
# ============================================================

class MarginalCopulaModel(nn.Module):

    def __init__(
        self,
        marginal_bank,
        copula,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.marginals = (
            marginal_bank.marginals
        )

        self.copula = copula

    def x_to_u_and_log_marginal(
        self,
        x,
        cond,
    ):
        values = []

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

            values.append(
                uj
            )

            log_marg = (
                log_marg
                - forward_logdet.squeeze(-1)
            )

        return (
            torch.cat(
                values,
                dim=1,
            ),
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

        gates = (
            self.copula
            .gate_probabilities(
                cond
            )
        )

        log_copula = (
            self.copula
            .log_prob_u(
                u,
                cond,
                gates,
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

        cond_batch = cond.expand(
            n,
            -1,
        )

        u = (
            self.copula
            .sample_u(
                n,
                cond_batch,
            )
        )

        values = []

        for j in range(
            self.dim
        ):

            xj, _ = (
                self.marginals[j]
                .forward(
                    u[:, j:j+1],
                    cond_batch,
                )
            )

            values.append(
                xj.clamp(
                    0.0,
                    1.0,
                )
            )

        return torch.cat(
            values,
            dim=1,
        )


# ============================================================
# Training
# ============================================================

def train_marginals(
    model,
    benchmark,
    device,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR_MARGINAL,
    )

    for step in range(
        MARGINAL_STEPS
    ):

        cond = sample_conditions(
            BATCH_SIZE,
            device,
        )

        x = sample_target(
            benchmark,
            cond,
            DIM,
        )

        loss = 0.0

        for j in range(
            DIM
        ):

            _, logdet = (
                model.marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            loss += (
                logdet.mean()
            )

        loss /= DIM

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"  marginals "
                f"{step+1}/{MARGINAL_STEPS} "
                f"loss={loss.item():.6f}"
            )


def train_bayesian_copula(
    model,
    benchmark,
    beta_final,
    device,
):
    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    for step in range(
        COPULA_STEPS
    ):

        cond = sample_conditions(
            BATCH_SIZE,
            device,
        )

        with torch.no_grad():

            x = sample_target(
                benchmark,
                cond,
                DIM,
            )

            u, _ = (
                model
                .x_to_u_and_log_marginal(
                    x,
                    cond,
                )
            )

        temperature = linear_schedule(
            step,
            COPULA_STEPS,
            TEMPERATURE_START,
            TEMPERATURE_END,
        )

        beta = beta_schedule(
            step,
            COPULA_STEPS,
            beta_final,
        )

        logits = (
            model.copula
            .gate_logits(
                cond
            )
        )

        probs = torch.sigmoid(
            logits
        )

        gates = (
            sample_binary_concrete(
                logits,
                temperature,
            )
        )

        logq = (
            model.copula
            .log_prob_u(
                u,
                cond,
                gates,
            )
        )

        nll = (
            -logq.mean()
        )

        kl = (
            bernoulli_kl(
                probs,
                PRIOR_EDGE_PROB,
            )
            .sum(dim=1)
            .mean()
        )

        loss = (
            nll
            + beta * kl
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copula.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        if (
            step + 1
        ) % 250 == 0:

            expected_edges = (
                probs.sum(
                    dim=1
                )
                .mean()
                .item()
            )

            print(
                f"  copula "
                f"{step+1}/{COPULA_STEPS} "
                f"nll={nll.item():.5f} "
                f"kl={kl.item():.4f} "
                f"Eedges={expected_edges:.2f} "
                f"beta={beta:.5f}"
            )


# ============================================================
# Evaluation
# ============================================================

def evaluate_model(
    model,
    benchmark,
    device,
):
    rows = []

    for condition_id, cond in enumerate(
        TEST_CONDITIONS
    ):

        cond = (
            cond.to(device)
            .reshape(
                1,
                -1,
            )
        )

        cond_eval = cond.expand(
            N_EVAL,
            -1,
        )

        with torch.no_grad():

            x = model.sample(
                N_EVAL,
                cond,
            )

            logq = model.log_prob(
                x,
                cond_eval,
            )

            logp = target_logpdf(
                benchmark,
                x,
                cond_eval,
            )

            weights = torch.exp(
                logp - logq
            )

            ess = (
                weights.sum() ** 2
                /
                (
                    N_EVAL
                    * weights.square().sum()
                )
            )

            estimate = (
                weights.mean()
            )

        rows.append(
            {
                "condition_id":
                    condition_id,

                "ess":
                    ess.item(),

                "estimate":
                    estimate.item(),

                "relative_error":
                    abs(
                        estimate.item()
                        - 1.0
                    ),
            }
        )

    return rows


# ============================================================
# Gate variation analysis
# ============================================================

def analyze_random_conditions(
    model,
    benchmark,
    model_name,
    beta,
    device,
):
    conditions = torch.rand(
        N_DIAGNOSTIC_CONDITIONS,
        COND_DIM,
        device=device,
    )

    with torch.no_grad():

        probs = (
            model.copula
            .gate_probabilities(
                conditions
            )
            .cpu()
            .numpy()
        )

    edge_pairs = (
        model.copula
        .gates
        .edge_pairs
    )

    rows = []

    for edge_id, (
        parent,
        child,
    ) in enumerate(
        edge_pairs
    ):

        values = (
            probs[:, edge_id]
        )

        rows.append(
            {
                "benchmark":
                    benchmark,

                "model":
                    model_name,

                "beta":
                    beta,

                "parent":
                    parent,

                "child":
                    child,

                "mean_probability":
                    float(
                        np.mean(
                            values
                        )
                    ),

                "std_probability":
                    float(
                        np.std(
                            values
                        )
                    ),

                "min_probability":
                    float(
                        np.min(
                            values
                        )
                    ),

                "max_probability":
                    float(
                        np.max(
                            values
                        )
                    ),

                "range_probability":
                    float(
                        np.max(
                            values
                        )
                        - np.min(
                            values
                        )
                    ),
            }
        )

    return pd.DataFrame(
        rows
    )


def condition_axis_sweep(
    model,
    benchmark,
    model_name,
    beta,
    device,
    output_dir,
):

    grid = torch.linspace(
        0.0,
        1.0,
        N_CONDITION_GRID,
        device=device,
    )

    edge_pairs = (
        model.copula
        .gates
        .edge_pairs
    )

    all_rows = []

    for axis in range(
        COND_DIM
    ):

        conditions = torch.full(
            (
                N_CONDITION_GRID,
                COND_DIM,
            ),
            0.5,
            device=device,
        )

        conditions[
            :,
            axis
        ] = grid

        with torch.no_grad():

            probs = (
                model.copula
                .gate_probabilities(
                    conditions
                )
                .cpu()
                .numpy()
            )

        grid_np = (
            grid.cpu()
            .numpy()
        )

        # Save every edge
        for edge_id, (
            parent,
            child,
        ) in enumerate(
            edge_pairs
        ):

            for k in range(
                N_CONDITION_GRID
            ):

                all_rows.append(
                    {
                        "benchmark":
                            benchmark,

                        "model":
                            model_name,

                        "beta":
                            beta,

                        "condition_axis":
                            axis,

                        "condition_value":
                            float(
                                grid_np[k]
                            ),

                        "parent":
                            parent,

                        "child":
                            child,

                        "probability":
                            float(
                                probs[
                                    k,
                                    edge_id,
                                ]
                            ),
                    }
                )

        # --------------------------------------------
        # Plot top 6 most variable edges along axis
        # --------------------------------------------

        ranges = (
            probs.max(
                axis=0
            )
            - probs.min(
                axis=0
            )
        )

        top_edge_ids = (
            np.argsort(
                ranges
            )[
                -6:
            ][
                ::-1
            ]
        )

        plt.figure(
            figsize=(8, 5)
        )

        for edge_id in (
            top_edge_ids
        ):

            parent, child = (
                edge_pairs[
                    edge_id
                ]
            )

            plt.plot(
                grid_np,
                probs[
                    :,
                    edge_id
                ],
                label=
                    f"{parent}->{child}",
            )

        plt.xlabel(
            f"condition c{axis}"
        )

        plt.ylabel(
            "Gate probability"
        )

        plt.ylim(
            0.0,
            1.0,
        )

        plt.title(
            f"{benchmark} | "
            f"{model_name} | "
            f"beta={beta} | "
            f"vary c{axis}"
        )

        plt.legend()

        plt.tight_layout()

        filename = (
            f"{benchmark}_"
            f"{model_name}_"
            f"beta{beta}_"
            f"c{axis}.png"
        )

        plt.savefig(
            output_dir
            / filename,
            dpi=180,
        )

        plt.close()

    return pd.DataFrame(
        all_rows
    )


# ============================================================
# Main
# ============================================================

def run(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}\n"
    )

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    plot_dir = (
        output_dir
        / "plots"
    )

    plot_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    gate_stats = []
    sweep_rows = []
    eval_rows = []

    for benchmark in (
        args.benchmarks
    ):

        beta = (
            args.curved_beta
            if benchmark
            == "curved_ridge"
            else args.sharp_beta
        )

        print()
        print(
            "=" * 80
        )

        print(
            f"{benchmark} "
            f"| beta={beta}"
        )

        print(
            "=" * 80
        )

        seed_everything(
            args.seed
        )

        # --------------------------------------------
        # Shared marginals
        # --------------------------------------------

        marginal_bank = (
            MarginalBank(
                DIM
            )
            .to(device)
        )

        train_marginals(
            marginal_bank,
            benchmark,
            device,
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

        # ============================================
        # FIXED vs CONDITIONAL
        # ============================================

        for conditional in [
            False,
            True,
        ]:

            model_name = (
                "bayes_conditional"
                if conditional
                else "bayes_fixed"
            )

            print()
            print(
                f"MODEL: {model_name}"
            )

            local_bank = (
                MarginalBank(
                    DIM
                )
                .to(device)
            )

            local_bank.load_state_dict(
                shared_state
            )

            for marginal in (
                local_bank.marginals
            ):

                for p in (
                    marginal.parameters()
                ):
                    p.requires_grad = False

            copula = BayesianGatedCopula(
                dim=DIM,
                conditional=conditional,
                prior_edge_prob=
                    PRIOR_EDGE_PROB,
            )

            model = (
                MarginalCopulaModel(
                    local_bank,
                    copula,
                    DIM,
                )
                .to(device)
            )

            seed_everything(
                args.seed
            )

            synchronize(
                device
            )

            start = (
                time.perf_counter()
            )

            train_bayesian_copula(
                model,
                benchmark,
                beta,
                device,
            )

            synchronize(
                device
            )

            train_time = (
                time.perf_counter()
                - start
            )

            # ----------------------------------------
            # Performance
            # ----------------------------------------

            model_eval = (
                evaluate_model(
                    model,
                    benchmark,
                    device,
                )
            )

            for row in (
                model_eval
            ):

                row.update(
                    {
                        "benchmark":
                            benchmark,

                        "model":
                            model_name,

                        "beta":
                            beta,

                        "train_time":
                            train_time,
                    }
                )

                eval_rows.append(
                    row
                )

            # ----------------------------------------
            # Random-c gate statistics
            # ----------------------------------------

            stats_df = (
                analyze_random_conditions(
                    model,
                    benchmark,
                    model_name,
                    beta,
                    device,
                )
            )

            gate_stats.append(
                stats_df
            )

            # ----------------------------------------
            # Axis sweeps
            # ----------------------------------------

            sweep_df = (
                condition_axis_sweep(
                    model,
                    benchmark,
                    model_name,
                    beta,
                    device,
                    plot_dir,
                )
            )

            sweep_rows.append(
                sweep_df
            )

            del model
            del local_bank

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del marginal_bank

    # ========================================================
    # Save
    # ========================================================

    gate_stats_df = pd.concat(
        gate_stats,
        ignore_index=True,
    )

    gate_stats_df.to_csv(
        output_dir
        / "gate_variability.csv",
        index=False,
    )

    sweep_df = pd.concat(
        sweep_rows,
        ignore_index=True,
    )

    sweep_df.to_csv(
        output_dir
        / "condition_sweeps.csv",
        index=False,
    )

    eval_df = pd.DataFrame(
        eval_rows
    )

    eval_df.to_csv(
        output_dir
        / "raw_results.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    performance_summary = (
        eval_df
        .groupby(
            [
                "benchmark",
                "model",
                "beta",
            ]
        )
        .agg(
            ess_mean=(
                "ess",
                "mean",
            ),

            relative_error_mean=(
                "relative_error",
                "mean",
            ),

            train_time_mean=(
                "train_time",
                "mean",
            ),
        )
        .reset_index()
    )

    variability_summary = (
        gate_stats_df
        .groupby(
            [
                "benchmark",
                "model",
                "beta",
            ]
        )
        .agg(
            mean_gate_probability=(
                "mean_probability",
                "mean",
            ),

            mean_gate_std=(
                "std_probability",
                "mean",
            ),

            max_gate_std=(
                "std_probability",
                "max",
            ),

            mean_gate_range=(
                "range_probability",
                "mean",
            ),

            max_gate_range=(
                "range_probability",
                "max",
            ),
        )
        .reset_index()
    )

    summary = (
        performance_summary
        .merge(
            variability_summary,
            on=[
                "benchmark",
                "model",
                "beta",
            ],
            how="left",
        )
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
        "CONDITIONAL GATE ANALYSIS COMPLETE"
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
        default=[
            "curved_ridge",
            "sharp_broad",
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--curved-beta",
        type=float,
        default=
            DEFAULT_BETAS[
                "curved_ridge"
            ],
    )

    parser.add_argument(
        "--sharp-beta",
        type=float,
        default=
            DEFAULT_BETAS[
                "sharp_broad"
            ],
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_DIR
        ),
    )

    return (
        parser.parse_args()
    )


if __name__ == "__main__":

    run(
        parse_args()
    )