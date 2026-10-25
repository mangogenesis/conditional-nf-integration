import argparse
import json
import math
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
    "curved_ridge",
    "sharp_broad",
]

DEFAULT_DIMS = [8]
DEFAULT_SEEDS = [0]

COND_DIM = 3

N_BINS = 32
HIDDEN = 128
GATE_HIDDEN = 64

MARGINAL_STEPS = 1200
COPULA_STEPS = 2500

BATCH_SIZE = 4096

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4

GRAD_CLIP = 10.0

# ------------------------------------------------------------
# Bayesian gate settings
# ------------------------------------------------------------

PRIOR_EDGE_PROB = 0.20

DEFAULT_BETAS = [
    0.001,
    0.01,
    0.05,
    0.10,
]

TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.20

# keep beta near zero early, then turn on sparsity
BETA_WARMUP_FRAC = 0.20

# hard threshold only for diagnostic hard-graph evaluation
HARD_GATE_THRESHOLD = 0.50

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

N_EVAL = 100_000
N_KL = 50_000

OUTPUT_DIR = Path(
    "results/bayesian_gate_beta_sweep"
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
    warmup_frac,
):
    """
    First warmup_frac of training:
        beta = 0

    Remaining training:
        linearly ramp to beta_final
    """

    warmup_steps = int(
        warmup_frac * total_steps
    )

    if step < warmup_steps:
        return 0.0

    remaining = max(
        total_steps - warmup_steps,
        1,
    )

    local_step = (
        step - warmup_steps
    )

    frac = min(
        max(
            local_step / remaining,
            0.0,
        ),
        1.0,
    )

    return (
        beta_final * frac
    )


# ============================================================
# Marginal bank
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
# Deterministic parent structures
# ============================================================

def build_markov1_parents(dim):

    parents = {
        0: []
    }

    for j in range(
        1,
        dim,
    ):
        parents[j] = [
            j - 1
        ]

    return parents


def build_full_ar_parents(dim):

    return {
        j: list(range(j))
        for j in range(dim)
    }


# ============================================================
# Deterministic copula
# ============================================================

class StructuredCopula(nn.Module):

    def __init__(
        self,
        dim,
        parents,
    ):
        super().__init__()

        self.dim = dim
        self.parents = parents

        self.conditionals = nn.ModuleDict()

        for j in range(
            1,
            dim,
        ):

            context_dim = (
                COND_DIM
                + len(
                    parents[j]
                )
            )

            self.conditionals[
                str(j)
            ] = ConditionalCubicFlow1D(
                cond_dim=context_dim,
                hidden=HIDDEN,
                n_bins=N_BINS,
            )

    def _context(
        self,
        u,
        cond,
        j,
    ):
        parents = self.parents[j]

        if len(parents) == 0:
            return cond

        return torch.cat(
            [
                u[:, parents],
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
# Conditional gate network
# ============================================================

class ConditionalGateNetwork(nn.Module):

    def __init__(
        self,
        dim,
        prior_edge_prob,
    ):
        super().__init__()

        self.dim = dim

        self.n_edges = (
            dim
            * (dim - 1)
            // 2
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

        # ----------------------------------------------------
        # IMPORTANT FIX:
        # initialize every condition at the Bernoulli prior
        #
        # sigmoid(logit(pi0)) = pi0
        # ----------------------------------------------------

        prior_logit = math.log(
            prior_edge_prob
            /
            (
                1.0
                - prior_edge_prob
            )
        )

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

    def forward(
        self,
        cond,
    ):
        return self.net(
            cond
        )


# ============================================================
# Binary Concrete
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

    relaxed = torch.sigmoid(
        (
            logits
            + logistic_noise
        )
        / temperature
    )

    return relaxed


def bernoulli_kl(
    probs,
    prior_prob,
):
    eps = 1e-7

    probs = probs.clamp(
        eps,
        1.0 - eps,
    )

    p0 = torch.as_tensor(
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
                p0
            )
        )
        +
        (
            1.0
            - probs
        )
        * (
            torch.log(
                1.0 - probs
            )
            - torch.log(
                1.0 - p0
            )
        )
    )


# ============================================================
# Conditional Bayesian gated copula
# ============================================================

class BayesianGatedCopula(nn.Module):

    def __init__(
        self,
        dim,
        prior_edge_prob,
    ):
        super().__init__()

        self.dim = dim

        self.gates = (
            ConditionalGateNetwork(
                dim=dim,
                prior_edge_prob=
                    prior_edge_prob,
            )
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

            # For each possible parent:
            #
            # 1. gated centered coordinate
            # 2. gate value itself
            #
            # plus c
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
        edge_indices = (
            self.edge_index[j]
        )

        gate_j = gates[
            :,
            edge_indices
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

            context = self._context(
                u,
                cond,
                gates,
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

            logq = (
                logq
                - forward_logdet.squeeze(-1)
            )

        return logq

    def sample_u_with_gates(
        self,
        n,
        cond,
        gates,
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
                gates,
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
# Full marginal + copula model
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
        u_list = []

        log_marginal = torch.zeros(
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

            log_marginal = (
                log_marginal
                - forward_logdet.squeeze(-1)
            )

        return (
            torch.cat(
                u_list,
                dim=1,
            ),
            log_marginal,
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

        loss = (
            loss / dim
        )

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
# Deterministic copula training
# ============================================================

def train_structured_copula(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
):
    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

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

        logq = (
            model.copula
            .log_prob_u(
                u,
                cond,
            )
        )

        loss = (
            -logq.mean()
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
# Bayesian gate training
# ============================================================

def train_bayesian_copula(
    model,
    benchmark,
    dim,
    device,
    steps,
    batch_size,
    prior_edge_prob,
    beta_final,
):
    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    recent_loss = []
    recent_nll = []
    recent_kl = []
    recent_edges = []
    recent_hard = []

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

        temperature = (
            linear_schedule(
                step,
                steps,
                TEMPERATURE_START,
                TEMPERATURE_END,
            )
        )

        beta = beta_schedule(
            step,
            steps,
            beta_final,
            BETA_WARMUP_FRAC,
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

        relaxed_gates = (
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
                relaxed_gates,
            )
        )

        nll = (
            -logq.mean()
        )

        kl_matrix = (
            bernoulli_kl(
                probs,
                prior_edge_prob,
            )
        )

        kl = (
            kl_matrix
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

        expected_edges = (
            probs
            .sum(dim=1)
            .mean()
        )

        hard_edges = (
            (
                probs
                > HARD_GATE_THRESHOLD
            )
            .float()
            .sum(dim=1)
            .mean()
        )

        recent_loss.append(
            loss.item()
        )

        recent_nll.append(
            nll.item()
        )

        recent_kl.append(
            kl.item()
        )

        recent_edges.append(
            expected_edges.item()
        )

        recent_hard.append(
            hard_edges.item()
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"      bayes "
                f"{step+1:5d}/{steps} "
                f"loss="
                f"{np.mean(recent_loss[-100:]):.5f} "
                f"nll="
                f"{np.mean(recent_nll[-100:]):.5f} "
                f"kl="
                f"{np.mean(recent_kl[-100:]):.4f} "
                f"E[edges]="
                f"{np.mean(recent_edges[-100:]):.2f} "
                f"hard="
                f"{np.mean(recent_hard[-100:]):.2f} "
                f"T={temperature:.3f} "
                f"beta={beta:.5f}"
            )


# ============================================================
# Gate helpers
# ============================================================

def get_gate_values(
    copula,
    cond,
    mode,
):
    probs = (
        copula
        .gate_probabilities(
            cond
        )
    )

    if mode == "soft":

        return probs

    if mode == "hard":

        return (
            probs
            > HARD_GATE_THRESHOLD
        ).float()

    raise ValueError(
        mode
    )


def gate_diagnostics(
    model,
    conditions,
):
    with torch.no_grad():

        probs = (
            model.copula
            .gate_probabilities(
                conditions
            )
        )

    return {
        "probs":
            probs.detach()
            .cpu()
            .numpy(),

        "mean_probs":
            probs.mean(dim=0)
            .detach()
            .cpu()
            .numpy(),

        "std_probs":
            probs.std(dim=0)
            .detach()
            .cpu()
            .numpy(),

        "expected_edges":
            probs.sum(dim=1)
            .mean()
            .item(),

        "hard_edges":
            (
                probs
                > HARD_GATE_THRESHOLD
            )
            .float()
            .sum(dim=1)
            .mean()
            .item(),
    }


# ============================================================
# Deterministic model evaluation
# ============================================================

def deterministic_log_prob(
    model,
    x,
    cond,
):
    u, log_marginal = (
        model
        .x_to_u_and_log_marginal(
            x,
            cond,
        )
    )

    log_copula = (
        model.copula
        .log_prob_u(
            u,
            cond,
        )
    )

    return (
        log_marginal
        + log_copula
    )


def deterministic_sample(
    model,
    n,
    cond,
):
    cond_batch = cond.expand(
        n,
        -1,
    )

    u = (
        model.copula
        .sample_u(
            n,
            cond_batch,
        )
    )

    x_list = []

    for j in range(
        model.dim
    ):

        xj, _ = (
            model.marginals[j]
            .forward(
                u[:, j:j+1],
                cond_batch,
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
# Bayesian evaluation
# ============================================================

def bayesian_log_prob(
    model,
    x,
    cond,
    gate_mode,
):
    u, log_marginal = (
        model
        .x_to_u_and_log_marginal(
            x,
            cond,
        )
    )

    gates = get_gate_values(
        model.copula,
        cond,
        gate_mode,
    )

    log_copula = (
        model.copula
        .log_prob_u(
            u,
            cond,
            gates,
        )
    )

    return (
        log_marginal
        + log_copula
    )


def bayesian_sample(
    model,
    n,
    cond,
    gate_mode,
):
    cond_batch = cond.expand(
        n,
        -1,
    )

    gates = get_gate_values(
        model.copula,
        cond_batch,
        gate_mode,
    )

    u = (
        model.copula
        .sample_u_with_gates(
            n,
            cond_batch,
            gates,
        )
    )

    x_list = []

    for j in range(
        model.dim
    ):

        xj, _ = (
            model.marginals[j]
            .forward(
                u[:, j:j+1],
                cond_batch,
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
# Forward KL
# ============================================================

def estimate_forward_kl(
    model,
    benchmark,
    cond,
    dim,
    n,
    model_type,
    gate_mode=None,
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

        if model_type == "deterministic":

            logq = deterministic_log_prob(
                model,
                x,
                cond_batch,
            )

        else:

            logq = bayesian_log_prob(
                model,
                x,
                cond_batch,
                gate_mode,
            )

        values = (
            logp - logq
        )

    return (
        values.mean().item()
    )


# ============================================================
# Importance sampling
# ============================================================

def evaluate_is(
    model,
    benchmark,
    cond,
    dim,
    n,
    model_type,
    gate_mode=None,
):
    cond_batch = cond.expand(
        n,
        -1,
    )

    with torch.no_grad():

        if model_type == "deterministic":

            x = deterministic_sample(
                model,
                n,
                cond,
            )

            logq = deterministic_log_prob(
                model,
                x,
                cond_batch,
            )

        else:

            x = bayesian_sample(
                model,
                n,
                cond,
                gate_mode,
            )

            logq = bayesian_log_prob(
                model,
                x,
                cond_batch,
                gate_mode,
            )

        logp = target_logpdf(
            benchmark,
            x,
            cond_batch,
        )

        logw = (
            logp - logq
        )

        weights = torch.exp(
            logw
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

        ess_frac = (
            weights.sum() ** 2
            /
            (
                weights.square().sum()
                * n
            )
        )

        relative_error = (
            torch.abs(
                estimate
                - 1.0
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
            ess_frac.item(),
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

    gate_dir = (
        output_dir
        / "gates"
    )

    gate_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []
    gate_rows = []

    for benchmark in (
        args.benchmarks
    ):

        for dim in (
            args.dims
        ):

            for seed in (
                args.seeds
            ):

                print()
                print(
                    "=" * 80
                )

                print(
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

                # ------------------------------------------------
                # Shared marginals
                # ------------------------------------------------

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
                    key:
                    value.detach()
                    .cpu()
                    .clone()

                    for key, value
                    in marginal_bank
                    .state_dict()
                    .items()
                }

                # ------------------------------------------------
                # Baselines
                # ------------------------------------------------

                baseline_defs = {
                    "markov1":
                        build_markov1_parents(
                            dim
                        ),

                    "full_ar":
                        build_full_ar_parents(
                            dim
                        ),
                }

                for (
                    model_name,
                    parents,
                ) in (
                    baseline_defs.items()
                ):

                    print()
                    print(
                        f"MODEL: {model_name}"
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

                    for marginal in (
                        local_bank.marginals
                    ):
                        for p in (
                            marginal.parameters()
                        ):
                            p.requires_grad = False

                    copula = StructuredCopula(
                        dim=dim,
                        parents=parents,
                    )

                    model = (
                        MarginalCopulaModel(
                            local_bank,
                            copula,
                            dim,
                        )
                        .to(device)
                    )

                    seed_everything(
                        seed
                    )

                    synchronize(device)

                    t0 = (
                        time.perf_counter()
                    )

                    train_structured_copula(
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

                    for (
                        condition_id,
                        cond,
                    ) in enumerate(
                        TEST_CONDITIONS
                    ):

                        cond = (
                            cond.to(device)
                            .reshape(
                                1,
                                -1,
                            )
                        )

                        kl = (
                            estimate_forward_kl(
                                model,
                                benchmark,
                                cond,
                                dim,
                                args.n_kl,
                                model_type=
                                    "deterministic",
                            )
                        )

                        result = evaluate_is(
                            model,
                            benchmark,
                            cond,
                            dim,
                            args.n_eval,
                            model_type=
                                "deterministic",
                        )

                        rows.append(
                            {
                                "benchmark":
                                    benchmark,

                                "model":
                                    model_name,

                                "dimension":
                                    dim,

                                "seed":
                                    seed,

                                "beta":
                                    np.nan,

                                "gate_mode":
                                    "none",

                                "condition_id":
                                    condition_id,

                                "ess":
                                    result["ess"],

                                "forward_kl":
                                    kl,

                                "variance":
                                    result[
                                        "variance"
                                    ],

                                "relative_error":
                                    result[
                                        "relative_error"
                                    ],

                                "expected_edges":
                                    np.nan,

                                "hard_edges":
                                    np.nan,

                                "parameter_count":
                                    count_parameters(
                                        model
                                    ),

                                "marginal_train_time":
                                    marginal_time,

                                "copula_train_time":
                                    copula_time,

                                "total_train_time":
                                    (
                                        marginal_time
                                        + copula_time
                                    ),
                            }
                        )

                    del model
                    del local_bank

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # ------------------------------------------------
                # Bayesian beta sweep
                # ------------------------------------------------

                for beta_final in (
                    args.betas
                ):

                    print()
                    print(
                        f"MODEL: bayes_conditional "
                        f"| beta={beta_final}"
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

                    for marginal in (
                        local_bank.marginals
                    ):
                        for p in (
                            marginal.parameters()
                        ):
                            p.requires_grad = False

                    copula = BayesianGatedCopula(
                        dim=dim,
                        prior_edge_prob=
                            args.prior_edge_prob,
                    )

                    model = (
                        MarginalCopulaModel(
                            local_bank,
                            copula,
                            dim,
                        )
                        .to(device)
                    )

                    seed_everything(
                        seed
                    )

                    synchronize(device)

                    t0 = (
                        time.perf_counter()
                    )

                    train_bayesian_copula(
                        model,
                        benchmark,
                        dim,
                        device,
                        args.copula_steps,
                        args.batch_size,
                        args.prior_edge_prob,
                        beta_final,
                    )

                    synchronize(device)

                    copula_time = (
                        time.perf_counter()
                        - t0
                    )

                    # --------------------------------------------
                    # Gate diagnostics across held-out c
                    # --------------------------------------------

                    test_conditions_device = (
                        TEST_CONDITIONS
                        .to(device)
                    )

                    diag = gate_diagnostics(
                        model,
                        test_conditions_device,
                    )

                    expected_edges = (
                        diag[
                            "expected_edges"
                        ]
                    )

                    hard_edges = (
                        diag[
                            "hard_edges"
                        ]
                    )

                    beta_tag = (
                        str(beta_final)
                        .replace(
                            ".",
                            "p",
                        )
                    )

                    np.save(
                        gate_dir
                        /
                        (
                            f"{benchmark}"
                            f"_d{dim}"
                            f"_seed{seed}"
                            f"_beta{beta_tag}"
                            f"_probs.npy"
                        ),
                        diag["probs"],
                    )

                    edge_pairs = (
                        model.copula
                        .gates
                        .edge_pairs
                    )

                    for edge_id, (
                        parent,
                        child,
                    ) in enumerate(
                        edge_pairs
                    ):

                        gate_rows.append(
                            {
                                "benchmark":
                                    benchmark,

                                "dimension":
                                    dim,

                                "seed":
                                    seed,

                                "beta":
                                    beta_final,

                                "parent":
                                    parent,

                                "child":
                                    child,

                                "mean_probability":
                                    float(
                                        diag[
                                            "mean_probs"
                                        ][edge_id]
                                    ),

                                "std_probability":
                                    float(
                                        diag[
                                            "std_probs"
                                        ][edge_id]
                                    ),
                            }
                        )

                    # --------------------------------------------
                    # Evaluate both SOFT and HARD gates
                    # --------------------------------------------

                    for gate_mode in [
                        "soft",
                        "hard",
                    ]:

                        print(
                            f"  evaluation: "
                            f"{gate_mode}"
                        )

                        for (
                            condition_id,
                            cond,
                        ) in enumerate(
                            TEST_CONDITIONS
                        ):

                            cond = (
                                cond.to(device)
                                .reshape(
                                    1,
                                    -1,
                                )
                            )

                            kl = (
                                estimate_forward_kl(
                                    model,
                                    benchmark,
                                    cond,
                                    dim,
                                    args.n_kl,
                                    model_type=
                                        "bayesian",
                                    gate_mode=
                                        gate_mode,
                                )
                            )

                            result = (
                                evaluate_is(
                                    model,
                                    benchmark,
                                    cond,
                                    dim,
                                    args.n_eval,
                                    model_type=
                                        "bayesian",
                                    gate_mode=
                                        gate_mode,
                                )
                            )

                            rows.append(
                                {
                                    "benchmark":
                                        benchmark,

                                    "model":
                                        "bayes_conditional",

                                    "dimension":
                                        dim,

                                    "seed":
                                        seed,

                                    "beta":
                                        beta_final,

                                    "gate_mode":
                                        gate_mode,

                                    "condition_id":
                                        condition_id,

                                    "ess":
                                        result[
                                            "ess"
                                        ],

                                    "forward_kl":
                                        kl,

                                    "variance":
                                        result[
                                            "variance"
                                        ],

                                    "relative_error":
                                        result[
                                            "relative_error"
                                        ],

                                    "expected_edges":
                                        expected_edges,

                                    "hard_edges":
                                        hard_edges,

                                    "parameter_count":
                                        count_parameters(
                                            model
                                        ),

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
                            )

                    pd.DataFrame(
                        rows
                    ).to_csv(
                        output_dir
                        / "raw_results.csv",
                        index=False,
                    )

                    pd.DataFrame(
                        gate_rows
                    ).to_csv(
                        output_dir
                        / "gate_summary.csv",
                        index=False,
                    )

                    del model
                    del local_bank

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                del marginal_bank

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================

    df = pd.DataFrame(
        rows
    )

    summary = (
        df
        .groupby(
            [
                "benchmark",
                "model",
                "dimension",
                "beta",
                "gate_mode",
            ],
            dropna=False,
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

            expected_edges_mean=(
                "expected_edges",
                "mean",
            ),

            hard_edges_mean=(
                "hard_edges",
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
        "=" * 90
    )

    print(
        "BAYESIAN GATE BETA SWEEP COMPLETE"
    )

    print(
        "=" * 90
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
        "--betas",
        nargs="+",
        type=float,
        default=
            DEFAULT_BETAS,
    )

    parser.add_argument(
        "--prior-edge-prob",
        type=float,
        default=
            PRIOR_EDGE_PROB,
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

        args.betas = [
            0.001,
            0.01,
            0.05,
        ]

        args.marginal_steps = 100
        args.copula_steps = 250

        args.batch_size = 1024

        args.n_eval = 10_000
        args.n_kl = 5_000

        args.output = (
            "results/"
            "bayesian_gate_beta_sweep_smoke"
        )

    return args


if __name__ == "__main__":

    args = parse_args()
    run(args)