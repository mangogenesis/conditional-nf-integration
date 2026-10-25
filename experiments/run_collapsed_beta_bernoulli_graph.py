import argparse
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

TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.20

REG_WARMUP_FRAC = 0.20

# ------------------------------------------------------------
# Collapsed Beta-Bernoulli priors
#
# Beta(1,1):
#   neutral over global graph density.
#
# Beta(1,3):
#   mild preference toward lower graph density.
# ------------------------------------------------------------

DEFAULT_PRIORS = [
    (1.0, 1.0),
    (1.0, 3.0),
]

# Regularizer strength.
DEFAULT_LAMBDAS = [
    0.001,
    0.01,
    0.05,
]

N_EVAL = 100_000
N_KL = 50_000

HARD_THRESHOLD = 0.50

OUTPUT_DIR = Path(
    "results/collapsed_beta_bernoulli_graph"
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


def regularization_schedule(
    step,
    total_steps,
    final_lambda,
):
    warmup = int(
        REG_WARMUP_FRAC
        * total_steps
    )

    if step < warmup:
        return 0.0

    remaining = max(
        total_steps - warmup,
        1,
    )

    frac = (
        step - warmup
    ) / remaining

    frac = min(
        max(frac, 0.0),
        1.0,
    )

    return (
        final_lambda * frac
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


# ============================================================
# Deterministic baseline graphs
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


class StructuredCopula(nn.Module):

    def __init__(
        self,
        dim,
        parents,
    ):
        super().__init__()

        self.dim = dim
        self.parents = parents

        self.conditionals = (
            nn.ModuleDict()
        )

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

        parents = (
            self.parents[j]
        )

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
# Global gate posterior
#
# Same constant-MLP parameterization that already worked
# much better than direct scalar logits.
# ============================================================

class GlobalEdgeGateMLP(nn.Module):

    def __init__(
        self,
        dim,
        initial_prob=0.5,
    ):
        super().__init__()

        self.dim = dim

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

        self.net = nn.Sequential(
            nn.Linear(
                1,
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

        initial_logit = math.log(
            initial_prob
            /
            (
                1.0
                - initial_prob
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
            initial_logit,
        )

    def logits(
        self,
        batch_size,
        device,
    ):

        dummy = torch.zeros(
            batch_size,
            1,
            device=device,
        )

        return self.net(
            dummy
        )

    def probabilities(
        self,
        batch_size,
        device,
    ):

        return torch.sigmoid(
            self.logits(
                batch_size,
                device,
            )
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

    return torch.sigmoid(
        (
            logits
            + logistic_noise
        )
        / temperature
    )


# ============================================================
# Collapsed Beta-Bernoulli graph regularizer
# ============================================================

def log_beta_fn(
    a,
    b,
):
    return (
        torch.lgamma(a)
        + torch.lgamma(b)
        - torch.lgamma(a + b)
    )


def collapsed_graph_nll(
    gates,
    prior_alpha,
    prior_beta,
):
    """
    Relaxed collapsed Beta-Bernoulli graph penalty.

    Exact discrete model:

        rho ~ Beta(a,b)
        Z_e | rho ~ Bernoulli(rho)

    Integrating out rho:

        p(Z)
        = B(a + K, b + M - K)
          / B(a,b)

    where
        K = sum_e Z_e
        M = number of possible edges.

    Here gates are relaxed values in [0,1], so we use

        K_soft = sum_e gates_e

    as a differentiable continuous extension.

    Returns mean negative log prior over batch.
    """

    batch_size = (
        gates.shape[0]
    )

    m_edges = (
        gates.shape[1]
    )

    k_soft = gates.sum(
        dim=1
    )

    dtype = gates.dtype
    device = gates.device

    a = torch.tensor(
        prior_alpha,
        dtype=dtype,
        device=device,
    )

    b = torch.tensor(
        prior_beta,
        dtype=dtype,
        device=device,
    )

    m = torch.tensor(
        float(m_edges),
        dtype=dtype,
        device=device,
    )

    log_prior_graph = (
        log_beta_fn(
            a + k_soft,
            b + m - k_soft,
        )
        -
        log_beta_fn(
            a,
            b,
        )
    )

    return (
        -log_prior_graph.mean()
    )


# ============================================================
# Collapsed gated copula
# ============================================================

class CollapsedBayesianCopula(nn.Module):

    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.dim = dim

        self.gates = (
            GlobalEdgeGateMLP(
                dim=dim,
                initial_prob=0.5,
            )
        )

        self.edge_index = {}

        self.conditionals = (
            nn.ModuleDict()
        )

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

            # gated centered previous coordinates
            # + gate values
            # + physical condition c

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
        batch_size,
        device,
    ):

        return (
            self.gates
            .logits(
                batch_size,
                device,
            )
        )

    def gate_probabilities(
        self,
        batch_size,
        device,
    ):

        return (
            self.gates
            .probabilities(
                batch_size,
                device,
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

        previous = (
            u[:, :j]
            - 0.5
        )

        gated_previous = (
            gate_j
            * previous
        )

        return torch.cat(
            [
                gated_previous,
                gate_j,
                cond,
            ],
            dim=1,
        )

    def log_prob_u_with_gates(
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

    def log_prob_u_soft(
        self,
        u,
        cond,
    ):

        probs = (
            self.gate_probabilities(
                u.shape[0],
                u.device,
            )
        )

        return (
            self.log_prob_u_with_gates(
                u,
                cond,
                probs,
            )
        )

    def sample_u_soft(
        self,
        n,
        cond,
    ):

        if cond.shape[0] == 1:

            cond = cond.expand(
                n,
                -1,
            )

        probs = (
            self.gate_probabilities(
                n,
                cond.device,
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

            context = self._context(
                u,
                cond,
                probs,
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
# Full model
# ============================================================

class MarginalCopulaModel(nn.Module):

    def __init__(
        self,
        marginal_bank,
        copula,
        dim,
        gated=False,
    ):
        super().__init__()

        self.dim = dim

        self.marginals = (
            marginal_bank.marginals
        )

        self.copula = copula
        self.gated = gated

    def x_to_u_and_log_marginal(
        self,
        x,
        cond,
    ):

        values = []

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

            values.append(
                uj
            )

            log_marginal = (
                log_marginal
                - forward_logdet.squeeze(-1)
            )

        return (
            torch.cat(
                values,
                dim=1,
            ),
            log_marginal,
        )

    def log_prob(
        self,
        x,
        cond,
    ):

        u, log_marginal = (
            self.x_to_u_and_log_marginal(
                x,
                cond,
            )
        )

        if self.gated:

            log_copula = (
                self.copula
                .log_prob_u_soft(
                    u,
                    cond,
                )
            )

        else:

            log_copula = (
                self.copula
                .log_prob_u(
                    u,
                    cond,
                )
            )

        return (
            log_marginal
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

        if self.gated:

            u = (
                self.copula
                .sample_u_soft(
                    n,
                    cond_batch,
                )
            )

        else:

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
# Marginal training
# ============================================================

def train_marginals(
    marginal_bank,
    benchmark,
    device,
):

    optimizer = torch.optim.Adam(
        marginal_bank.parameters(),
        lr=LR_MARGINAL,
    )

    recent = []

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
                marginal_bank
                .marginals[j]
                .inverse(
                    x[:, j:j+1],
                    cond,
                )
            )

            loss += (
                logdet.mean()
            )

        loss = (
            loss / DIM
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
        ) % 300 == 0:

            print(
                f"  marginal "
                f"{step+1}/{MARGINAL_STEPS} "
                f"loss="
                f"{np.mean(recent[-100:]):.5f}"
            )


# ============================================================
# Baseline training
# ============================================================

def train_structured(
    model,
    benchmark,
    device,
):

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    recent = []

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
        ) % 500 == 0:

            print(
                f"  copula "
                f"{step+1}/{COPULA_STEPS} "
                f"NLL="
                f"{np.mean(recent[-100:]):.5f}"
            )


# ============================================================
# Collapsed graph training
# ============================================================

def train_collapsed(
    model,
    benchmark,
    prior_alpha,
    prior_beta,
    lambda_final,
    device,
):

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    recent_nll = []
    recent_prior = []
    recent_edges = []
    recent_hard = []

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

        lambda_now = regularization_schedule(
            step,
            COPULA_STEPS,
            lambda_final,
        )

        logits = (
            model.copula
            .gate_logits(
                BATCH_SIZE,
                device,
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

        # ----------------------------------------------------
        # Data fit
        # ----------------------------------------------------

        logq = (
            model.copula
            .log_prob_u_with_gates(
                u,
                cond,
                relaxed_gates,
            )
        )

        nll = (
            -logq.mean()
        )

        # ----------------------------------------------------
        # Collapsed graph prior
        #
        # Use posterior probabilities for a lower-noise
        # structural signal rather than the noisy sampled gate.
        # ----------------------------------------------------

        graph_nll = (
            collapsed_graph_nll(
                probs,
                prior_alpha,
                prior_beta,
            )
        )

        # Normalize by number of edges so lambda scale is
        # reasonably dimension-independent.
        graph_nll_per_edge = (
            graph_nll
            /
            model.copula
            .gates
            .n_edges
        )

        loss = (
            nll
            + lambda_now
            * graph_nll_per_edge
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copula.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        expected_edges = (
            probs[0]
            .sum()
            .item()
        )

        hard_edges = (
            (
                probs[0]
                >= HARD_THRESHOLD
            )
            .float()
            .sum()
            .item()
        )

        recent_nll.append(
            nll.item()
        )

        recent_prior.append(
            graph_nll_per_edge.item()
        )

        recent_edges.append(
            expected_edges
        )

        recent_hard.append(
            hard_edges
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"  collapsed "
                f"{step+1:4d}/{COPULA_STEPS} "
                f"NLL="
                f"{np.mean(recent_nll[-100:]):.5f} "
                f"graphNLL/edge="
                f"{np.mean(recent_prior[-100:]):.4f} "
                f"Eedges="
                f"{np.mean(recent_edges[-100:]):.2f} "
                f"hard="
                f"{np.mean(recent_hard[-100:]):.2f} "
                f"T={temperature:.3f} "
                f"lambda={lambda_now:.5f}"
            )


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    benchmark,
    device,
):

    rows = []

    for condition_id, cond in enumerate(
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
                / N_EVAL
            )

            ess = (
                weights.sum() ** 2
                /
                (
                    N_EVAL
                    * weights.square().sum()
                )
            )

        # ----------------------------------------------------
        # Forward KL
        # ----------------------------------------------------

        cond_kl = cond.expand(
            N_KL,
            -1,
        )

        with torch.no_grad():

            x_target = sample_target(
                benchmark,
                cond_kl,
                DIM,
            )

            logp_target = target_logpdf(
                benchmark,
                x_target,
                cond_kl,
            )

            logq_target = model.log_prob(
                x_target,
                cond_kl,
            )

            forward_kl = (
                logp_target
                - logq_target
            ).mean()

        rows.append(
            {
                "condition_id":
                    condition_id,

                "ess":
                    ess.item(),

                "forward_kl":
                    forward_kl.item(),

                "variance":
                    variance.item(),

                "relative_error":
                    abs(
                        estimate.item()
                        - 1.0
                    ),
            }
        )

    return rows


# ============================================================
# Graph diagnostics
# ============================================================

def graph_diagnostics(
    model,
    benchmark,
    prior_alpha,
    prior_beta,
    lambda_final,
    device,
):

    with torch.no_grad():

        probs = (
            model.copula
            .gate_probabilities(
                1,
                device,
            )[0]
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

        probability = float(
            probs[edge_id]
        )

        rows.append(
            {
                "benchmark":
                    benchmark,

                "prior_alpha":
                    prior_alpha,

                "prior_beta":
                    prior_beta,

                "lambda":
                    lambda_final,

                "parent":
                    parent,

                "child":
                    child,

                "probability":
                    probability,

                "selected_0p5":
                    int(
                        probability
                        >= HARD_THRESHOLD
                    ),
            }
        )

    summary = {
        "expected_edges":
            float(
                probs.sum()
            ),

        "hard_edges":
            int(
                (
                    probs
                    >= HARD_THRESHOLD
                ).sum()
            ),

        "mean_probability":
            float(
                probs.mean()
            ),

        "min_probability":
            float(
                probs.min()
            ),

        "max_probability":
            float(
                probs.max()
            ),

        "prob_std_across_edges":
            float(
                probs.std()
            ),
    }

    return (
        rows,
        summary,
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

    all_rows = []
    edge_rows = []

    for benchmark in (
        args.benchmarks
    ):

        print()
        print(
            "=" * 80
        )

        print(
            f"{benchmark} | d={DIM}"
        )

        print(
            "=" * 80
        )

        seed_everything(
            args.seed
        )

        # ====================================================
        # Shared marginals
        # ====================================================

        print(
            "\nTraining shared marginals..."
        )

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

        marginal_state = {
            key:
            value.detach()
            .cpu()
            .clone()

            for key, value
            in marginal_bank
            .state_dict()
            .items()
        }

        def make_bank():

            bank = (
                MarginalBank(
                    DIM
                )
                .to(device)
            )

            bank.load_state_dict(
                marginal_state
            )

            for marginal in (
                bank.marginals
            ):

                for p in (
                    marginal.parameters()
                ):

                    p.requires_grad = False

            return bank

        # ====================================================
        # Markov-1 baseline
        # ====================================================

        print(
            "\nMODEL: markov1"
        )

        markov_model = (
            MarginalCopulaModel(
                make_bank(),
                StructuredCopula(
                    DIM,
                    build_markov1_parents(
                        DIM
                    ),
                ),
                DIM,
                gated=False,
            )
            .to(device)
        )

        train_structured(
            markov_model,
            benchmark,
            device,
        )

        results = evaluate(
            markov_model,
            benchmark,
            device,
        )

        for row in results:

            row.update(
                {
                    "benchmark":
                        benchmark,

                    "model":
                        "markov1",

                    "prior_alpha":
                        np.nan,

                    "prior_beta":
                        np.nan,

                    "lambda":
                        np.nan,

                    "expected_edges":
                        7.0,

                    "hard_edges":
                        7.0,

                    "mean_probability":
                        np.nan,

                    "min_probability":
                        np.nan,

                    "max_probability":
                        np.nan,

                    "prob_std_across_edges":
                        np.nan,

                    "parameter_count":
                        count_parameters(
                            markov_model
                        ),
                }
            )

            all_rows.append(
                row
            )

        del markov_model

        # ====================================================
        # Full AR baseline
        # ====================================================

        print(
            "\nMODEL: full_ar"
        )

        full_model = (
            MarginalCopulaModel(
                make_bank(),
                StructuredCopula(
                    DIM,
                    build_full_ar_parents(
                        DIM
                    ),
                ),
                DIM,
                gated=False,
            )
            .to(device)
        )

        train_structured(
            full_model,
            benchmark,
            device,
        )

        results = evaluate(
            full_model,
            benchmark,
            device,
        )

        for row in results:

            row.update(
                {
                    "benchmark":
                        benchmark,

                    "model":
                        "full_ar",

                    "prior_alpha":
                        np.nan,

                    "prior_beta":
                        np.nan,

                    "lambda":
                        np.nan,

                    "expected_edges":
                        28.0,

                    "hard_edges":
                        28.0,

                    "mean_probability":
                        np.nan,

                    "min_probability":
                        np.nan,

                    "max_probability":
                        np.nan,

                    "prob_std_across_edges":
                        np.nan,

                    "parameter_count":
                        count_parameters(
                            full_model
                        ),
                }
            )

            all_rows.append(
                row
            )

        del full_model

        # ====================================================
        # Collapsed graph models
        # ====================================================

        for (
            prior_alpha,
            prior_beta,
        ) in args.priors:

            for lambda_final in (
                args.lambdas
            ):

                print()
                print(
                    "MODEL: collapsed "
                    f"| Beta({prior_alpha},{prior_beta}) "
                    f"| lambda={lambda_final}"
                )

                model = (
                    MarginalCopulaModel(
                        make_bank(),
                        CollapsedBayesianCopula(
                            DIM
                        ),
                        DIM,
                        gated=True,
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

                train_collapsed(
                    model,
                    benchmark,
                    prior_alpha,
                    prior_beta,
                    lambda_final,
                    device,
                )

                synchronize(
                    device
                )

                train_time = (
                    time.perf_counter()
                    - start
                )

                (
                    current_edges,
                    graph,
                ) = graph_diagnostics(
                    model,
                    benchmark,
                    prior_alpha,
                    prior_beta,
                    lambda_final,
                    device,
                )

                edge_rows.extend(
                    current_edges
                )

                print(
                    f"  E[edges]="
                    f"{graph['expected_edges']:.2f} "
                    f"| hard="
                    f"{graph['hard_edges']} "
                    f"| p_min="
                    f"{graph['min_probability']:.3f} "
                    f"| p_max="
                    f"{graph['max_probability']:.3f}"
                )

                results = evaluate(
                    model,
                    benchmark,
                    device,
                )

                for row in results:

                    row.update(
                        {
                            "benchmark":
                                benchmark,

                            "model":
                                "collapsed",

                            "prior_alpha":
                                prior_alpha,

                            "prior_beta":
                                prior_beta,

                            "lambda":
                                lambda_final,

                            "expected_edges":
                                graph[
                                    "expected_edges"
                                ],

                            "hard_edges":
                                graph[
                                    "hard_edges"
                                ],

                            "mean_probability":
                                graph[
                                    "mean_probability"
                                ],

                            "min_probability":
                                graph[
                                    "min_probability"
                                ],

                            "max_probability":
                                graph[
                                    "max_probability"
                                ],

                            "prob_std_across_edges":
                                graph[
                                    "prob_std_across_edges"
                                ],

                            "parameter_count":
                                count_parameters(
                                    model
                                ),

                            "train_time":
                                train_time,
                        }
                    )

                    all_rows.append(
                        row
                    )

                # --------------------------------------------
                # Intermediate save
                # --------------------------------------------

                pd.DataFrame(
                    all_rows
                ).to_csv(
                    output_dir
                    / "raw_results.csv",
                    index=False,
                )

                pd.DataFrame(
                    edge_rows
                ).to_csv(
                    output_dir
                    / "edge_probabilities.csv",
                    index=False,
                )

                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        del marginal_bank

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Summary
    # ========================================================

    raw_df = pd.DataFrame(
        all_rows
    )

    summary = (
        raw_df
        .groupby(
            [
                "benchmark",
                "model",
                "prior_alpha",
                "prior_beta",
                "lambda",
                "expected_edges",
                "hard_edges",
                "mean_probability",
                "min_probability",
                "max_probability",
                "prob_std_across_edges",
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

            parameter_count_mean=(
                "parameter_count",
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
        "=" * 110
    )

    print(
        "COLLAPSED BETA-BERNOULLI GRAPH TEST COMPLETE"
    )

    print(
        "=" * 110
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

def parse_prior_list(
    strings,
):

    priors = []

    for item in strings:

        a, b = item.split(",")

        priors.append(
            (
                float(a),
                float(b),
            )
        )

    return priors


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
        "--priors",
        nargs="+",
        default=[
            "1,1",
            "1,3",
        ],
    )

    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=
            DEFAULT_LAMBDAS,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_DIR
        ),
    )

    parser.add_argument(
        "--smoke",
        action="store_true",
    )

    args = parser.parse_args()

    args.priors = (
        parse_prior_list(
            args.priors
        )
    )

    global MARGINAL_STEPS
    global COPULA_STEPS
    global BATCH_SIZE
    global N_EVAL
    global N_KL

    if args.smoke:

        MARGINAL_STEPS = 100
        COPULA_STEPS = 250

        BATCH_SIZE = 1024

        N_EVAL = 10_000
        N_KL = 5_000

        args.lambdas = [
            0.001,
            0.01,
            0.05,
        ]

        args.output = (
            "results/"
            "collapsed_beta_bernoulli_graph_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )