import argparse
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta, kl_divergence

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

# ------------------------------------------------------------
# Structure regularization
#
# This controls the overall strength of the hierarchical
# Bayesian structure regularizer relative to copula NLL.
#
# We sweep this separately from the Beta prior itself.
# ------------------------------------------------------------

DEFAULT_STRUCTURE_WEIGHTS = [
    0.01,
    0.05,
    0.10,
]

BETA_WARMUP_FRAC = 0.20

# ------------------------------------------------------------
# Priors for rho
#
# Beta(1,1): neutral graph-density prior
# Beta(1,3): mild sparse preference
# ------------------------------------------------------------

DEFAULT_PRIORS = [
    (1.0, 1.0),
    (1.0, 3.0),
]

N_EVAL = 100_000
N_KL = 50_000

OUTPUT_DIR = Path(
    "results/hierarchical_bayesian_graph"
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

    t = step / (total_steps - 1)

    return (
        start
        + t * (end - start)
    )


def structure_weight_schedule(
    step,
    total_steps,
    final_weight,
):
    warmup = int(
        BETA_WARMUP_FRAC
        * total_steps
    )

    if step < warmup:
        return 0.0

    denom = max(
        total_steps - warmup,
        1,
    )

    t = (
        step - warmup
    ) / denom

    t = min(
        max(t, 0.0),
        1.0,
    )

    return (
        final_weight * t
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
# Standard structured copula
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
                + len(parents[j])
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

            context = (
                self._context(
                    u,
                    cond,
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

            logq -= (
                forward_logdet
                .squeeze(-1)
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

            context = (
                self._context(
                    u,
                    cond,
                    j,
                )
            )

            v = torch.rand(
                n,
                1,
                device=cond.device,
            )

            uj, _ = (
                self.conditionals[
                    str(j)
                ].forward(
                    v,
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
# Constant MLP edge posterior q(Z)
# ============================================================

class EdgePosteriorMLP(nn.Module):

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

        final = self.net[-1]

        nn.init.zeros_(
            final.weight
        )

        nn.init.constant_(
            final.bias,
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
# Variational posterior q(rho)
# ============================================================

class GraphDensityPosterior(nn.Module):

    def __init__(
        self,
        initial_mean=0.5,
        initial_concentration=4.0,
    ):
        super().__init__()

        alpha0 = (
            initial_mean
            * initial_concentration
        )

        beta0 = (
            (1.0 - initial_mean)
            * initial_concentration
        )

        self.raw_alpha = nn.Parameter(
            torch.tensor(
                math.log(
                    math.exp(alpha0)
                    - 1.0
                ),
                dtype=torch.float32,
            )
        )

        self.raw_beta = nn.Parameter(
            torch.tensor(
                math.log(
                    math.exp(beta0)
                    - 1.0
                ),
                dtype=torch.float32,
            )
        )

    def parameters_positive(
        self,
    ):

        alpha = (
            F.softplus(
                self.raw_alpha
            )
            + 1e-4
        )

        beta = (
            F.softplus(
                self.raw_beta
            )
            + 1e-4
        )

        return (
            alpha,
            beta,
        )

    def distribution(
        self,
    ):

        alpha, beta = (
            self.parameters_positive()
        )

        return Beta(
            alpha,
            beta,
        )

    def mean(
        self,
    ):

        alpha, beta = (
            self.parameters_positive()
        )

        return (
            alpha
            /
            (
                alpha + beta
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
# KL(q(Z) || p(Z | rho))
# ============================================================

def bernoulli_kl_to_rho(
    probs,
    rho,
):

    eps = 1e-7

    probs = probs.clamp(
        eps,
        1.0 - eps,
    )

    rho = rho.clamp(
        eps,
        1.0 - eps,
    )

    return (
        probs
        * (
            torch.log(probs)
            - torch.log(rho)
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
                1.0 - rho
            )
        )
    )


# ============================================================
# Hierarchical Bayesian gated copula
# ============================================================

class HierarchicalBayesianCopula(nn.Module):

    def __init__(
        self,
        dim,
    ):
        super().__init__()

        self.dim = dim

        # q(Z)
        self.edge_posterior = (
            EdgePosteriorMLP(
                dim=dim,
                initial_prob=0.5,
            )
        )

        # q(rho)
        self.rho_posterior = (
            GraphDensityPosterior(
                initial_mean=0.5,
                initial_concentration=4.0,
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

            # gated centered u
            # + gate values
            # + condition c

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
            self.edge_posterior
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
            self.edge_posterior
            .probabilities(
                batch_size,
                device,
            )
        )

    def rho_mean(
        self,
    ):

        return (
            self.rho_posterior
            .mean()
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

            logq -= (
                forward_logdet
                .squeeze(-1)
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

            context = (
                self._context(
                    u,
                    cond,
                    probs,
                    j,
                )
            )

            v = torch.rand(
                n,
                1,
                device=cond.device,
            )

            uj, _ = (
                self.conditionals[
                    str(j)
                ].forward(
                    v,
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
# Combined marginal/copula model
# ============================================================

class MarginalCopulaModel(nn.Module):

    def __init__(
        self,
        marginal_bank,
        copula,
        dim,
        hierarchical=False,
    ):

        super().__init__()

        self.dim = dim

        self.marginals = (
            marginal_bank.marginals
        )

        self.copula = copula

        self.hierarchical = (
            hierarchical
        )

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

            log_marg -= (
                forward_logdet
                .squeeze(-1)
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

        if self.hierarchical:

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
            log_marg
            + log_copula
        )

    def sample(
        self,
        n,
        cond,
    ):

        cond_batch = (
            cond.expand(
                n,
                -1,
            )
        )

        if self.hierarchical:

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
    model,
    benchmark,
    device,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
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
# Standard copula training
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
# Hierarchical Bayesian training
# ============================================================

def train_hierarchical(
    model,
    benchmark,
    prior_alpha,
    prior_beta,
    structure_weight_final,
    device,
):

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_COPULA,
    )

    prior_rho = Beta(
        torch.tensor(
            prior_alpha,
            device=device,
        ),
        torch.tensor(
            prior_beta,
            device=device,
        ),
    )

    recent_nll = []
    recent_edge_kl = []
    recent_rho_kl = []
    recent_rho = []
    recent_edges = []

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

        temperature = (
            linear_schedule(
                step,
                COPULA_STEPS,
                TEMPERATURE_START,
                TEMPERATURE_END,
            )
        )

        structure_weight = (
            structure_weight_schedule(
                step,
                COPULA_STEPS,
                structure_weight_final,
            )
        )

        # ----------------------------------------------------
        # q(Z)
        # ----------------------------------------------------

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
        # q(rho)
        # ----------------------------------------------------

        q_rho = (
            model.copula
            .rho_posterior
            .distribution()
        )

        # reparameterized Beta sample
        rho_sample = (
            q_rho.rsample()
        )

        # ----------------------------------------------------
        # likelihood
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
        # KL for edge posterior
        #
        # average over edges so scale does not grow with d
        # ----------------------------------------------------

        edge_kl = (
            bernoulli_kl_to_rho(
                probs,
                rho_sample,
            )
            .mean()
        )

        # ----------------------------------------------------
        # KL for graph-density posterior
        # ----------------------------------------------------

        rho_kl = (
            kl_divergence(
                q_rho,
                prior_rho,
            )
        )

        # normalize rho KL by number of edges
        rho_kl_scaled = (
            rho_kl
            / model.copula
            .edge_posterior
            .n_edges
        )

        structure_loss = (
            edge_kl
            + rho_kl_scaled
        )

        loss = (
            nll
            + structure_weight
            * structure_loss
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.copula.parameters(),
            GRAD_CLIP,
        )

        optimizer.step()

        rho_mean = (
            model.copula
            .rho_mean()
            .item()
        )

        expected_edges = (
            probs[0]
            .sum()
            .item()
        )

        recent_nll.append(
            nll.item()
        )

        recent_edge_kl.append(
            edge_kl.item()
        )

        recent_rho_kl.append(
            rho_kl.item()
        )

        recent_rho.append(
            rho_mean
        )

        recent_edges.append(
            expected_edges
        )

        if (
            step + 1
        ) % 250 == 0:

            print(
                f"  hier "
                f"{step+1:4d}/{COPULA_STEPS} "
                f"NLL="
                f"{np.mean(recent_nll[-100:]):.5f} "
                f"edgeKL="
                f"{np.mean(recent_edge_kl[-100:]):.4f} "
                f"rhoKL="
                f"{np.mean(recent_rho_kl[-100:]):.4f} "
                f"rho="
                f"{np.mean(recent_rho[-100:]):.3f} "
                f"Eedges="
                f"{np.mean(recent_edges[-100:]):.2f} "
                f"T={temperature:.3f} "
                f"lambda={structure_weight:.4f}"
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
            cond.to(device)
            .reshape(
                1,
                -1,
            )
        )

        cond_eval = (
            cond.expand(
                N_EVAL,
                -1,
            )
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

            estimate = (
                weights.mean()
            )

            ess = (
                weights.sum() ** 2
                /
                (
                    N_EVAL
                    * weights.square().sum()
                )
            )

            variance = (
                weights.var(
                    unbiased=True
                )
                / N_EVAL
            )

        # forward KL

        cond_kl = (
            cond.expand(
                N_KL,
                -1,
            )
        )

        with torch.no_grad():

            x_target = (
                sample_target(
                    benchmark,
                    cond_kl,
                    DIM,
                )
            )

            logp_target = (
                target_logpdf(
                    benchmark,
                    x_target,
                    cond_kl,
                )
            )

            logq_target = (
                model.log_prob(
                    x_target,
                    cond_kl,
                )
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
    structure_weight,
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

        rho_mean = (
            model.copula
            .rho_mean()
            .item()
        )

        alpha_q, beta_q = (
            model.copula
            .rho_posterior
            .parameters_positive()
        )

        alpha_q = (
            alpha_q.item()
        )

        beta_q = (
            beta_q.item()
        )

    edges = (
        model.copula
        .edge_posterior
        .edge_pairs
    )

    rows = []

    for edge_id, (
        parent,
        child,
    ) in enumerate(
        edges
    ):

        rows.append(
            {
                "benchmark":
                    benchmark,

                "prior_alpha":
                    prior_alpha,

                "prior_beta":
                    prior_beta,

                "structure_weight":
                    structure_weight,

                "parent":
                    parent,

                "child":
                    child,

                "probability":
                    float(
                        probs[edge_id]
                    ),

                "selected_0p5":
                    int(
                        probs[edge_id]
                        >= 0.5
                    ),
            }
        )

    summary = {
        "rho_mean":
            rho_mean,

        "rho_alpha_q":
            alpha_q,

        "rho_beta_q":
            beta_q,

        "expected_edges":
            float(
                probs.sum()
            ),

        "hard_edges_0p5":
            int(
                (
                    probs >= 0.5
                ).sum()
            ),

        "mean_edge_probability":
            float(
                probs.mean()
            ),

        "min_edge_probability":
            float(
                probs.min()
            ),

        "max_edge_probability":
            float(
                probs.max()
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
    all_edge_rows = []

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
        # Shared marginal bank
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
            k:
            v.detach()
            .cpu()
            .clone()

            for k, v
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
        # Markov baseline
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
                hierarchical=False,
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

                    "structure_weight":
                        np.nan,

                    "rho_mean":
                        np.nan,

                    "expected_edges":
                        7.0,

                    "hard_edges":
                        7.0,

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
                hierarchical=False,
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

                    "structure_weight":
                        np.nan,

                    "rho_mean":
                        np.nan,

                    "expected_edges":
                        28.0,

                    "hard_edges":
                        28.0,

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
        # Hierarchical Bayesian models
        # ====================================================

        for (
            prior_alpha,
            prior_beta,
        ) in args.priors:

            for structure_weight in (
                args.structure_weights
            ):

                print()
                print(
                    "MODEL: hierarchical "
                    f"| Beta({prior_alpha},{prior_beta}) "
                    f"| lambda={structure_weight}"
                )

                model = (
                    MarginalCopulaModel(
                        make_bank(),
                        HierarchicalBayesianCopula(
                            DIM
                        ),
                        DIM,
                        hierarchical=True,
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

                train_hierarchical(
                    model,
                    benchmark,
                    prior_alpha,
                    prior_beta,
                    structure_weight,
                    device,
                )

                synchronize(
                    device
                )

                train_time = (
                    time.perf_counter()
                    - start
                )

                edge_rows, graph = (
                    graph_diagnostics(
                        model,
                        benchmark,
                        prior_alpha,
                        prior_beta,
                        structure_weight,
                        device,
                    )
                )

                all_edge_rows.extend(
                    edge_rows
                )

                print(
                    "  learned rho="
                    f"{graph['rho_mean']:.4f} "
                    "| E[edges]="
                    f"{graph['expected_edges']:.2f} "
                    "| hard edges="
                    f"{graph['hard_edges_0p5']}"
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
                                "hierarchical",

                            "prior_alpha":
                                prior_alpha,

                            "prior_beta":
                                prior_beta,

                            "structure_weight":
                                structure_weight,

                            "rho_mean":
                                graph[
                                    "rho_mean"
                                ],

                            "rho_alpha_q":
                                graph[
                                    "rho_alpha_q"
                                ],

                            "rho_beta_q":
                                graph[
                                    "rho_beta_q"
                                ],

                            "expected_edges":
                                graph[
                                    "expected_edges"
                                ],

                            "hard_edges":
                                graph[
                                    "hard_edges_0p5"
                                ],

                            "mean_edge_probability":
                                graph[
                                    "mean_edge_probability"
                                ],

                            "min_edge_probability":
                                graph[
                                    "min_edge_probability"
                                ],

                            "max_edge_probability":
                                graph[
                                    "max_edge_probability"
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

                del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # intermediate save
                pd.DataFrame(
                    all_rows
                ).to_csv(
                    output_dir
                    / "raw_results.csv",
                    index=False,
                )

                pd.DataFrame(
                    all_edge_rows
                ).to_csv(
                    output_dir
                    / "edge_probabilities.csv",
                    index=False,
                )

        del marginal_bank

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Final summary
    # ========================================================

    raw_df = pd.DataFrame(
        all_rows
    )

    group_cols = [
        "benchmark",
        "model",
        "prior_alpha",
        "prior_beta",
        "structure_weight",
        "rho_mean",
        "expected_edges",
        "hard_edges",
    ]

    summary = (
        raw_df
        .groupby(
            group_cols,
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
        "=" * 100
    )

    print(
        "HIERARCHICAL BAYESIAN GRAPH TEST COMPLETE"
    )

    print(
        "=" * 100
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
        "--structure-weights",
        nargs="+",
        type=float,
        default=
            DEFAULT_STRUCTURE_WEIGHTS,
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

        args.structure_weights = [
            0.01,
            0.10,
        ]

        args.output = (
            "results/"
            "hierarchical_bayesian_graph_smoke"
        )

    return args


if __name__ == "__main__":

    run(
        parse_args()
    )