import argparse
import math
import time
from copy import deepcopy
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

FULL_AR_PRETRAIN_STEPS = 1800
GATE_TRAIN_STEPS = 1200
SPARSE_FINETUNE_STEPS = 1200

BATCH_SIZE = 4096

LR_MARGINAL = 1e-4
LR_COPULA = 1e-4
LR_GATE = 1e-4

GRAD_CLIP = 10.0

PRIOR_EDGE_PROB = 0.20

# Soft sparsity strength during gate learning
DEFAULT_BETAS = {
    "curved_ridge": 0.02,
    "sharp_broad": 0.005,
}

PRUNE_THRESHOLDS = [
    0.30,
    0.50,
    0.70,
]

N_EVAL = 100_000
N_KL = 50_000

OUTPUT_DIR = Path(
    "results/gate_prune_retrain"
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


def prior_logit(prob):
    return math.log(
        prob / (1.0 - prob)
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
# Parent structures
# ============================================================

def build_full_ar_parents(dim):

    return {
        j: list(range(j))
        for j in range(dim)
    }


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


def count_edges(parents):

    return sum(
        len(v)
        for v in parents.values()
    )


# ============================================================
# Ordinary structured copula
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

        parents = self.parents[
            j
        ]

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
# Soft gate network
# ============================================================

class ConstantGateMLP(nn.Module):

    def __init__(
        self,
        dim,
        prior_prob,
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

        final_layer = (
            self.net[-1]
        )

        nn.init.zeros_(
            final_layer.weight
        )

        nn.init.constant_(
            final_layer.bias,
            prior_logit(
                prior_prob
            ),
        )

    def forward(
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


# ============================================================
# Soft-gated full AR
# ============================================================

class SoftGatedCopula(nn.Module):

    def __init__(
        self,
        dim,
        prior_prob,
    ):
        super().__init__()

        self.dim = dim

        self.gate_net = (
            ConstantGateMLP(
                dim,
                prior_prob,
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

            # gated u + gate values + c
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

    def gate_probabilities(
        self,
        batch_size,
        device,
    ):

        logits = (
            self.gate_net(
                batch_size,
                device,
            )
        )

        return torch.sigmoid(
            logits
        )

    def _context(
        self,
        u,
        cond,
        gates,
        j,
    ):

        idx = self.edge_index[
            j
        ]

        gate_j = gates[
            :,
            idx
        ]

        previous = (
            u[:, :j]
            - 0.5
        )

        gated_previous = (
            previous
            * gate_j
        )

        return torch.cat(
            [
                gated_previous,
                gate_j,
                cond,
            ],
            dim=1,
        )

    def log_prob_u(
        self,
        u,
        cond,
    ):

        gates = (
            self.gate_probabilities(
                u.shape[0],
                u.device,
            )
        )

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

        gates = (
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
# Full marginal/copula model
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

        cond_batch = cond.expand(
            n,
            -1,
        )

        u = self.copula.sample_u(
            n,
            cond_batch,
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
# KL sparsity penalty
# ============================================================

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


def train_structured(
    model,
    benchmark,
    device,
    steps,
    label,
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
        ) % 300 == 0:

            print(
                f"  {label} "
                f"{step+1}/{steps} "
                f"NLL="
                f"{np.mean(recent[-100:]):.5f}"
            )


def train_soft_gates(
    model,
    benchmark,
    beta,
    device,
):

    optimizer = torch.optim.Adam(
        model.copula.parameters(),
        lr=LR_GATE,
    )

    recent_nll = []
    recent_kl = []
    recent_edges = []

    for step in range(
        GATE_TRAIN_STEPS
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

        probs = (
            model.copula
            .gate_probabilities(
                BATCH_SIZE,
                device,
            )
        )

        logq = (
            model.copula
            .log_prob_u(
                u,
                cond,
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

        recent_nll.append(
            nll.item()
        )

        recent_kl.append(
            kl.item()
        )

        recent_edges.append(
            probs.sum(
                dim=1
            )
            .mean()
            .item()
        )

        if (
            step + 1
        ) % 300 == 0:

            print(
                f"  gates "
                f"{step+1}/{GATE_TRAIN_STEPS} "
                f"NLL="
                f"{np.mean(recent_nll[-100:]):.5f} "
                f"KL="
                f"{np.mean(recent_kl[-100:]):.4f} "
                f"Eedges="
                f"{np.mean(recent_edges[-100:]):.2f}"
            )


# ============================================================
# Full AR -> soft gated initialization
# ============================================================

def copy_full_ar_to_soft(
    full_model,
    soft_model,
):
    """
    Copy spline weights where possible.

    Input dimensions differ because the soft model receives
    [gated_u, gate, c] rather than [u, c].

    Therefore exact first-layer copying is not generally possible.

    We copy matching tensors only.
    """

    full_state = (
        full_model.copula
        .state_dict()
    )

    soft_state = (
        soft_model.copula
        .state_dict()
    )

    copied = 0

    for key in soft_state:

        if (
            key in full_state
            and
            soft_state[key].shape
            == full_state[key].shape
        ):

            soft_state[key] = (
                full_state[key]
                .detach()
                .clone()
            )

            copied += 1

    soft_model.copula.load_state_dict(
        soft_state
    )

    print(
        f"  copied {copied} "
        f"compatible Full-AR tensors"
    )


# ============================================================
# Prune graph
# ============================================================

def extract_pruned_parents(
    soft_model,
    threshold,
    device,
):

    with torch.no_grad():

        probs = (
            soft_model.copula
            .gate_probabilities(
                1,
                device,
            )[0]
            .cpu()
            .numpy()
        )

    edge_pairs = (
        soft_model.copula
        .gate_net
        .edge_pairs
    )

    parents = {
        j: []
        for j in range(DIM)
    }

    selected_edges = []

    for edge_id, (
        parent,
        child,
    ) in enumerate(
        edge_pairs
    ):

        probability = float(
            probs[
                edge_id
            ]
        )

        if probability >= threshold:

            parents[
                child
            ].append(
                parent
            )

            selected_edges.append(
                (
                    parent,
                    child,
                    probability,
                )
            )

    return (
        parents,
        selected_edges,
        probs,
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

            logw = (
                logp - logq
            )

            weights = torch.exp(
                logw
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

    rows = []
    edge_rows = []

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

        # ====================================================
        # Shared marginals
        # ====================================================

        print(
            "\nTraining marginals..."
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

        # freeze helper
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
            )
            .to(device)
        )

        train_structured(
            markov_model,
            benchmark,
            device,
            FULL_AR_PRETRAIN_STEPS,
            "markov1",
        )

        result = evaluate_model(
            markov_model,
            benchmark,
            device,
        )

        for r in result:

            r.update(
                {
                    "benchmark":
                        benchmark,

                    "model":
                        "markov1",

                    "threshold":
                        np.nan,

                    "edges":
                        DIM - 1,

                    "parameter_count":
                        count_parameters(
                            markov_model
                        ),
                }
            )

            rows.append(
                r
            )

        # ====================================================
        # Full AR
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
            )
            .to(device)
        )

        train_structured(
            full_model,
            benchmark,
            device,
            FULL_AR_PRETRAIN_STEPS,
            "full_ar",
        )

        result = evaluate_model(
            full_model,
            benchmark,
            device,
        )

        for r in result:

            r.update(
                {
                    "benchmark":
                        benchmark,

                    "model":
                        "full_ar",

                    "threshold":
                        np.nan,

                    "edges":
                        28,

                    "parameter_count":
                        count_parameters(
                            full_model
                        ),
                }
            )

            rows.append(
                r
            )

        # ====================================================
        # Soft-gated model
        # ====================================================

        print(
            "\nMODEL: soft_gated"
        )

        soft_model = (
            MarginalCopulaModel(
                make_bank(),
                SoftGatedCopula(
                    DIM,
                    PRIOR_EDGE_PROB,
                ),
                DIM,
            )
            .to(device)
        )

        # partial compatible initialization
        copy_full_ar_to_soft(
            full_model,
            soft_model,
        )

        train_soft_gates(
            soft_model,
            benchmark,
            beta,
            device,
        )

        with torch.no_grad():

            gate_probs = (
                soft_model.copula
                .gate_probabilities(
                    1,
                    device,
                )[0]
                .cpu()
                .numpy()
            )

        expected_edges = float(
            gate_probs.sum()
        )

        result = evaluate_model(
            soft_model,
            benchmark,
            device,
        )

        for r in result:

            r.update(
                {
                    "benchmark":
                        benchmark,

                    "model":
                        "soft_gated",

                    "threshold":
                        np.nan,

                    "edges":
                        expected_edges,

                    "parameter_count":
                        count_parameters(
                            soft_model
                        ),
                }
            )

            rows.append(
                r
            )

        # ====================================================
        # Prune + rebuild + retrain
        # ====================================================

        for threshold in (
            args.thresholds
        ):

            print()
            print(
                f"MODEL: pruned "
                f"| threshold={threshold}"
            )

            (
                parents,
                selected_edges,
                probs,
            ) = extract_pruned_parents(
                soft_model,
                threshold,
                device,
            )

            n_edges = count_edges(
                parents
            )

            print(
                f"  selected edges: "
                f"{n_edges}/28"
            )

            for (
                parent,
                child,
                probability,
            ) in selected_edges:

                print(
                    f"    "
                    f"{parent}->{child} "
                    f"p={probability:.4f}"
                )

                edge_rows.append(
                    {
                        "benchmark":
                            benchmark,

                        "threshold":
                            threshold,

                        "parent":
                            parent,

                        "child":
                            child,

                        "probability":
                            probability,
                    }
                )

            sparse_model = (
                MarginalCopulaModel(
                    make_bank(),
                    StructuredCopula(
                        DIM,
                        parents,
                    ),
                    DIM,
                )
                .to(device)
            )

            # IMPORTANT:
            #
            # We intentionally retrain the new sparse architecture.
            # This is the actual test:
            #
            # did the learned gates identify a useful parent graph?

            train_structured(
                sparse_model,
                benchmark,
                device,
                SPARSE_FINETUNE_STEPS,
                f"pruned_tau{threshold}",
            )

            result = evaluate_model(
                sparse_model,
                benchmark,
                device,
            )

            for r in result:

                r.update(
                    {
                        "benchmark":
                            benchmark,

                        "model":
                            "pruned_retrained",

                        "threshold":
                            threshold,

                        "edges":
                            n_edges,

                        "parameter_count":
                            count_parameters(
                                sparse_model
                            ),
                    }
                )

                rows.append(
                    r
                )

            del sparse_model

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del markov_model
        del full_model
        del soft_model
        del marginal_bank

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # intermediate save
        pd.DataFrame(
            rows
        ).to_csv(
            output_dir
            / "raw_results.csv",
            index=False,
        )

        pd.DataFrame(
            edge_rows
        ).to_csv(
            output_dir
            / "selected_edges.csv",
            index=False,
        )

    # ========================================================
    # Summary
    # ========================================================

    raw_df = pd.DataFrame(
        rows
    )

    summary = (
        raw_df
        .groupby(
            [
                "benchmark",
                "model",
                "threshold",
                "edges",
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
        "=" * 90
    )

    print(
        "GATE PRUNE + RETRAIN COMPLETE"
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
        "--thresholds",
        nargs="+",
        type=float,
        default=
            PRUNE_THRESHOLDS,
    )

    parser.add_argument(
        "--output",
        type=str,
        default=str(
            OUTPUT_DIR
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":

    run(
        parse_args()
    )