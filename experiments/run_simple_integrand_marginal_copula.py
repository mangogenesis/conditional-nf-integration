#!/usr/bin/env python3
"""Simple integrand-only training: marginals first, copula second.

This deliberately omits Bayesian graph search, candidate pruning, adaptive KL
rounds, and target sampling.  At each fixed annealing temperature it:

1. builds one self-normalized importance cache using only f(x | c),
2. trains all conditional marginals and restores their best validation state,
3. freezes the marginals and caches u_i = F_i(x_i | c),
4. trains the copula from cached u and restores its best validation state.

The final integral and ESS are evaluated with fresh defensive-mixture samples.
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from run_bayesian_selected_copula_benchmark import (
    DIM as IMPORTED_DIM,
    MarginalBank,
    MarginalCopulaModel,
    StructuredCopula,
    full_ar_parents,
    markov1_parents,
)


DIM = 32
COND_DIM = 3
TRUE_INTEGRAL = 1.0
LOG_2PI = math.log(2.0 * math.pi)
BRIDGE_COPULA_LR_LEVELS = (1.0e-3, 3.0e-4, 1.0e-4)
WARM_START_LR_LEVELS = (3.0e-4, 1.0e-4, 3.0e-5)
CHI2_LR_LEVELS = (1.0e-4, 3.0e-5, 1.0e-5)
FORMAL_COPULA_LR_LEVELS = (
    1.0e-3, 3.0e-4, 1.0e-4, 3.0e-5, 1.0e-5,
)
def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_adam(parameters, lr: float):
    parameters = list(parameters)
    if parameters and parameters[0].is_cuda:
        try:
            return torch.optim.Adam(parameters, lr=lr, fused=True)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(parameters, lr=lr)


def defensive_uniform_fraction(args, beta):
    """Retain global support while reducing uniform waste as beta increases."""
    return max(
        args.min_uniform_fraction,
        args.uniform_fraction * (1.0 - float(beta)),
    )


def cpu_state(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def load_state(module, state):
    module.load_state_dict(state, strict=True)


def sparse_wave_32_log_integrand(x: torch.Tensor, cond: torch.Tensor):
    """The benchmark density; its integral is not used during training."""
    log_f = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    c0, c1, c2 = cond[:, 0], cond[:, 1], cond[:, 2]
    gate = torch.sigmoid(6.0 * (c0 - 0.5))
    for child in range(2, DIM):
        parent = child - 1
        skip = max(0, child // 2 - 1)
        a, b = x[:, parent], x[:, skip]
        phase = 0.031 * child + 0.10 * c1
        signal = (
            gate * torch.sin(2.0 * math.pi * (a + phase))
            + (1.0 - gate) * torch.cos(2.0 * math.pi * (b - 0.07 * c0))
            + 0.15 * torch.sin(2.0 * math.pi * (a - b))
        )
        mu = 0.50 + 0.25 * torch.tanh(1.35 * signal)
        sigma = 0.060 + 0.018 * c2 + 0.003 * float(child % 3)
        value = x[:, child]
        z = (value - mu) / sigma
        log_base = -0.5 * z.square() - torch.log(sigma) - 0.5 * LOG_2PI
        lower = (0.0 - mu) / sigma
        upper = (1.0 - mu) / sigma
        normalizer = (
            torch.special.ndtr(upper) - torch.special.ndtr(lower)
        ).clamp_min(1.0e-12)
        log_f += log_base - torch.log(normalizer)
    return log_f


@torch.no_grad()
def sample_proposal(model, cond, uniform_fraction):
    n = cond.shape[0]
    if model is None:
        return torch.rand(n, DIM, device=cond.device), torch.zeros(
            n, device=cond.device
        )
    x = model.sample(n, cond)
    use_uniform = torch.rand(n, device=cond.device) < uniform_fraction
    if use_uniform.any():
        x[use_uniform] = torch.rand(
            int(use_uniform.sum()), DIM, device=cond.device
        )
    log_q = model.log_prob(x, cond)
    log_r = torch.logaddexp(
        log_q + math.log1p(-uniform_fraction),
        torch.full_like(log_q, math.log(uniform_fraction)),
    )
    return x, log_r


@torch.no_grad()
def build_cache(model, beta, conditions, points, device, seed, uniform_fraction):
    seed_everything(seed)
    anchors = torch.rand(conditions, COND_DIM, device=device)
    cond = anchors[:, None].expand(conditions, points, COND_DIM).reshape(-1, COND_DIM)
    x, log_r = sample_proposal(model, cond, uniform_fraction)
    log_f = sparse_wave_32_log_integrand(x, cond)
    logw = (beta * log_f.double() - log_r.double()).reshape(conditions, points)
    alpha = torch.softmax(logw, dim=1).float()
    raw_ess = 1.0 / alpha.double().square().sum(dim=1)
    return {
        "x": x.reshape(conditions, points, DIM).detach(),
        "cond": cond.reshape(conditions, points, COND_DIM).detach(),
        "log_f": log_f.reshape(conditions, points).detach(),
        "log_r": log_r.reshape(conditions, points).detach(),
        "alpha": alpha.detach(),
        "raw_ess": raw_ess.detach(),
    }


@torch.no_grad()
def select_next_beta(model, current_beta, args, device, seed):
    """Choose the largest beta increment satisfying conditional ESS targets."""
    seed_everything(seed)
    conditions = args.temperature_probe_conditions
    points = args.temperature_probe_points
    anchors = torch.rand(conditions, COND_DIM, device=device)
    cond = anchors[:, None].expand(conditions, points, COND_DIM).reshape(-1, COND_DIM)
    uniform_fraction = defensive_uniform_fraction(args, current_beta)
    x, log_r = sample_proposal(model, cond, uniform_fraction)
    log_f = sparse_wave_32_log_integrand(x, cond).double().reshape(conditions, points)
    log_r = log_r.double().reshape(conditions, points)

    def diagnostics(beta):
        alpha = torch.softmax(beta * log_f - log_r, dim=1)
        raw_ess = 1.0 / alpha.square().sum(dim=1)
        return float(raw_ess.median()), float(torch.quantile(raw_ess, 0.05))

    upper = min(1.0, current_beta + args.temperature_max_increment)
    median_ess, q05_ess = diagnostics(upper)
    if (
        median_ess >= args.temperature_median_ess
        and q05_ess >= args.temperature_q05_ess
    ):
        chosen = upper
    else:
        low, high = current_beta, upper
        for _ in range(16):
            middle = 0.5 * (low + high)
            middle_median, middle_q05 = diagnostics(middle)
            if (
                middle_median >= args.temperature_median_ess
                and middle_q05 >= args.temperature_q05_ess
            ):
                low = middle
            else:
                high = middle
        chosen = max(
            low,
            min(1.0, current_beta + args.temperature_min_increment),
        )
        median_ess, q05_ess = diagnostics(chosen)

    print(
        f"TEMPERATURE SELECT: {current_beta:.4f} -> {chosen:.4f} | "
        f"median raw ESS={median_ess:.1f} q05={q05_ess:.1f} | "
        f"uniform={uniform_fraction:.3f}",
        flush=True,
    )
    if median_ess < args.temperature_hard_median or q05_ess < args.temperature_hard_q05:
        raise RuntimeError(
            "No numerically safe annealing increment: "
            f"median raw ESS={median_ess:.2f}, q05={q05_ess:.2f}. "
            "Increase --temperature-probe-points or reduce "
            "--temperature-min-increment."
        )
    return chosen, uniform_fraction


def marginal_log_prob(model, x, cond):
    _, value = model.x_to_u_and_log_marginal(x, cond)
    return value


@torch.no_grad()
def systematic_resample_marginals(cache, draws, seed):
    """Convert conditional weights into low-variance sample multiplicities."""
    seed_everything(seed)
    conditions, points, _ = cache["x"].shape
    draws = points if draws <= 0 else draws
    cdf = cache["alpha"].cumsum(dim=1)
    cdf[:, -1] = 1.0
    offset = torch.rand(conditions, 1, device=cdf.device) / draws
    positions = offset + (
        torch.arange(draws, device=cdf.device, dtype=cdf.dtype)[None, :] / draws
    )
    indices = torch.searchsorted(cdf.contiguous(), positions.contiguous())
    gather_x = indices[..., None].expand(-1, -1, DIM)
    gather_c = indices[..., None].expand(-1, -1, COND_DIM)
    return {
        "x": torch.gather(cache["x"], 1, gather_x).reshape(-1, DIM).detach(),
        "cond": torch.gather(cache["cond"], 1, gather_c)
        .reshape(-1, COND_DIM).detach(),
        "draws_per_condition": draws,
    }


@torch.no_grad()
def weighted_marginal_nll(model, cache, chunk=16_384):
    model.eval()
    x = cache["x"].reshape(-1, DIM)
    cond = cache["cond"].reshape(-1, COND_DIM)
    alpha = cache["alpha"].reshape(-1)
    total = torch.zeros((), device=x.device, dtype=torch.float64)
    for start in range(0, x.shape[0], chunk):
        end = min(start + chunk, x.shape[0])
        logq = marginal_log_prob(model, x[start:end], cond[start:end])
        total += -(alpha[start:end].double() * logq.double()).sum()
    return float(total.item() / cache["x"].shape[0])


def train_marginals(
    model, train, val, steps, lr_levels, check, patience, batch_size,
    resamples_per_condition, seed,
):
    for parameter in model.copula.parameters():
        parameter.requires_grad_(False)
    parameters = [p for marginal in model.marginals for p in marginal.parameters()]
    for parameter in parameters:
        parameter.requires_grad_(True)
    lr_levels = list(lr_levels)
    if not lr_levels:
        raise ValueError("marginal learning-rate ladder cannot be empty")
    lr_index = 0
    lr = lr_levels[lr_index]
    optimizer = make_adam(parameters, lr)
    resampled = systematic_resample_marginals(
        train, resamples_per_condition, seed
    )
    train_x = resampled["x"]
    train_cond = resampled["cond"]
    n_train = train_x.shape[0]
    best = weighted_marginal_nll(model, val)
    best_state = cpu_state(model.marginals)
    stale = 0
    print(
        f"marginal resampling: {resampled['draws_per_condition']} draws/c, "
        f"{n_train:,} total training points",
        flush=True,
    )
    for block_start in range(0, steps, check):
        losses = []
        for _ in range(min(check, steps - block_start)):
            idx = torch.randint(
                n_train,
                (min(batch_size, n_train),),
                device=train_x.device,
            )
            loss = -marginal_log_prob(
                model, train_x[idx], train_cond[idx]
            ).mean()
            if not torch.isfinite(loss):
                load_state(model.marginals, best_state)
                if lr_index == len(lr_levels) - 1:
                    raise RuntimeError("non-finite marginal loss at minimum LR")
                lr_index += 1
                lr = lr_levels[lr_index]
                optimizer = make_adam(parameters, lr)
                print(f"  non-finite marginal loss: LR -> {lr:.1e}", flush=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 10.0)
            optimizer.step()
            losses.append(loss.detach())
        step = block_start + len(losses)
        score = weighted_marginal_nll(model, val)
        if score < best - 1.0e-4 * max(1.0, abs(best)):
            best, best_state, stale = score, cpu_state(model.marginals), 0
        else:
            stale += 1
        print(
            f"marginal {step:5d} train="
            f"{(torch.stack(losses).mean().item() if losses else float('nan')):.6f} "
            f"val={score:.6f} best={best:.6f} lr={lr:.1e} "
            f"patience={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            load_state(model.marginals, best_state)
            if lr_index == len(lr_levels) - 1:
                print("  MARGINAL CONVERGED at minimum LR", flush=True)
                break
            old_lr = lr
            lr_index += 1
            lr = lr_levels[lr_index]
            optimizer = make_adam(parameters, lr)
            stale = 0
            print(
                f"  MARGINAL LR DECAY {old_lr:.1e} -> {lr:.1e}",
                flush=True,
            )
    load_state(model.marginals, best_state)


@torch.no_grad()
def cache_copula_coordinates(model, cache, chunk=16_384):
    model.eval()
    x = cache["x"].reshape(-1, DIM)
    cond = cache["cond"].reshape(-1, COND_DIM)
    values = []
    for start in range(0, x.shape[0], chunk):
        u, _ = model.x_to_u_and_log_marginal(
            x[start:start + chunk], cond[start:start + chunk]
        )
        values.append(u)
    result = dict(cache)
    result["u"] = torch.cat(values).reshape_as(cache["x"]).detach()
    return result


@torch.no_grad()
def weighted_copula_nll(model, cache, chunk=16_384):
    model.eval()
    u = cache["u"].reshape(-1, DIM)
    cond = cache["cond"].reshape(-1, COND_DIM)
    alpha = cache["alpha"].reshape(-1)
    total = torch.zeros((), device=u.device, dtype=torch.float64)
    for start in range(0, u.shape[0], chunk):
        end = min(start + chunk, u.shape[0])
        logq = model.copula.log_prob_u(u[start:end], cond[start:end])
        total += -(alpha[start:end].double() * logq.double()).sum()
    return float(total.item() / cache["u"].shape[0])


def train_copula(
    model, train, val, steps, lr_levels, check, patience, batch_size,
):
    for marginal in model.marginals:
        for parameter in marginal.parameters():
            parameter.requires_grad_(False)
    for parameter in model.copula.parameters():
        parameter.requires_grad_(True)
    lr_levels = list(lr_levels)
    if not lr_levels:
        raise ValueError("copula learning-rate ladder cannot be empty")
    lr_index = 0
    lr = lr_levels[lr_index]
    optimizer = make_adam(model.copula.parameters(), lr)
    best = weighted_copula_nll(model, val)
    best_state = cpu_state(model.copula)
    stale = 0
    c_count, points, _ = train["u"].shape
    c_batch = min(c_count, max(1, batch_size // points))
    for block_start in range(0, steps, check):
        losses = []
        for _ in range(min(check, steps - block_start)):
            idx = torch.randint(c_count, (c_batch,), device=train["u"].device)
            u, cond, alpha = train["u"][idx], train["cond"][idx], train["alpha"][idx]
            logq = model.copula.log_prob_u(
                u.reshape(-1, DIM), cond.reshape(-1, COND_DIM)
            ).reshape(c_batch, points)
            loss = -(alpha * logq).sum(dim=1).mean()
            if not torch.isfinite(loss):
                load_state(model.copula, best_state)
                if lr_index == len(lr_levels) - 1:
                    raise RuntimeError("non-finite copula loss at minimum LR")
                lr_index += 1
                lr = lr_levels[lr_index]
                optimizer = make_adam(model.copula.parameters(), lr)
                print(f"  non-finite loss: LR -> {lr:.1e}", flush=True)
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.copula.parameters(), 10.0)
            optimizer.step()
            losses.append(loss.detach())
        step = block_start + len(losses)
        score = weighted_copula_nll(model, val)
        if score < best - 1.0e-4 * max(1.0, abs(best)):
            best, best_state, stale = score, cpu_state(model.copula), 0
        else:
            stale += 1
        print(
            f"copula   {step:5d} train="
            f"{(torch.stack(losses).mean().item() if losses else float('nan')):.6f} "
            f"val={score:.6f} best={best:.6f} lr={lr:.1e} "
            f"patience={stale}/{patience}",
            flush=True,
        )
        if stale >= patience:
            load_state(model.copula, best_state)
            if lr_index == len(lr_levels) - 1:
                print("  COPULA CONVERGED at minimum LR", flush=True)
                break
            old_lr = lr
            lr_index += 1
            lr = lr_levels[lr_index]
            optimizer = make_adam(model.copula.parameters(), lr)
            stale = 0
            print(f"  LR DECAY {old_lr:.1e} -> {lr:.1e}", flush=True)
    load_state(model.copula, best_state)


@torch.no_grad()
def chi2_validation_loss(model, cache, condition_chunk=8):
    """Log second moment of f/q, estimated from the frozen proposal cache."""
    model.eval()
    values = []
    conditions, points, _ = cache["x"].shape
    for start in range(0, conditions, condition_chunk):
        end = min(start + condition_chunk, conditions)
        x = cache["x"][start:end]
        cond = cache["cond"][start:end]
        logq = model.log_prob(
            x.reshape(-1, DIM), cond.reshape(-1, COND_DIM)
        ).reshape(end - start, points).double()
        log_second = (
            2.0 * cache["log_f"][start:end].double()
            - logq
            - cache["log_r"][start:end].double()
        )
        values.append(torch.logsumexp(log_second, dim=1) - math.log(points))
    values = torch.cat(values)
    priority = torch.sqrt(
        cache["raw_ess"].median() / cache["raw_ess"].clamp_min(1.0)
    ).clamp(1.0, 4.0)
    return float((values * priority).sum().div(priority.sum()).item())


def train_copula_chi2(
    model, train, val, steps, lr_levels, check, patience, batch_size,
):
    """Fine-tune q by minimizing the conditional Pearson chi-square objective."""
    for marginal in model.marginals:
        for parameter in marginal.parameters():
            parameter.requires_grad_(False)
    parameters = list(model.copula.parameters())
    for parameter in parameters:
        parameter.requires_grad_(True)
    lr_levels = list(lr_levels)
    lr_index = 0
    optimizer = make_adam(parameters, lr_levels[lr_index])
    best = chi2_validation_loss(model, val)
    best_state = cpu_state(model.copula)
    stale = 0
    conditions, points, _ = train["x"].shape
    c_batch = min(conditions, max(1, batch_size // points))
    condition_priority = torch.sqrt(
        train["raw_ess"].median() / train["raw_ess"].clamp_min(1.0)
    ).clamp(1.0, 4.0)
    for block_start in range(0, steps, check):
        losses = []
        for _ in range(min(check, steps - block_start)):
            idx = torch.randint(conditions, (c_batch,), device=train["x"].device)
            x, cond = train["x"][idx], train["cond"][idx]
            logq = model.log_prob(
                x.reshape(-1, DIM), cond.reshape(-1, COND_DIM)
            ).reshape(c_batch, points).double()
            log_second = (
                2.0 * train["log_f"][idx].double()
                - logq
                - train["log_r"][idx].double()
            )
            per_condition = torch.logsumexp(log_second, dim=1) - math.log(points)
            priority = condition_priority[idx]
            loss = (per_condition * priority).sum() / priority.sum()
            if not torch.isfinite(loss):
                load_state(model.copula, best_state)
                if lr_index == len(lr_levels) - 1:
                    break
                lr_index += 1
                optimizer = make_adam(parameters, lr_levels[lr_index])
                continue
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            losses.append(loss.detach())
        score = chi2_validation_loss(model, val)
        if score < best - 1.0e-4 * max(1.0, abs(best)):
            best, best_state, stale = score, cpu_state(model.copula), 0
        else:
            stale += 1
        print(
            f"chi2     {block_start + len(losses):5d} train="
            f"{(torch.stack(losses).mean().item() if losses else float('nan')):.6f} "
            f"val={score:.6f} best={best:.6f} lr={lr_levels[lr_index]:.1e} "
            f"patience={stale}/{patience}", flush=True,
        )
        if stale >= patience:
            load_state(model.copula, best_state)
            if lr_index == len(lr_levels) - 1:
                print("  CHI2 CONVERGED at minimum LR", flush=True)
                break
            lr_index += 1
            optimizer = make_adam(parameters, lr_levels[lr_index])
            stale = 0
            print(f"  CHI2 LR -> {lr_levels[lr_index]:.1e}", flush=True)
    load_state(model.copula, best_state)


@torch.no_grad()
def evaluate(
    model, device, conditions, points, seed, uniform_fraction,
    return_max_weight=False,
):
    seed_everything(seed)
    anchors = torch.rand(conditions, COND_DIM, device=device)
    cond = anchors[:, None].expand(conditions, points, COND_DIM).reshape(-1, COND_DIM)
    x, log_r = sample_proposal(model, cond, uniform_fraction)
    log_f = sparse_wave_32_log_integrand(x, cond)
    logw = (log_f.double() - log_r.double()).reshape(conditions, points)
    scale = logw.max(dim=1, keepdim=True).values
    w_scaled = torch.exp(logw - scale)
    mean_w = w_scaled.mean(dim=1) * torch.exp(scale.squeeze(1))
    ess = w_scaled.sum(dim=1).square() / (
        points * w_scaled.square().sum(dim=1)
    )
    estimates = mean_w.cpu().numpy()
    esses = ess.cpu().numpy()
    max_weights = (w_scaled / w_scaled.sum(dim=1, keepdim=True)).max(dim=1).values.cpu().numpy()
    if return_max_weight:
        return estimates, esses, max_weights
    return estimates, esses


def tune_defensive_fraction(model, args, device, seed):
    """Choose tail protection without sacrificing more than 10% of mean ESS."""
    low, high = args.min_uniform_fraction, args.uniform_fraction
    fractions = np.geomspace(low, high, 5) if high > low else np.array([low])
    trials = []
    for fraction in fractions:
        _, ess = evaluate(
            model, device, min(args.eval_conditions, 64), args.eval_points,
            seed, float(fraction),
        )
        trials.append((float(fraction), float(ess.mean()), float(np.quantile(ess, 0.05))))
    best_mean = max(item[1] for item in trials)
    eligible = [item for item in trials if item[1] >= 0.90 * best_mean]
    chosen = max(eligible, key=lambda item: (item[2], item[1]))
    print("\nDEFENSIVE MIXTURE TUNING", flush=True)
    for fraction, mean_ess, q05_ess in trials:
        marker = " SELECTED" if fraction == chosen[0] else ""
        print(
            f"uniform={fraction:.4f} mean ESS={mean_ess:.6f} "
            f"q05 ESS={q05_ess:.6f}{marker}", flush=True,
        )
    return chosen[0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--graph", choices=("markov1", "full_ar"), default="full_ar")
    parser.add_argument("--max-annealing-stages", type=int, default=100)
    parser.add_argument("--temperature-probe-conditions", type=int, default=256)
    parser.add_argument("--temperature-probe-points", type=int, default=2048)
    parser.add_argument("--temperature-median-ess", type=float, default=24.0)
    parser.add_argument("--temperature-q05-ess", type=float, default=4.0)
    parser.add_argument("--temperature-hard-median", type=float, default=5.0)
    parser.add_argument("--temperature-hard-q05", type=float, default=1.5)
    parser.add_argument("--temperature-min-increment", type=float, default=0.02)
    parser.add_argument("--temperature-max-increment", type=float, default=0.35)
    parser.add_argument("--train-conditions", type=int, default=256)
    parser.add_argument("--val-conditions", type=int, default=256)
    parser.add_argument("--points", type=int, default=4096)
    parser.add_argument(
        "--initial-points", type=int, default=8192,
        help="Points per condition for both cache fits in annealing stage 0.",
    )
    parser.add_argument("--batch-size", type=int, default=16_384)
    parser.add_argument(
        "--marginal-steps", type=int, default=3000,
        help="Maximum marginal steps per temperature; convergence may stop earlier.",
    )
    parser.add_argument(
        "--marginal-resamples",
        type=int,
        default=1024,
        help=(
            "Systematic weighted draws per condition for marginal MLE. "
            "Use 0 to keep the full points-per-condition count."
        ),
    )
    parser.add_argument(
        "--initial-marginal-resamples",
        type=int,
        default=2048,
        help="Marginal resamples per condition in annealing stage 0.",
    )
    parser.add_argument(
        "--copula-steps", type=int, default=6000,
        help="Maximum bridge-copula steps per temperature; convergence may stop earlier.",
    )
    parser.add_argument(
        "--stage-cache-refreshes",
        type=int,
        default=2,
        help=(
            "Fresh importance-cache fitting passes at each temperature. "
            "Step budgets are divided across passes."
        ),
    )
    parser.add_argument(
        "--formal-copula-rounds",
        type=int,
        default=3,
        help="Full-target beta=1 proposal-refresh rounds after annealing.",
    )
    parser.add_argument(
        "--formal-copula-steps",
        type=int,
        default=14000,
        help=(
            "Maximum copula-only steps in each formal beta=1 round; "
            "convergence may stop earlier."
        ),
    )
    parser.add_argument("--chi2-rounds", type=int, default=2)
    parser.add_argument(
        "--chi2-steps", type=int, default=3000,
        help="Maximum ESS-directed copula steps per fresh-cache round.",
    )
    parser.add_argument(
        "--marginal-check-interval",
        type=int,
        default=50,
        help="Validation interval for the shorter marginal LR ladder.",
    )
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument(
        "--patience", type=int, default=3,
        help="Validation checks without improvement before reducing LR or stopping.",
    )
    parser.add_argument(
        "--marginal-lr",
        type=float,
        default=None,
        help="Override the marginal LR ladder with one fixed learning rate.",
    )
    parser.add_argument(
        "--copula-lr",
        type=float,
        default=None,
        help="Override both copula LR ladders with one fixed learning rate.",
    )
    parser.add_argument("--uniform-fraction", type=float, default=0.10)
    parser.add_argument("--min-uniform-fraction", type=float, default=0.02)
    parser.add_argument("--eval-conditions", type=int, default=256)
    parser.add_argument("--eval-points", type=int, default=8192)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/simple_separate_32"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.max_annealing_stages = 30
        args.temperature_probe_conditions = 32
        args.temperature_probe_points = 128
        args.temperature_median_ess = 16.0
        args.temperature_q05_ess = 3.0
        args.temperature_hard_median = 5.0
        args.temperature_hard_q05 = 1.25
        args.temperature_min_increment = 0.01
        args.temperature_max_increment = 0.30
        args.train_conditions = 32
        args.val_conditions = 64
        args.points = 128
        args.initial_points = 128
        args.batch_size = 2048
        args.marginal_steps = 100
        args.marginal_resamples = 64
        args.initial_marginal_resamples = 64
        args.copula_steps = 100
        args.stage_cache_refreshes = 1
        args.formal_copula_rounds = 1
        args.formal_copula_steps = 100
        args.chi2_rounds = 1
        args.chi2_steps = 100
        args.marginal_check_interval = 50
        args.check_interval = 50
        args.patience = 1
        args.eval_conditions = 32
        args.eval_points = 512
        args.output = Path("results/simple_separate_32_smoke")
    if not (0.0 < args.min_uniform_fraction <= args.uniform_fraction < 1.0):
        parser.error(
            "require 0 < --min-uniform-fraction <= --uniform-fraction < 1"
        )
    if args.stage_cache_refreshes < 1:
        parser.error("--stage-cache-refreshes must be at least 1")
    return args


def main():
    args = parse_args()
    if IMPORTED_DIM != 8:
        # The imported module's global DIM is expected to be 8, but its classes
        # accept the requested dimension explicitly; document this intentionally.
        print(f"Imported module default DIM={IMPORTED_DIM}; constructing DIM={DIM} explicitly.")
    device = torch.device(args.device)
    seed_everything(args.seed)
    parents = full_ar_parents(DIM) if args.graph == "full_ar" else markov1_parents(DIM)
    bank = MarginalBank(DIM).to(device)
    model = MarginalCopulaModel(bank, StructuredCopula(DIM, parents), DIM).to(device)
    initial_marginal_lr_levels = (
        (args.marginal_lr,)
        if args.marginal_lr is not None else BRIDGE_COPULA_LR_LEVELS
    )
    warm_marginal_lr_levels = (
        (args.marginal_lr,)
        if args.marginal_lr is not None else WARM_START_LR_LEVELS
    )
    initial_bridge_lr_levels = (
        (args.copula_lr,)
        if args.copula_lr is not None else BRIDGE_COPULA_LR_LEVELS
    )
    warm_bridge_lr_levels = (
        (args.copula_lr,)
        if args.copula_lr is not None else WARM_START_LR_LEVELS
    )
    formal_lr_levels = (
        (args.copula_lr,)
        if args.copula_lr is not None else FORMAL_COPULA_LR_LEVELS
    )
    proposal = None
    history = []
    started = time.perf_counter()

    print("=" * 96)
    print("SIMPLE INTEGRAND-ONLY MARGINAL THEN COPULA")
    print(f"device={device} dim={DIM} graph={args.graph}")
    print("target sampler=NEVER USED")
    print(
        "temperatures=adaptive "
        f"(median raw ESS {args.temperature_median_ess:g}, "
        f"q05 {args.temperature_q05_ess:g})"
    )
    print(
        f"defensive uniform={args.uniform_fraction:.3f} -> "
        f"{args.min_uniform_fraction:.3f}"
    )
    print(
        f"cache sampling={args.train_conditions} train / "
        f"{args.val_conditions} val conditions x {args.points:,} points; "
        f"stage 0 x {args.initial_points:,}"
    )
    print(
        f"temperature probe={args.temperature_probe_conditions} conditions x "
        f"{args.temperature_probe_points:,} points; "
        f"final evaluation={args.eval_conditions} x {args.eval_points:,}"
    )
    print(
        f"annealed maximums=marginal {args.marginal_steps} steps, "
        f"copula {args.copula_steps} steps, "
        f"{args.stage_cache_refreshes} fresh caches, patience {args.patience}"
    )
    print(
        f"formal copula={args.formal_copula_rounds} rounds x up to "
        f"{args.formal_copula_steps} steps at beta=1"
    )
    print(
        "stage-0 marginal/copula LR="
        + " -> ".join(f"{lr:.0e}" for lr in initial_marginal_lr_levels)
    )
    print(
        "warm-start marginal/copula LR="
        + " -> ".join(f"{lr:.0e}" for lr in warm_marginal_lr_levels)
    )
    print(
        "formal LR ladder=" + " -> ".join(f"{lr:.0e}" for lr in formal_lr_levels)
    )
    print(
        f"chi2 fine-tuning={args.chi2_rounds} rounds x up to "
        f"{args.chi2_steps} steps"
    )
    print("=" * 96)

    stage = 0
    current_beta = 0.0
    while current_beta < 1.0 - 1.0e-12:
        if stage >= args.max_annealing_stages:
            raise RuntimeError(
                f"Annealing did not reach beta=1 after {stage} stages. "
                "Increase --max-annealing-stages."
            )
        beta, stage_uniform_fraction = select_next_beta(
            proposal,
            current_beta,
            args,
            device,
            args.seed + 1000 * stage + 3,
        )
        print(f"\nSTAGE {stage}: beta={beta:.4f}", flush=True)
        marginal_steps = math.ceil(
            args.marginal_steps / args.stage_cache_refreshes
        )
        copula_steps = math.ceil(
            args.copula_steps / args.stage_cache_refreshes
        )
        for refresh in range(args.stage_cache_refreshes):
            cold_start = stage == 0 and refresh == 0
            cache_points = args.initial_points if stage == 0 else args.points
            marginal_resamples = (
                args.initial_marginal_resamples
                if stage == 0 else args.marginal_resamples
            )
            marginal_lr_levels = (
                initial_marginal_lr_levels if cold_start
                else warm_marginal_lr_levels
            )
            bridge_lr_levels = (
                initial_bridge_lr_levels if cold_start
                else warm_bridge_lr_levels
            )
            print(
                f"  CACHE FIT {refresh + 1}/{args.stage_cache_refreshes} "
                f"(marginal <= {marginal_steps}, copula <= {copula_steps}, "
                f"start LR={marginal_lr_levels[0]:.1e}, "
                f"points/c={cache_points:,})",
                flush=True,
            )
            refresh_seed = args.seed + 10_000 * stage + 100 * refresh
            train = build_cache(
                proposal, beta, args.train_conditions, cache_points, device,
                refresh_seed + 11, stage_uniform_fraction,
            )
            val = build_cache(
                proposal, beta, args.val_conditions, cache_points, device,
                refresh_seed + 29, stage_uniform_fraction,
            )
            print(
                f"train raw ESS median={train['raw_ess'].median().item():.2f}; "
                f"val={val['raw_ess'].median().item():.2f}", flush=True
            )
            train_marginals(
                model, train, val, marginal_steps, marginal_lr_levels,
                args.marginal_check_interval, args.patience, args.batch_size,
                marginal_resamples, refresh_seed + 71,
            )
            train_u = cache_copula_coordinates(model, train)
            val_u = cache_copula_coordinates(model, val)
            train_copula(
                model, train_u, val_u, copula_steps, bridge_lr_levels,
                args.check_interval, args.patience, args.batch_size,
            )
            proposal = copy.deepcopy(model).to(device).eval()
            for parameter in proposal.parameters():
                parameter.requires_grad_(False)
        history.append({
            "stage": stage,
            "beta": beta,
            "phase": "annealed_marginal_copula",
            "train_median_raw_ess": float(train["raw_ess"].median().item()),
            "val_median_raw_ess": float(val["raw_ess"].median().item()),
        })
        current_beta = beta
        stage += 1

    final_uniform_fraction = tune_defensive_fraction(
        model, args, device, args.seed + 880_000,
    )
    estimates, esses = evaluate(
        model, device, args.eval_conditions, args.eval_points,
        args.seed + 888_888, final_uniform_fraction,
    )
    print("\nANNEALED BRIDGE PERFORMANCE (BEFORE FORMAL COPULA)")
    print(f"mean ESS             = {esses.mean():.6f}")
    print(f"median ESS           = {np.median(esses):.6f}")
    print(f"mean estimate        = {estimates.mean():.8f}")

    # Dedicated beta=1 copula refinement. Marginals remain frozen, and every
    # round receives a fresh importance cache from the preceding proposal.
    for formal_round in range(args.formal_copula_rounds):
        print(
            f"\nFORMAL COPULA ROUND {formal_round + 1}/"
            f"{args.formal_copula_rounds} (beta=1)",
            flush=True,
        )
        formal_train = build_cache(
            proposal, 1.0, args.train_conditions, args.points, device,
            args.seed + 100_000 + 1000 * formal_round + 11,
            final_uniform_fraction,
        )
        formal_val = build_cache(
            proposal, 1.0, args.val_conditions, args.points, device,
            args.seed + 100_000 + 1000 * formal_round + 29,
            final_uniform_fraction,
        )
        print(
            f"formal train raw ESS median="
            f"{formal_train['raw_ess'].median().item():.2f}; "
            f"val={formal_val['raw_ess'].median().item():.2f}",
            flush=True,
        )
        formal_train_u = cache_copula_coordinates(model, formal_train)
        formal_val_u = cache_copula_coordinates(model, formal_val)
        train_copula(
            model,
            formal_train_u,
            formal_val_u,
            args.formal_copula_steps,
            formal_lr_levels,
            args.check_interval,
            args.patience,
            args.batch_size,
        )
        proposal = copy.deepcopy(model).to(device).eval()
        for parameter in proposal.parameters():
            parameter.requires_grad_(False)
        round_estimates, round_esses = evaluate(
            model,
            device,
            min(args.eval_conditions, 64),
            args.eval_points,
            args.seed + 200_000 + formal_round,
            final_uniform_fraction,
        )
        print(
            f"formal round {formal_round + 1}: mean ESS="
            f"{round_esses.mean():.6f}, estimate={round_estimates.mean():.8f}",
            flush=True,
        )
        history.append({
            "stage": stage + formal_round,
            "beta": 1.0,
            "phase": "formal_copula",
            "train_median_raw_ess": float(
                formal_train["raw_ess"].median().item()
            ),
            "val_median_raw_ess": float(
                formal_val["raw_ess"].median().item()
            ),
        })

    final_uniform_fraction = tune_defensive_fraction(
        model, args, device, args.seed + 890_000,
    )

    # Pearson chi-square directly targets the second moment of importance
    # weights and therefore ESS. Accept each round only on a common-random-
    # numbers evaluation, so noisy fine-tuning cannot degrade the final model.
    for chi2_round in range(args.chi2_rounds):
        print(f"\nCHI2 COPULA ROUND {chi2_round + 1}/{args.chi2_rounds}", flush=True)
        comparison_seed = args.seed + 700_000 + chi2_round
        _, baseline_ess, baseline_max_weight = evaluate(
            model, device, min(args.eval_conditions, 128), args.eval_points,
            comparison_seed, final_uniform_fraction, return_max_weight=True,
        )
        before_state = cpu_state(model.copula)
        chi2_train = build_cache(
            proposal, 1.0, args.train_conditions, args.points, device,
            args.seed + 500_000 + 1000 * chi2_round + 11,
            final_uniform_fraction,
        )
        chi2_val = build_cache(
            proposal, 1.0, args.val_conditions, args.points, device,
            args.seed + 500_000 + 1000 * chi2_round + 29,
            final_uniform_fraction,
        )
        train_copula_chi2(
            model, chi2_train, chi2_val, args.chi2_steps, CHI2_LR_LEVELS,
            args.check_interval, args.patience, args.batch_size,
        )
        candidate_estimate, candidate_ess, candidate_max_weight = evaluate(
            model, device, min(args.eval_conditions, 128), args.eval_points,
            comparison_seed, final_uniform_fraction, return_max_weight=True,
        )
        baseline_mean = float(baseline_ess.mean())
        candidate_mean = float(candidate_ess.mean())
        baseline_q05 = float(np.quantile(baseline_ess, 0.05))
        candidate_q05 = float(np.quantile(candidate_ess, 0.05))
        baseline_w95 = float(np.quantile(baseline_max_weight, 0.95))
        candidate_w95 = float(np.quantile(candidate_max_weight, 0.95))
        baseline_score = baseline_mean * math.sqrt(max(baseline_q05, 1.0e-12))
        candidate_score = candidate_mean * math.sqrt(max(candidate_q05, 1.0e-12))
        accepted = (
            candidate_mean >= 0.98 * baseline_mean
            and candidate_q05 >= 0.98 * baseline_q05
            and candidate_w95 <= 1.02 * baseline_w95
            and candidate_score > baseline_score
        )
        if accepted:
            proposal = copy.deepcopy(model).to(device).eval()
            for parameter in proposal.parameters():
                parameter.requires_grad_(False)
        else:
            load_state(model.copula, before_state)
        print(
            f"chi2 round {chi2_round + 1}: mean ESS "
            f"{baseline_mean:.6f} -> {candidate_mean:.6f}; "
            f"q05 ESS {baseline_q05:.6f} -> {candidate_q05:.6f}; "
            f"q95 max-weight {baseline_w95:.6f} -> {candidate_w95:.6f}; "
            f"estimate={candidate_estimate.mean():.8f}; "
            f"{'ACCEPT' if accepted else 'REJECT'}",
            flush=True,
        )
        history.append({
            "stage": stage + args.formal_copula_rounds + chi2_round,
            "beta": 1.0,
            "phase": "chi2_copula_accept" if accepted else "chi2_copula_reject",
            "train_median_raw_ess": float(chi2_train["raw_ess"].median().item()),
            "val_median_raw_ess": float(chi2_val["raw_ess"].median().item()),
        })

    final_uniform_fraction = tune_defensive_fraction(
        model, args, device, args.seed + 900_000,
    )

    estimates, esses, max_weights = evaluate(
        model, device, args.eval_conditions, args.eval_points,
        args.seed + 999_999, final_uniform_fraction,
        return_max_weight=True,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(args.output / "training_stages.csv", index=False)
    pd.DataFrame({
        "estimate": estimates,
        "ess": esses,
        "max_normalized_weight": max_weights,
    }).to_csv(
        args.output / "final_conditions.csv", index=False
    )
    torch.save({
        "model_state_dict": cpu_state(model),
        "graph": args.graph,
        "temperatures": [
            float(row["beta"])
            for row in history
            if row["phase"] == "annealed_marginal_copula"
        ],
        "formal_copula_rounds": args.formal_copula_rounds,
        "dimension": DIM,
        "final_uniform_fraction": final_uniform_fraction,
    }, args.output / "model.pt")
    print("\nFINAL")
    print(f"mean ESS             = {esses.mean():.6f}")
    print(f"median ESS           = {np.median(esses):.6f}")
    print(f"q05 ESS              = {np.quantile(esses, 0.05):.6f}")
    print(f"q95 maximum weight   = {np.quantile(max_weights, 0.95):.6f}")
    print(f"mean estimate        = {estimates.mean():.8f}")
    print(f"mean absolute error  = {np.abs(estimates - TRUE_INTEGRAL).mean():.6g}")
    print(f"wall seconds         = {time.perf_counter() - started:.1f}")
    print(f"output               = {args.output.resolve()}")


if __name__ == "__main__":
    main()
