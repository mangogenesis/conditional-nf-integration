"""Bomb-rejuvenated, two-pass conditional mixture training for irregular 100D.

This experiment combines three ideas without changing the original trainers:

1. build annealing *training* caches with resampling plus MALA rejuvenation;
2. assign cache points softly to conditional mixture specialists;
3. train each specialist's marginals first, then its copula, and finally train
   the gate to the estimated conditional basin masses.

Diagnostic and residual-repair caches deliberately use ordinary importance
sampling.  A MALA endpoint is useful training data, but its proposal density is
not available and therefore it must not be used as an importance sample.
"""

from __future__ import annotations

import argparse
import copy
import sys
from contextlib import contextmanager

import torch

import run_adaptive_mixture_irregular_100d as mixture
import run_bomb_bridge_irregular_100d as bomb


adaptive = mixture.adaptive


def parse_wrapper_arguments(argv):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--bomb-mala-steps", type=int, default=6)
    parser.add_argument("--bomb-step-size", type=float, default=0.01)
    parser.add_argument("--bomb-gradient-clip", type=float, default=50.0)
    parser.add_argument("--bomb-chunk", type=int, default=1024)
    parser.add_argument("--specialist-em-rounds", type=int, default=1)
    parser.add_argument("--specialist-marginal-steps", type=int, default=600)
    parser.add_argument("--specialist-copula-steps", type=int, default=600)
    parser.add_argument("--specialist-gate-steps", type=int, default=600)
    parser.add_argument("--specialist-check-interval", type=int, default=100)
    parser.add_argument("--responsibility-temperature", type=float, default=1.0)
    parser.add_argument("--responsibility-floor", type=float, default=1.0e-4)
    parser.add_argument("--diversity-steps", type=int, default=0)
    parser.add_argument("--diversity-learning-rate", type=float, default=1.0e-5)
    parser.add_argument("--repulsion-weight", type=float, default=0.05)
    parser.add_argument("--usage-balance-weight", type=float, default=0.02)
    parser.add_argument("--diversity-nll-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-overlap-improvement", type=float, default=0.01)
    args, remaining = parser.parse_known_args(argv)
    return args, remaining


WRAPPER_ARGS = None


def training_cache_dispatch(*args, **kwargs):
    """Use bombs only where samples are consumed as weighted training data."""
    label = str(kwargs.get("label", ""))
    if label.startswith("mixture train") or label.startswith("mixture val"):
        return bomb.bomb_rejuvenated_cache(*args, **kwargs)
    return bomb.ORIGINAL_BUILD_WEIGHTED_CACHE(*args, **kwargs)


@contextmanager
def specialist_budgets():
    names = {
        "MAX_MARGINAL_STEPS": WRAPPER_ARGS.specialist_marginal_steps,
        "MIN_MARGINAL_STEPS": min(
            getattr(adaptive, "MIN_MARGINAL_STEPS", 300),
            WRAPPER_ARGS.specialist_marginal_steps,
        ),
        "MARGINAL_CHECK_INTERVAL": WRAPPER_ARGS.specialist_check_interval,
        "BRIDGE_COPULA_MAX_STEPS": WRAPPER_ARGS.specialist_copula_steps,
        "BRIDGE_COPULA_CHECK_INTERVAL": WRAPPER_ARGS.specialist_check_interval,
    }
    previous = {name: getattr(adaptive, name) for name in names}
    try:
        for name, value in names.items():
            setattr(adaptive, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(adaptive, name, value)


@torch.no_grad()
def responsibilities(model, cache, chunk=8192):
    x = cache["x"].reshape(-1, cache["x"].shape[-1])
    cond = cache["cond"].reshape(-1, cache["cond"].shape[-1])
    pieces = []
    temperature = max(WRAPPER_ARGS.responsibility_temperature, 1.0e-3)
    for start in range(0, len(x), chunk):
        stop = min(start + chunk, len(x))
        log_joint = (
            model.log_gate(cond[start:stop])
            + model.component_log_probs(x[start:stop], cond[start:stop])
        )
        pieces.append(torch.softmax(log_joint / temperature, dim=-1))
    result = torch.cat(pieces, dim=0)
    floor = max(0.0, min(WRAPPER_ARGS.responsibility_floor, 0.99))
    if floor:
        result = (1.0 - floor) * result + floor / result.shape[-1]
    return result.reshape(*cache["alpha"].shape, -1)


def basin_masses(cache, resp):
    mass = (cache["alpha"].unsqueeze(-1) * resp).sum(dim=1)
    return mass / mass.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)


def component_cache(cache, resp, component):
    alpha = cache["alpha"] * resp[..., component]
    alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp_min(1.0e-30)
    result = dict(cache)
    result["alpha"] = alpha.detach()
    return result


def diversity_objective(model, cache, condition_indices=None):
    """Fit plus distribution-level component repulsion on target particles.

    Responsibilities are evaluated on the weighted SMC cache.  Their pairwise
    cosine overlap is one when two components explain the same particles and
    approaches zero as they specialize to different target regions.  A KL
    penalty on mean component usage temporarily prevents the gate from making
    the repulsion vacuous by switching all but one component off.
    """
    if condition_indices is None:
        condition_indices = torch.arange(
            cache["x"].shape[0], device=cache["x"].device
        )
    x = cache["x"][condition_indices]
    cond = cache["cond"][condition_indices]
    alpha = cache["alpha"][condition_indices]
    b, p, d = x.shape
    log_joint = (
        model.log_gate(cond[:, 0])[:, None, :]
        + model.component_log_probs(
            x.reshape(b * p, d), cond.reshape(b * p, -1)
        ).reshape(b, p, model.n_components)
    )
    log_mixture = torch.logsumexp(log_joint, dim=-1)
    fit = (-(alpha * log_mixture).sum(dim=1)).mean()

    responsibility = torch.softmax(log_joint, dim=-1)
    vectors = alpha.sqrt().unsqueeze(-1) * responsibility
    gram = torch.einsum("bpk,bpl->bkl", vectors, vectors)
    norm = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1.0e-20).sqrt()
    cosine = gram / (norm.unsqueeze(-1) * norm.unsqueeze(-2)).clamp_min(1.0e-20)
    if model.n_components > 1:
        mask = torch.triu(
            torch.ones(
                model.n_components,
                model.n_components,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        overlap = cosine[:, mask].mean()
    else:
        overlap = fit.new_zeros(())

    usage = (alpha.unsqueeze(-1) * responsibility).sum(dim=1).mean(dim=0)
    usage = usage / usage.sum().clamp_min(1.0e-20)
    balance = (
        usage.clamp_min(1.0e-20)
        * torch.log(usage.clamp_min(1.0e-20) * model.n_components)
    ).sum()
    return fit, overlap, balance, usage


@torch.no_grad()
def evaluate_diversity(model, cache, condition_chunk=2):
    totals = torch.zeros(3, dtype=torch.float64)
    usage = torch.zeros(model.n_components, dtype=torch.float64)
    count = 0
    for start in range(0, cache["x"].shape[0], condition_chunk):
        stop = min(start + condition_chunk, cache["x"].shape[0])
        index = torch.arange(start, stop, device=cache["x"].device)
        fit, overlap, balance, block_usage = diversity_objective(
            model, cache, index
        )
        size = stop - start
        totals += size * torch.tensor(
            [float(fit), float(overlap), float(balance)], dtype=torch.float64
        )
        usage += size * block_usage.detach().double().cpu()
        count += size
    return {
        "nll": float(totals[0] / count),
        "overlap": float(totals[1] / count),
        "balance": float(totals[2] / count),
        "usage": (usage / count).tolist(),
    }


def train_diversity_refinement(model, train_cache, validation_cache):
    """Jointly separate mixture components, accepting only held-out gains."""
    steps = int(WRAPPER_ARGS.diversity_steps)
    if steps <= 0 or model.n_components <= 1:
        return {"enabled": False, "accepted": False}

    baseline_state = adaptive.cpu_state_dict(model)
    baseline = evaluate_diversity(model, validation_cache)
    best_state = None
    best = None
    optimizer = torch.optim.Adam(
        model.parameters(), lr=WRAPPER_ARGS.diversity_learning_rate
    )
    points = train_cache["x"].shape[1]
    condition_batch = max(1, min(4, 8192 // points))
    check = max(1, int(WRAPPER_ARGS.specialist_check_interval))

    for step in range(1, steps + 1):
        index = torch.randint(
            0,
            train_cache["x"].shape[0],
            (min(condition_batch, train_cache["x"].shape[0]),),
            device=train_cache["x"].device,
        )
        fit, overlap, balance, _ = diversity_objective(
            model, train_cache, index
        )
        # Balance is a discovery aid, not a claim that every available slot
        # must remain active.  Anneal it to zero so redundant components may
        # lose gate mass once the components have separated.
        usage_weight = WRAPPER_ARGS.usage_balance_weight * (1.0 - step / steps)
        loss = (
            fit
            + WRAPPER_ARGS.repulsion_weight * overlap
            + usage_weight * balance
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

        if step == 1 or step % check == 0 or step == steps:
            current = evaluate_diversity(model, validation_cache)
            overlap_gain = baseline["overlap"] - current["overlap"]
            fit_safe = (
                current["nll"]
                <= baseline["nll"] + WRAPPER_ARGS.diversity_nll_tolerance
            )
            improved = (
                overlap_gain >= WRAPPER_ARGS.minimum_overlap_improvement
            )
            score = (
                current["nll"]
                + WRAPPER_ARGS.repulsion_weight * current["overlap"]
                + usage_weight * current["balance"]
            )
            if fit_safe and improved and (
                best is None or score < best["score"]
            ):
                best = dict(current, score=score, step=step)
                best_state = adaptive.cpu_state_dict(model)
            usage_text = ", ".join(
                f"{value:.3f}" for value in current["usage"]
            )
            print(
                f"diversity step {step:4d}/{steps}: "
                f"NLL={current['nll']:.6f} "
                f"overlap={current['overlap']:.4f} "
                f"balance={current['balance']:.4f} "
                f"usage-weight={usage_weight:.4g} "
                f"usage=[{usage_text}]"
            )

    accepted = best_state is not None
    adaptive.load_cpu_state(model, best_state if accepted else baseline_state)
    final = best if accepted else baseline
    print(
        f"DIVERSITY REFINEMENT {'ACCEPTED' if accepted else 'REJECTED'}: "
        f"overlap {baseline['overlap']:.4f} -> {final['overlap']:.4f}; "
        f"NLL {baseline['nll']:.6f} -> {final['nll']:.6f}"
    )
    return {
        "enabled": True,
        "accepted": accepted,
        "baseline": baseline,
        "final": final,
    }


def train_gate(model, train_cache, val_cache, train_mass, val_mass):
    train_cond = train_cache["cond"][:, 0, :]
    val_cond = val_cache["cond"][:, 0, :]
    optimizer = torch.optim.Adam(model.gate.parameters(), lr=3.0e-4)
    best = copy.deepcopy(model.gate.state_dict())
    best_loss = float("inf")
    patience = 0
    last_step = 0

    for step in range(1, WRAPPER_ARGS.specialist_gate_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        loss = -(train_mass * model.log_gate(train_cond)).sum(dim=-1).mean()
        loss.backward()
        optimizer.step()
        last_step = step

        if step == 1 or step % WRAPPER_ARGS.specialist_check_interval == 0:
            with torch.no_grad():
                val_loss = -(
                    val_mass * model.log_gate(val_cond)
                ).sum(dim=-1).mean()
            value = float(val_loss.item())
            print(
                f"gate step {step:4d}/{WRAPPER_ARGS.specialist_gate_steps}: "
                f"train={float(loss.item()):.6f} val={value:.6f}"
            )
            if value < best_loss - 1.0e-5:
                best_loss = value
                best = copy.deepcopy(model.gate.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= 3:
                    break

    model.gate.load_state_dict(best)
    return {"steps": last_step, "val_cross_entropy": best_loss}


def train_clustered_two_pass(
    model,
    train_cache,
    validation_cache,
    steps,
    check_interval,
    learning_rate,
):
    """Generalized-EM fit with explicit marginal/copula/gate separation."""
    del steps, check_interval, learning_rate
    baseline_state = adaptive.cpu_state_dict(model)
    baseline_nll = float(mixture.weighted_nll(model, validation_cache))
    best_state = baseline_state
    best_nll = baseline_nll
    history = []
    print(
        f"CLUSTERED TWO-PASS FIT: K={model.n_components} "
        f"baseline val NLL={baseline_nll:.6f}"
    )

    with specialist_budgets():
        for em_round in range(1, WRAPPER_ARGS.specialist_em_rounds + 1):
            train_resp = responsibilities(model, train_cache)
            val_resp = responsibilities(model, validation_cache)
            train_mass = basin_masses(train_cache, train_resp).detach()
            val_mass = basin_masses(validation_cache, val_resp).detach()

            mean_mass = train_mass.mean(dim=0)
            print(
                f"EM round {em_round}: mean basin masses="
                + ", ".join(f"{v:.4f}" for v in mean_mass.tolist())
            )

            specialist_rows = []
            for index, component in enumerate(model.components):
                train_part = component_cache(train_cache, train_resp, index)
                val_part = component_cache(validation_cache, val_resp, index)
                before = float(mixture.weighted_nll(component, val_part))
                print(
                    f"SPECIALIST {index + 1}/{model.n_components}: "
                    f"mass={float(mean_mass[index]):.4f} before={before:.6f}"
                )

                bank = adaptive.get_marginal_bank(component)
                marginal_result = adaptive.train_marginals_on_dataset(
                    bank,
                    train_part,
                    val_part,
                    train_part["x"].device,
                    10000 * em_round + index,
                )
                copula_result = adaptive.train_bridge_copula_on_cache(
                    component,
                    train_part,
                    val_part,
                    label=f"specialist {index} copula",
                    optimize_joint=False,
                )
                after = float(mixture.weighted_nll(component, val_part))
                specialist_rows.append(
                    {
                        "component": index,
                        "mass": float(mean_mass[index]),
                        "before": before,
                        "after": after,
                        "marginal": marginal_result,
                        "copula": copula_result,
                    }
                )
                print(f"SPECIALIST {index}: after={after:.6f}")

            gate_result = train_gate(
                model,
                train_cache,
                validation_cache,
                train_mass,
                val_mass,
            )
            val_nll = float(mixture.weighted_nll(model, validation_cache))
            print(
                f"EM round {em_round} complete: mixture val NLL={val_nll:.6f}"
            )
            history.append(
                {
                    "em_round": em_round,
                    "val_nll": val_nll,
                    "gate": gate_result,
                    "specialists": specialist_rows,
                }
            )
            if val_nll < best_nll:
                best_nll = val_nll
                best_state = adaptive.cpu_state_dict(model)

    adaptive.load_cpu_state(model, best_state)
    if best_nll >= baseline_nll:
        print("CLUSTERED FIT REJECTED: held-out mixture NLL did not improve")
    else:
        print(
            f"CLUSTERED FIT ACCEPTED: held-out mixture NLL "
            f"{baseline_nll:.6f} -> {best_nll:.6f}"
        )
    diversity_result = train_diversity_refinement(
        model, train_cache, validation_cache
    )
    final_nll = float(mixture.weighted_nll(model, validation_cache))
    return {
        "best_step": WRAPPER_ARGS.specialist_em_rounds,
        "val_nll": final_nll,
        "history": history,
        "diversity": diversity_result,
    }


def main():
    global WRAPPER_ARGS
    WRAPPER_ARGS, remaining = parse_wrapper_arguments(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    bomb.BOMB_MALA_STEPS = WRAPPER_ARGS.bomb_mala_steps
    bomb.BOMB_INITIAL_STEP_SIZE = WRAPPER_ARGS.bomb_step_size
    bomb.BOMB_GRADIENT_CLIP = WRAPPER_ARGS.bomb_gradient_clip
    bomb.BOMB_CHUNK = WRAPPER_ARGS.bomb_chunk

    adaptive.build_weighted_cache = training_cache_dispatch
    mixture.train_mixture_on_cache = train_clustered_two_pass

    print(
        "Clustered bomb mixture enabled: bomb-rejuvenated training caches; "
        "soft basin specialists; marginal -> copula -> mass-gate fitting"
    )
    mixture.main()


if __name__ == "__main__":
    main()
