#!/usr/bin/env python3
"""Blind adaptive mixture of independently advancing annealing bridges.

The algorithm deliberately receives neither the number nor the labels of the
four benchmark structures.  It starts with two full-autoregressive flows.  A
component owns its own annealing temperature and may advance, hold, or freeze.
When at least two distinct components advance in the same round, a child is
born from the largest importance residuals of the current mixture.  A held-out
responsibility-overlap penalty discourages duplicate components.

No exact target samples and no value of the target integral are used in
training.  The benchmark's structure labels remain available only to the
separate audit program.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_heterogeneous_topology_32d as audit  # noqa: E402
import run_heterogeneous_topology_32d as heterogeneous  # noqa: E402
from nf.conditional_mixture import ConditionalMixtureProposal  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resume-mixture", type=Path)
    parser.add_argument(
        "--reactivate-frozen", action="store_true",
        help=(
            "when resuming an older run, give frozen components another "
            "refresh-and-advance opportunity"
        ),
    )
    parser.add_argument(
        "--fixed-condition", type=float, nargs=3, default=(0.5, 0.5, 0.5)
    )
    parser.add_argument("--initial-components", type=int, default=2)
    parser.add_argument("--maximum-components", type=int, default=8)
    parser.add_argument("--maximum-rounds", type=int, default=24)
    parser.add_argument("--stall-patience", type=int, default=2)
    parser.add_argument("--train-conditions", type=int, default=48)
    parser.add_argument("--validation-conditions", type=int, default=24)
    parser.add_argument("--points-per-condition", type=int, default=8192)
    parser.add_argument("--marginal-steps", type=int, default=12000)
    parser.add_argument("--copula-steps", type=int, default=12000)
    parser.add_argument("--check-interval", type=int, default=200)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--diversity-steps", type=int, default=800)
    parser.add_argument("--diversity-learning-rate", type=float, default=2e-5)
    parser.add_argument("--repulsion-weight", type=float, default=0.25)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--maximum-overlap", type=float, default=0.92)
    parser.add_argument("--fit-tolerance", type=float, default=0.02)
    parser.add_argument("--residual-fraction", type=float, default=0.10)
    parser.add_argument("--birth-noise", type=float, default=2e-3)
    parser.add_argument("--uniform-fraction", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 2 <= args.initial_components <= args.maximum_components:
        parser.error("initial components must be between 2 and the maximum")
    if args.checkpoint is not None and args.resume_mixture is not None:
        parser.error("--checkpoint and --resume-mixture are mutually exclusive")
    counts = (
        args.maximum_rounds, args.stall_patience, args.train_conditions,
        args.validation_conditions, args.points_per_condition,
        args.marginal_steps, args.copula_steps, args.check_interval,
        args.patience,
    )
    if min(counts) <= 0:
        parser.error("all training budgets must be positive")
    if not 0 < args.residual_fraction <= 0.5:
        parser.error("--residual-fraction must lie in (0, 0.5]")
    if not 0 < args.uniform_fraction < 1:
        parser.error("--uniform-fraction must lie in (0, 1)")
    if not 0 <= args.maximum_overlap <= 1:
        parser.error("--maximum-overlap must lie in [0, 1]")
    return args


def full_autoregressive_graph(dim):
    return {child: list(range(child)) for child in range(dim)}


def cpu_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def set_equal_gate(mixture):
    with torch.no_grad():
        final = mixture.gate[-1]
        final.weight.zero_()
        final.bias.fill_(-20.0)
        final.bias[: mixture.n_components].zero_()
    for parameter in mixture.gate.parameters():
        parameter.requires_grad_(False)


def make_mixture(components, maximum_components):
    mixture = ConditionalMixtureProposal(
        components,
        cond_dim=heterogeneous.CONDITION_DIMENSION,
        max_components=maximum_components,
    )
    set_equal_gate(mixture)
    return mixture


def configure_adaptive(adaptive, args):
    adaptive.DIM = heterogeneous.DIMENSION
    adaptive.configure_speed_profile("balanced")
    adaptive.COMPILE_TRAINING_KERNELS = False
    adaptive.MAX_MARGINAL_STEPS = args.marginal_steps
    adaptive.MIN_MARGINAL_STEPS = min(2000, args.marginal_steps)
    adaptive.MARGINAL_CHECK_INTERVAL = args.check_interval
    adaptive.MARGINAL_EARLY_STOP_PATIENCE = args.patience
    adaptive.BRIDGE_COPULA_MAX_STEPS = args.copula_steps
    adaptive.BRIDGE_COPULA_CHECK_INTERVAL = args.check_interval
    adaptive.BRIDGE_COPULA_PATIENCE = args.patience


def load_or_make_template(args, device):
    graph = full_autoregressive_graph(heterogeneous.DIMENSION)
    adaptive = heterogeneous.adaptive
    if args.checkpoint is None:
        template = adaptive.make_real_model(graph, None, device)
        source = None
    else:
        adaptive, loaded = audit.load_model(args.checkpoint, device)
        graph = torch.load(
            args.checkpoint, map_location="cpu", weights_only=False
        )["graph"]
        template = loaded.to(device)
        source = str(args.checkpoint)
    return adaptive, template, graph, source


def load_resumable_mixture(path, args, device):
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if payload.get("format") != "heterogeneous_discovered_region_mixture_v1":
        raise ValueError("unsupported mixture checkpoint format")
    if payload.get("benchmark") != heterogeneous.BENCHMARK:
        raise ValueError("resume checkpoint belongs to another benchmark")
    graph = payload["graph"]
    states = payload["specialist_state_dicts"]
    if len(states) > args.maximum_components:
        raise ValueError(
            "resume checkpoint contains more components than the configured maximum"
        )
    # make_real_model reads the adaptive module's global dimension when it
    # constructs the marginal bank.  Set it before rebuilding any component;
    # otherwise a prior imported benchmark can leave DIM=8 and a valid 32D
    # state dict then appears to contain hundreds of unexpected keys.
    heterogeneous.adaptive.DIM = heterogeneous.DIMENSION
    components = []
    for state in states:
        component = heterogeneous.adaptive.make_real_model(graph, None, device)
        component.load_state_dict(state, strict=True)
        components.append(component)
    mixture = make_mixture(components, args.maximum_components).to(device)
    betas = [float(value) for value in payload.get(
        "component_betas", [0.0] * len(components)
    )]
    statuses = list(payload.get(
        "component_statuses", ["active"] * len(components)
    ))
    if len(betas) != len(components) or len(statuses) != len(components):
        raise ValueError("resume checkpoint has inconsistent component metadata")
    if args.reactivate_frozen:
        statuses = [
            "complete" if beta >= 1.0 - 1e-10 else "active"
            for beta in betas
        ]
    return (
        heterogeneous.adaptive, mixture, graph,
        payload.get("source_checkpoint"), betas, statuses,
    )


def weighted_objective(mixture, cache):
    x = cache["x"]
    cond = cache["cond"]
    alpha = cache["alpha"]
    batch, points, dim = x.shape
    component_logs = mixture.component_log_probs(
        x.reshape(batch * points, dim), cond.reshape(batch * points, -1)
    ).reshape(batch, points, mixture.n_components)
    log_joint = mixture.log_gate(cond[:, 0])[:, None, :] + component_logs
    log_mix = torch.logsumexp(log_joint, dim=-1)
    fit = (-(alpha * log_mix).sum(dim=1)).mean()

    responsibility = torch.softmax(log_joint, dim=-1)
    vectors = alpha.sqrt().unsqueeze(-1) * responsibility
    gram = torch.einsum("bpk,bpl->bkl", vectors, vectors)
    norm = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-20).sqrt()
    cosine = gram / (
        norm.unsqueeze(-1) * norm.unsqueeze(-2)
    ).clamp_min(1e-20)
    mask = torch.triu(
        torch.ones(
            mixture.n_components, mixture.n_components,
            dtype=torch.bool, device=x.device,
        ),
        diagonal=1,
    )
    overlap = cosine[:, mask].mean() if mixture.n_components > 1 else fit * 0
    maximum_overlap = (
        cosine[:, mask].mean(dim=0).max()
        if mixture.n_components > 1 else fit * 0
    )
    usage = (alpha.unsqueeze(-1) * responsibility).sum(dim=1).mean(dim=0)
    usage = usage / usage.sum().clamp_min(1e-20)
    balance = (
        usage.clamp_min(1e-20)
        * torch.log(usage.clamp_min(1e-20) * mixture.n_components)
    ).sum()
    return fit, overlap, maximum_overlap, balance, usage


@torch.no_grad()
def evaluate_diversity(mixture, cache, condition_chunk=2):
    fit_total = overlap_total = maximum = balance_total = 0.0
    usage_total = torch.zeros(mixture.n_components, dtype=torch.float64)
    count = 0
    for start in range(0, cache["x"].shape[0], condition_chunk):
        stop = min(start + condition_chunk, cache["x"].shape[0])
        block = {
            key: value[start:stop]
            if torch.is_tensor(value) and value.ndim and
            value.shape[0] == cache["x"].shape[0] else value
            for key, value in cache.items()
        }
        fit, overlap, block_maximum, balance, usage = weighted_objective(
            mixture, block
        )
        size = stop - start
        fit_total += size * float(fit)
        overlap_total += size * float(overlap)
        maximum = max(maximum, float(block_maximum))
        balance_total += size * float(balance)
        usage_total += size * usage.detach().double().cpu()
        count += size
    return {
        "nll": fit_total / count,
        "overlap": overlap_total / count,
        "maximum_overlap": maximum,
        "balance": balance_total / count,
        "usage": (usage_total / count).tolist(),
    }


def diversity_refinement(mixture, train_cache, validation_cache, args):
    if args.diversity_steps <= 0 or mixture.n_components < 2:
        return evaluate_diversity(mixture, validation_cache)
    before_state = cpu_state(mixture)
    before = evaluate_diversity(mixture, validation_cache)
    best_state = None
    best = None
    parameters = [
        parameter
        for component in mixture.components
        for parameter in component.parameters()
        if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(
        parameters, lr=args.diversity_learning_rate
    )
    condition_batch = min(2, train_cache["x"].shape[0])
    check = max(1, args.check_interval)
    for step in range(1, args.diversity_steps + 1):
        index = torch.randint(
            train_cache["x"].shape[0], (condition_batch,),
            device=train_cache["x"].device,
        )
        block = {
            key: value[index]
            if torch.is_tensor(value) and value.ndim and
            value.shape[0] == train_cache["x"].shape[0] else value
            for key, value in train_cache.items()
        }
        fit, overlap, _, balance, _ = weighted_objective(mixture, block)
        loss = (
            fit + args.repulsion_weight * overlap
            + args.balance_weight * balance
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, 10.0)
        optimizer.step()
        if step == 1 or step % check == 0 or step == args.diversity_steps:
            current = evaluate_diversity(mixture, validation_cache)
            fit_safe = current["nll"] <= before["nll"] + args.fit_tolerance
            improved = current["overlap"] < before["overlap"] - 1e-4
            score = current["nll"] + args.repulsion_weight * current["overlap"]
            if fit_safe and improved and (
                best is None or score < best["score"]
            ):
                best = dict(current, score=score, step=step)
                best_state = cpu_state(mixture)
            print(
                f"diversity step {step:4d}/{args.diversity_steps}: "
                f"NLL={current['nll']:.6f} "
                f"overlap={current['overlap']:.4f} "
                f"max-overlap={current['maximum_overlap']:.4f}",
                flush=True,
            )
    mixture.load_state_dict(best_state or before_state, strict=True)
    after = best if best is not None else before
    print(
        f"DIVERSITY {'ACCEPTED' if best is not None else 'REJECTED'}: "
        f"NLL {before['nll']:.6f}->{after['nll']:.6f}; "
        f"overlap {before['overlap']:.4f}->{after['overlap']:.4f}",
        flush=True,
    )
    return after


def residual_cache(cache, mixture, fraction):
    """Retain the largest normalized importance weights in every condition."""
    alpha = cache["alpha"]
    keep = max(32, int(math.ceil(alpha.shape[1] * fraction)))
    keep = min(keep, alpha.shape[1])
    indices = torch.topk(alpha, keep, dim=1, largest=True).indices
    gather_x = indices.unsqueeze(-1).expand(-1, -1, cache["x"].shape[-1])
    x = torch.gather(cache["x"], 1, gather_x)
    gather_c = indices.unsqueeze(-1).expand(-1, -1, cache["cond"].shape[-1])
    cond = torch.gather(cache["cond"], 1, gather_c)
    selected_alpha = torch.gather(alpha, 1, indices)
    selected_alpha = selected_alpha / selected_alpha.sum(
        dim=1, keepdim=True
    ).clamp_min(1e-30)
    with torch.no_grad():
        flat_x = x.reshape(-1, x.shape[-1])
        flat_cond = cond.reshape(-1, cond.shape[-1])
        owner = mixture.component_log_probs(flat_x, flat_cond).mean(dim=0).argmax()
    return {
        "x": x.detach(), "cond": cond.detach(),
        "alpha": selected_alpha.detach(),
    }, int(owner)


def train_component(adaptive, model, train_cache, validation_cache, label, seed):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    marginal = adaptive.train_marginals_on_dataset(
        adaptive.get_marginal_bank(model),
        train_cache, validation_cache, train_cache["x"].device, seed,
    )
    copula = adaptive.train_bridge_copula_on_cache(
        model, train_cache, validation_cache,
        label=label, optimize_joint=False,
    )
    return marginal, copula


def save_outputs(
    args, mixture, graph, source_checkpoint, betas, statuses,
    history, adaptive,
):
    args.output.mkdir(parents=True, exist_ok=True)
    template_path = args.output / "branching_bridge_template.pt"
    torch.save({
        "benchmark": heterogeneous.BENCHMARK,
        "graph": graph,
        "model_state_dict": cpu_state(mixture.components[0]),
        "training_mode": "blind independent branching annealing bridges",
    }, template_path)
    mixture_path = args.output / "heterogeneous_branching_bridge_mixture.pt"
    weights = torch.full(
        (mixture.n_components,), 1.0 / mixture.n_components,
        dtype=torch.float64,
    )
    torch.save({
        "format": "heterogeneous_discovered_region_mixture_v1",
        "benchmark": heterogeneous.BENCHMARK,
        "dimension": heterogeneous.DIMENSION,
        "template_checkpoint": str(template_path),
        "source_checkpoint": source_checkpoint,
        "graph": graph,
        "specialist_state_dicts": [
            cpu_state(component) for component in mixture.components
        ],
        "mixture_weights": weights,
        "uniform_fraction": args.uniform_fraction,
        "component_betas": list(betas),
        "component_statuses": list(statuses),
        "uses_structure_labels": False,
        "uses_target_sampler": False,
        "uses_known_integral": False,
        "discovery": "largest blind importance residuals",
    }, mixture_path)
    frame = pd.DataFrame(history)
    frame.to_csv(args.output / "branching_bridge_history.csv", index=False)
    summary = {
        "mixture": str(mixture_path),
        "components": mixture.n_components,
        "component_betas": list(betas),
        "component_statuses": list(statuses),
        "mixture_weights": weights.tolist(),
        "uses_structure_labels": False,
        "uses_target_sampler": False,
        "uses_known_integral": False,
        "rounds": history,
    }
    with (args.output / "summary.json").open("w") as stream:
        json.dump(summary, stream, indent=2)
    return mixture_path


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    heterogeneous.register_benchmark(args.fixed_condition)
    if args.resume_mixture is not None:
        (
            adaptive, mixture, graph, source_checkpoint, betas, statuses,
        ) = load_resumable_mixture(args.resume_mixture, args, device)
        print(
            f"RESUMING: {args.resume_mixture} with K={mixture.n_components} "
            f"betas={betas} statuses={statuses}",
            flush=True,
        )
    else:
        adaptive, template, graph, source_checkpoint = load_or_make_template(
            args, device
        )
        components = []
        for index in range(args.initial_components):
            torch.manual_seed(args.seed + 1009 * index)
            if args.checkpoint is None:
                # Truly independent initializations are important here:
                # copying a single random model would leave only a tiny
                # perturbation to break the initial symmetry.
                component = adaptive.make_real_model(graph, None, device)
            else:
                component = copy.deepcopy(template).to(device)
            if index and args.checkpoint is not None:
                with torch.no_grad():
                    for parameter in component.parameters():
                        if parameter.is_floating_point():
                            parameter.add_(
                                args.birth_noise * torch.randn_like(parameter)
                            )
            components.append(component)
        mixture = make_mixture(components, args.maximum_components).to(device)
        betas = [0.0] * mixture.n_components
        statuses = ["active"] * mixture.n_components
    configure_adaptive(adaptive, args)
    stalls = [0] * mixture.n_components
    history = []

    for round_index in range(args.maximum_rounds):
        print("\n" + "#" * 96, flush=True)
        print(
            f"BRANCHING ROUND {round_index}: K={mixture.n_components} "
            f"betas={[round(value, 6) for value in betas]}",
            flush=True,
        )
        advanced = []
        refreshed = []
        for index, component in enumerate(mixture.components):
            if statuses[index] != "active":
                continue
            if betas[index] >= 1.0 - 1e-10:
                statuses[index] = "complete"
                continue
            try:
                next_beta, uniform, did_advance = adaptive.select_next_temperature(
                    heterogeneous.BENCHMARK, component, betas[index], device,
                    args.seed + 100000 * round_index + 1000 * index,
                    allow_hold=True,
                )
            except RuntimeError as error:
                print(f"component {index} bridge hold: {error}", flush=True)
                did_advance = False
                next_beta = betas[index]
                uniform = adaptive.annealing_uniform_fraction(betas[index])
            if not did_advance or next_beta <= betas[index] + 1e-8:
                stalls[index] += 1
                # A hold is useful only if the flow is actually improved at
                # its present bridge law.  The previous implementation merely
                # printed "refreshing" and then skipped this training step.
                refresh_train = adaptive.build_weighted_cache(
                    heterogeneous.BENCHMARK, component,
                    args.train_conditions, args.points_per_condition,
                    device,
                    args.seed + 800000 * round_index + 2000 * index,
                    f"component {index} refresh train beta={betas[index]:.6f}",
                    target_temperature=betas[index],
                    uniform_fraction=uniform,
                )
                refresh_validation = adaptive.build_weighted_cache(
                    heterogeneous.BENCHMARK, component,
                    args.validation_conditions, args.points_per_condition,
                    device,
                    args.seed + 800000 * round_index + 2000 * index + 1,
                    f"component {index} refresh validation beta={betas[index]:.6f}",
                    target_temperature=betas[index],
                    uniform_fraction=uniform,
                )
                train_component(
                    adaptive, component, refresh_train, refresh_validation,
                    f"component {index} held-beta refresh={betas[index]:.6f}",
                    args.seed + 900000 * round_index + 3000 * index,
                )
                refreshed.append(index)
                if stalls[index] >= args.stall_patience:
                    statuses[index] = "frozen"
                continue
            train_cache = adaptive.build_weighted_cache(
                heterogeneous.BENCHMARK, component,
                args.train_conditions, args.points_per_condition,
                device, args.seed + 200000 * round_index + 2000 * index,
                f"component {index} train beta={next_beta:.6f}",
                target_temperature=next_beta, uniform_fraction=uniform,
            )
            validation_cache = adaptive.build_weighted_cache(
                heterogeneous.BENCHMARK, component,
                args.validation_conditions, args.points_per_condition,
                device, args.seed + 200000 * round_index + 2000 * index + 1,
                f"component {index} validation beta={next_beta:.6f}",
                target_temperature=next_beta, uniform_fraction=uniform,
            )
            train_component(
                adaptive, component, train_cache, validation_cache,
                f"component {index} beta={next_beta:.6f}",
                args.seed + 300000 * round_index + 3000 * index,
            )
            betas[index] = float(next_beta)
            stalls[index] = 0
            advanced.append(index)
            if betas[index] >= 1.0 - 1e-10:
                statuses[index] = "complete"

        if not advanced:
            row = {
                "round": round_index,
                "components": mixture.n_components,
                "advanced": "",
                "refreshed": ",".join(map(str, refreshed)),
                "born": None,
                "common_beta": None,
                "validation_nll": None,
                "mean_overlap": None,
                "maximum_overlap": None,
                "betas": json.dumps(betas),
                "statuses": json.dumps(statuses),
            }
            history.append(row)
            save_outputs(
                args, mixture, graph, source_checkpoint, betas, statuses,
                history, adaptive,
            )
            if any(status == "active" for status in statuses):
                print(
                    "NO ADVANCE THIS ROUND: held components were refreshed; "
                    "retrying their ESS-controlled bridge steps.",
                    flush=True,
                )
                continue
            print(
                "STOPPING: every unfinished component exhausted its "
                "refresh patience.",
                flush=True,
            )
            break

        common_beta = min(betas[index] for index in advanced)
        common_uniform = adaptive.annealing_uniform_fraction(common_beta)
        shared_train = adaptive.build_weighted_cache(
            heterogeneous.BENCHMARK, mixture,
            args.train_conditions, args.points_per_condition,
            device, args.seed + 500000 * round_index,
            f"shared diversity train beta={common_beta:.6f}",
            target_temperature=common_beta,
            uniform_fraction=common_uniform,
        )
        shared_validation = adaptive.build_weighted_cache(
            heterogeneous.BENCHMARK, mixture,
            args.validation_conditions, args.points_per_condition,
            device, args.seed + 500000 * round_index + 1,
            f"shared diversity validation beta={common_beta:.6f}",
            target_temperature=common_beta,
            uniform_fraction=common_uniform,
        )
        diversity = diversity_refinement(
            mixture, shared_train, shared_validation, args
        )
        distinct = diversity["maximum_overlap"] <= args.maximum_overlap
        born = None
        if (
            len(advanced) >= 2 and distinct
            and mixture.n_components < args.maximum_components
        ):
            residual_train, owner = residual_cache(
                shared_train, mixture, args.residual_fraction
            )
            residual_validation, _ = residual_cache(
                shared_validation, mixture, args.residual_fraction
            )
            born = mixture.spawn_component(owner, noise_scale=args.birth_noise)
            set_equal_gate(mixture)
            train_component(
                adaptive, mixture.components[born],
                residual_train, residual_validation,
                f"new residual component {born} at beta={common_beta:.6f}",
                args.seed + 700000 * round_index,
            )
            betas.append(float(common_beta))
            stalls.append(0)
            statuses.append("active")
            print(
                f"BIRTH: component {born} cloned from owner {owner} at "
                f"beta={common_beta:.6f}; K={mixture.n_components}",
                flush=True,
            )
        elif len(advanced) >= 2 and not distinct:
            print(
                "NO BIRTH: advancing components remain too similar "
                f"(maximum overlap={diversity['maximum_overlap']:.4f}).",
                flush=True,
            )

        row = {
            "round": round_index,
            "components": mixture.n_components,
            "advanced": ",".join(map(str, advanced)),
            "refreshed": ",".join(map(str, refreshed)),
            "born": born,
            "common_beta": common_beta,
            "validation_nll": diversity["nll"],
            "mean_overlap": diversity["overlap"],
            "maximum_overlap": diversity["maximum_overlap"],
            "betas": json.dumps(betas),
            "statuses": json.dumps(statuses),
        }
        history.append(row)
        save_outputs(
            args, mixture, graph, source_checkpoint, betas, statuses,
            history, adaptive,
        )
        if all(status != "active" for status in statuses):
            print("STOPPING: all components are complete or frozen.", flush=True)
            break

    mixture_path = save_outputs(
        args, mixture, graph, source_checkpoint, betas, statuses,
        history, adaptive,
    )
    print("\nBRANCHING ANNEALED MIXTURE FINISHED", flush=True)
    print(f"components = {mixture.n_components}", flush=True)
    print(f"betas      = {betas}", flush=True)
    print(f"statuses   = {statuses}", flush=True)
    print(f"mixture    = {mixture_path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
