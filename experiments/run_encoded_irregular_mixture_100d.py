#!/usr/bin/env python3
"""Train the existing bridge+copula NF on an encoded family of 100D targets.

Each target task starts from a 32-dimensional descriptor.  A small
autoencoder is trained once, then frozen.  The NF receives only

    (physical condition c in R^3, encoded task z in R^8),

while the target evaluator decodes z back to task parameters.  The adaptive
annealing bridge, marginal flows, and structured copula are otherwise the
same implementations used by the existing 100D experiments.

The decoded task parameters alter mixture gates, peak amplitudes,
interactions, and widths.  Every component remains a product of normalized
conditional truncated normals, so I(c,z)=1 for every encoded task even though
that fact is hidden from training.
"""

from __future__ import annotations

import argparse
import importlib
import math
import sys
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = ROOT / "experiments"
for path in (ROOT, EXPERIMENTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import run_irregular_mixture_100d as irregular  # noqa: E402


adaptive = irregular.adaptive
BENCHMARK = "encoded_irregular_mixture_100"
PHYSICAL_DIM = 3
DESCRIPTOR_DIM = 32
LATENT_DIM = 8
CONDITION_DIM = PHYSICAL_DIM + LATENT_DIM
AUTOENCODER_SEED = 910247


class TaskAutoencoder(nn.Module):
    """Compress a task descriptor; the decoder is also the target hypernetwork."""

    def __init__(self, descriptor_dim=DESCRIPTOR_DIM, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(descriptor_dim, 64),
            nn.SiLU(),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.Linear(32, latent_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.Linear(64, descriptor_dim),
            nn.Tanh(),
        )

    def forward(self, descriptor):
        code = self.encoder(descriptor)
        return code, self.decoder(code)


def _descriptor_basis():
    """Fixed nonlinear task manifold used only to create benchmark tasks."""
    generator = torch.Generator(device="cpu").manual_seed(AUTOENCODER_SEED)
    linear = torch.randn(6, DESCRIPTOR_DIM, generator=generator) / math.sqrt(6)
    harmonic = torch.randn(6, DESCRIPTOR_DIM, generator=generator) / math.sqrt(6)
    bias = 0.2 * torch.randn(DESCRIPTOR_DIM, generator=generator)
    return linear, harmonic, bias


DESCRIPTOR_LINEAR, DESCRIPTOR_HARMONIC, DESCRIPTOR_BIAS = _descriptor_basis()


def sample_descriptors(n, device="cpu", generator=None):
    """Draw high-dimensional task descriptions from a smooth 6D manifold."""
    factors = 2.0 * torch.rand(n, 6, generator=generator, device=device) - 1.0
    linear = DESCRIPTOR_LINEAR.to(device)
    harmonic = DESCRIPTOR_HARMONIC.to(device)
    bias = DESCRIPTOR_BIAS.to(device)
    raw = factors @ linear + 0.55 * torch.sin(math.pi * factors) @ harmonic + bias
    return torch.tanh(raw)


def train_task_autoencoder(steps, tasks, device, learning_rate):
    """Fit the task representation once, independently of NF training."""
    torch.manual_seed(AUTOENCODER_SEED)
    model = TaskAutoencoder().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-6)
    generator = torch.Generator(device="cpu").manual_seed(AUTOENCODER_SEED + 1)
    descriptors = sample_descriptors(tasks, generator=generator).to(device)
    validation = sample_descriptors(4096, generator=generator).to(device)
    batch_size = min(2048, tasks)
    best_loss = math.inf
    best_state = None

    for step in range(1, steps + 1):
        index = torch.randint(0, tasks, (batch_size,), device=device)
        batch = descriptors[index]
        _, reconstruction = model(batch)
        loss = (reconstruction - batch).square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 100 == 0 or step == steps:
            with torch.no_grad():
                _, predicted = model(validation)
                val_loss = float((predicted - validation).square().mean())
            print(
                f"task autoencoder {step:4d}/{steps}: "
                f"train MSE={float(loss):.6g} val MSE={val_loss:.6g}"
            )
            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, best_loss


TASK_AUTOENCODER = None


def sample_encoded_conditions(n, device):
    """Supply physical conditions and learned task codes to the old trainer."""
    physical = torch.rand(n, PHYSICAL_DIM, device=device)
    descriptor = sample_descriptors(n, device=device)
    with torch.no_grad():
        code = TASK_AUTOENCODER.encoder(descriptor)
    return torch.cat((physical, code), dim=1)


def decoded_task_parameters(cond):
    code = cond[:, PHYSICAL_DIM:]
    return TASK_AUTOENCODER.decoder(code)


def encoded_irregular_log_integrand(x, cond):
    if x.ndim != 2 or x.shape[1] != irregular.DIMENSION:
        raise ValueError(f"{BENCHMARK} requires shape (batch,{irregular.DIMENSION})")
    if cond.ndim != 2 or cond.shape[1] != CONDITION_DIM:
        raise ValueError(f"condition must have shape (batch,{CONDITION_DIM})")

    physical = cond[:, :PHYSICAL_DIM]
    task = decoded_task_parameters(cond).to(dtype=x.dtype)
    parameters = irregular.parameters_on(x.device, x.dtype)
    batch = x.shape[0]
    components = irregular.COMPONENTS
    component_index = torch.arange(components, device=x.device)
    log_component = torch.zeros(batch, components, device=x.device, dtype=x.dtype)

    # Four task-controlled parameter groups, one value per mixture component.
    gate_shift = 0.9 * task[:, 0:8]
    amplitude_scale = torch.exp(0.28 * task[:, 8:16])
    interaction_scale = torch.exp(0.32 * task[:, 16:24])
    width_scale = torch.exp(0.38 * task[:, 24:32])

    for child in range(irregular.DIMENSION):
        if child == 0:
            a = physical[:, :1].expand(-1, components)
            b = physical[:, 1:2].expand(-1, components)
        else:
            a = x[:, parameters["parent_a"][:, child]]
            b = x[:, parameters["parent_b"][:, child]]

        frequency = parameters["frequency"][:, child][None, :]
        phase = parameters["phase"][:, child][None, :]
        c0, c1, c2 = physical[:, :1], physical[:, 1:2], physical[:, 2:3]
        # The task also causes a smooth coordinate-dependent phase drift.
        task_phase = 0.22 * task[:, child % DESCRIPTOR_DIM:child % DESCRIPTOR_DIM + 1]
        wave = torch.sin(2 * math.pi * frequency * a + phase + 0.8 * c0 + task_phase)
        cross = torch.cos(2 * math.pi * (a - b) + 0.5 * phase - 0.7 * c1)
        switch = torch.tanh(4.0 * (c2 - 0.5) + 1.2 * cross)
        signal = (
            parameters["amplitude"][None, :] * amplitude_scale * wave
            + parameters["interaction"][None, :] * interaction_scale * cross * switch
        )
        drift = 0.035 * torch.sin(
            0.17 * child + component_index[None, :] + math.pi * c1
        )
        mu = (0.5 + signal + drift).clamp(0.055, 0.945)
        sigma = parameters["base_sigma"][None, :] * width_scale * (
            0.78 + 0.38 * c2 + 0.06 * ((child + component_index) % 5)[None, :]
        )
        x_child = x[:, child:child + 1].expand(-1, components)
        log_component += adaptive.truncated_normal_logpdf(x_child, mu, sigma)

    logits = physical @ parameters["gate_weight"].transpose(0, 1)
    logits = logits + parameters["gate_bias"][None, :] + gate_shift
    logits = logits + 0.65 * torch.sin(
        2 * math.pi * (
            physical[:, :1] * parameters["condition_phase"][:, 0][None, :]
            + physical[:, 1:2] * parameters["condition_phase"][:, 1][None, :]
        )
    )
    log_density = torch.logsumexp(torch.log_softmax(logits, dim=1) + log_component, dim=1)
    inside = ((x >= 0.0) & (x <= 1.0)).all(dim=1)
    return torch.where(inside, log_density, torch.full_like(log_density, -torch.inf))


def _set_condition_dimension(dimension):
    """Update both the trainer and the module that owns its model classes."""
    adaptive.COND_DIM = dimension
    for cls in (adaptive.MarginalBank, adaptive.StructuredCopula, adaptive.MarginalCopulaModel):
        module = importlib.import_module(cls.__module__)
        if hasattr(module, "COND_DIM"):
            module.COND_DIM = dimension


def register_encoded_benchmark():
    irregular.register_benchmark()
    previous_target = adaptive.target_log_integrand
    _set_condition_dimension(CONDITION_DIM)

    def target_log_integrand(benchmark, x, cond):
        if benchmark == BENCHMARK:
            return encoded_irregular_log_integrand(x, cond)
        return previous_target(benchmark, x, cond)

    adaptive.SUPPORTED_BENCHMARKS.add(BENCHMARK)
    adaptive.BENCHMARK_DIMENSIONS[BENCHMARK] = irregular.DIMENSION
    adaptive.ORACLE_INTEGRALS[BENCHMARK] = 1.0
    adaptive.target_log_integrand = target_log_integrand
    adaptive.sample_conditions = sample_encoded_conditions


def parse_wrapper_arguments():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task-ae-steps", type=int, default=800)
    parser.add_argument("--task-ae-tasks", type=int, default=32768)
    parser.add_argument("--task-ae-learning-rate", type=float, default=2e-3)
    arguments, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    return arguments


def output_directory_from_arguments(default="results/encoded_irregular_mixture_100"):
    if "--output" in sys.argv:
        index = sys.argv.index("--output")
        if index + 1 < len(sys.argv):
            return Path(sys.argv[index + 1])
    return Path(default)


def main():
    global TASK_AUTOENCODER
    args = parse_wrapper_arguments()
    requested_device = "cuda" if "--device" not in sys.argv else sys.argv[sys.argv.index("--device") + 1]
    device = torch.device(requested_device if requested_device != "cuda" or torch.cuda.is_available() else "cpu")
    TASK_AUTOENCODER, validation_mse = train_task_autoencoder(
        args.task_ae_steps,
        args.task_ae_tasks,
        device,
        args.task_ae_learning_rate,
    )
    output = output_directory_from_arguments()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": TASK_AUTOENCODER.state_dict(),
            "descriptor_dim": DESCRIPTOR_DIM,
            "latent_dim": LATENT_DIM,
            "physical_dim": PHYSICAL_DIM,
            "validation_mse": validation_mse,
            "seed": AUTOENCODER_SEED,
        },
        output / "task_autoencoder.pt",
    )
    print(
        f"Encoded task family: descriptor {DESCRIPTOR_DIM} -> latent {LATENT_DIM}; "
        f"NF condition dimension={CONDITION_DIM}; decoder validation MSE={validation_mse:.6g}"
    )
    register_encoded_benchmark()
    adaptive.main()


if __name__ == "__main__":
    main()
