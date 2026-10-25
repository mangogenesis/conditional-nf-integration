"""Condition-gated mixtures of normalized conditional proposal models."""

from __future__ import annotations

import copy
import math

import torch
from torch import Tensor, nn


@torch.no_grad()
def cluster_worst_conditions(cache, clusters=3, worst_fraction=0.10, iterations=25):
    """Partition low-ESS conditions into deterministic condition-space clusters."""
    alpha = cache["alpha"]
    points = alpha.shape[1]
    normalized_ess = 1.0 / (
        points * alpha.double().square().sum(dim=1).clamp_min(1e-300)
    )
    worst_count = max(int(clusters), int(math.ceil(alpha.shape[0] * worst_fraction)))
    worst_count = min(alpha.shape[0], worst_count)
    worst = torch.topk(normalized_ess, worst_count, largest=False).indices

    condition = cache["cond"][worst, 0].double()
    scale = condition.std(dim=0, unbiased=False).clamp_min(1e-8)
    z = (condition - condition.mean(dim=0)) / scale
    cluster_count = min(int(clusters), worst_count)

    centers = [int(torch.argmin(normalized_ess[worst]).item())]
    distance = torch.full((worst_count,), float("inf"), device=z.device)
    for _ in range(1, cluster_count):
        newest = z[centers[-1]]
        distance = torch.minimum(distance, (z - newest).square().sum(dim=1))
        distance[torch.tensor(centers, device=z.device)] = -1.0
        centers.append(int(torch.argmax(distance).item()))
    centroids = z[torch.tensor(centers, device=z.device)].clone()

    labels = None
    for _ in range(iterations):
        new_labels = torch.cdist(z, centroids).argmin(dim=1)
        if labels is not None and torch.equal(new_labels, labels):
            break
        labels = new_labels
        for cluster in range(cluster_count):
            selected = labels == cluster
            if selected.any():
                centroids[cluster] = z[selected].mean(dim=0)

    assert labels is not None
    groups = []
    for cluster in range(cluster_count):
        members = worst[labels == cluster]
        if members.numel():
            groups.append(members)
    return groups, normalized_ess


def subset_condition_cache(cache, condition_indices):
    """Select conditions from a weighted cache without copying metadata."""
    condition_count = cache["x"].shape[0]
    return {
        key: value[condition_indices]
        if torch.is_tensor(value) and value.ndim and value.shape[0] == condition_count
        else value
        for key, value in cache.items()
    }


class ConditionalMixtureProposal(nn.Module):
    """An exactly normalized mixture with condition-dependent component weights."""

    def __init__(self, components, cond_dim=3, hidden=64, max_components=8):
        super().__init__()
        if not 1 <= len(components) <= max_components:
            raise ValueError("invalid initial component count")
        self.components = nn.ModuleList(components)
        self.max_components = int(max_components)
        self.gate = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, max_components),
        )
        with torch.no_grad():
            self.gate[-1].weight.zero_()
            self.gate[-1].bias.fill_(-8.0)
            self.gate[-1].bias[: len(components)].zero_()

    @property
    def n_components(self):
        return len(self.components)

    def log_gate(self, cond: Tensor) -> Tensor:
        return torch.log_softmax(self.gate(cond)[:, : self.n_components], dim=-1)

    def component_log_probs(self, x: Tensor, cond: Tensor) -> Tensor:
        return torch.stack(
            [component.log_prob(x, cond).reshape(-1) for component in self.components],
            dim=1,
        )

    def log_prob(self, x: Tensor, cond: Tensor) -> Tensor:
        return torch.logsumexp(self.log_gate(cond) + self.component_log_probs(x, cond), dim=1)

    @torch.no_grad()
    def sample(self, n: int, cond: Tensor) -> Tensor:
        if cond.shape[0] == 1:
            cond = cond.expand(n, -1)
        elif cond.shape[0] != n:
            raise ValueError("condition batch must have size one or n")
        labels = torch.distributions.Categorical(logits=self.log_gate(cond)).sample()
        result = torch.empty(
            n, self.components[0].dim, device=cond.device, dtype=cond.dtype
        )
        for index, component in enumerate(self.components):
            selected = torch.nonzero(labels == index, as_tuple=False).flatten()
            if selected.numel():
                result[selected] = component.sample(
                    int(selected.numel()), cond[selected]
                )
        return result

    @torch.no_grad()
    def spawn_component(self, source_index: int, noise_scale: float = 2e-3) -> int:
        """Clone a useful component and perturb it to break mixture symmetry."""
        if self.n_components >= self.max_components:
            raise RuntimeError("maximum number of components reached")
        child = copy.deepcopy(self.components[int(source_index)])
        for parameter in child.parameters():
            if parameter.is_floating_point():
                parameter.add_(noise_scale * torch.randn_like(parameter))
        self.components.append(child)
        new_index = self.n_components - 1
        self.gate[-1].bias[new_index] = -2.0
        return new_index
