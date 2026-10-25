import time
import torch


def effective_sample_size(weights: torch.Tensor) -> float:
    s1 = weights.sum()
    s2 = (weights ** 2).sum()
    return float((s1 ** 2 / (s2 + 1e-12)).item())


def nf_integrate(model, benchmark_fn, n_events: int, device: str = "cpu"):
    """
    Importance sampling with an NF proposal on [0,1]^d.

    Returns:
        estimate, variance of estimator, runtime, events, ess
    """
    model = model.to(device)
    model.eval()

    t0 = time.time()

    with torch.no_grad():
        x = model.sample(n_events, device=device)          # (N, d)
        f_vals = benchmark_fn(x)                           # (N,)
        log_q = model.log_prob(x)                          # (N,)

        weights = f_vals * torch.exp(-log_q)               # f/q

        estimate = weights.mean()
        variance = weights.var(unbiased=True) / n_events
        ess = effective_sample_size(weights)

    runtime = time.time() - t0

    return {
        "estimate": float(estimate.item()),
        "variance": float(variance.item()),
        "runtime": float(runtime),
        "events": int(n_events),
        "ess": float(ess),
    }