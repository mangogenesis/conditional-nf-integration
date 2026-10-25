import os
import json
import torch

from benchmarks.registry import get_benchmark
from mc_baselines.uniform_mc import uniform_mc_integrate

BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

DIM = 2
N_EVENTS = list(range(10000, 100001, 10000))


def run():
    results = []

    for i, n in enumerate(N_EVENTS):

        torch.manual_seed(i)

        res = uniform_mc_integrate(
            lambda x: benchmark_fn(x),
            dim=DIM,
            N=n
        )

        res["trial"] = i
        res["events"] = n
        res["method"] = "uniform"
        res["benchmark"] = BENCHMARK
        res["dim"] = DIM

        results.append(res)

        print(f"run {i}, N={n} → {res}")

    os.makedirs("results/uniform", exist_ok=True)

    with open(f"results/uniform/{benchmark_fn.__name__}_d{DIM}_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()