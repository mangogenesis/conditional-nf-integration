import os
import json
import torch
from benchmarks.registry import get_benchmark
from mc_baselines.vegas_integrate import vegas_integrate

DIM = 2
BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

NITN = 10
BUDGETS = list(range(10000, 100001, 10000))


def run():
    results = []

    for i, budget in enumerate(BUDGETS):
        torch.manual_seed(i)

        neval = budget // NITN

        res = vegas_integrate(
            lambda x: benchmark_fn(x),
            dim=DIM,
            nitn=NITN,
            neval=neval,
        )

        res["trial"] = i
        res["method"] = "vegas"
        res["benchmark"] = BENCHMARK
        res["dim"] = DIM

        results.append(res)
        print(f"run {i}, budget={budget} -> {res}")

    os.makedirs("results/vegas", exist_ok=True)

    with open(f"results/vegas/{benchmark_fn.__name__}_d{DIM}_sweep.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run()