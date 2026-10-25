import os
import json
import torch

from benchmarks.registry import get_benchmark
from nf.cube_flow import CubeRealNVP
from nf.cube_spline_flow import CubeSplineFlow
from nf.evaluate import nf_integrate

DIM = 2
BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

N_EVENTS = list(range(10000, 100001, 10000))


def build_model(model_type, dim, n_blocks, hidden, eps_mix=0.0, n_bins=8, permute="reverse"):
    if model_type == "realnvp":
        return CubeRealNVP(
            dim=dim,
            n_blocks=n_blocks,
            hidden=hidden,
            eps_mix=eps_mix,
        )

    if model_type == "spline":
        return CubeSplineFlow(
            dim=dim,
            n_blocks=n_blocks,
            hidden=hidden,
            n_bins=n_bins,
            permute=permute,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_checkpoint(
    checkpoint_path,
    method_name,
    out_path,
    model_type="realnvp",
    n_blocks=12,
    hidden=128,
    eps_mix=0.0,
    n_bins=8,
    permute="reverse",
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(
        model_type=model_type,
        dim=DIM,
        n_blocks=n_blocks,
        hidden=hidden,
        eps_mix=eps_mix,
        n_bins=n_bins,
        permute=permute,
    ).to(device)

    if checkpoint_path is not None:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])

    model.eval()

    results = []

    for i, n in enumerate(N_EVENTS):
        torch.manual_seed(i)

        res = nf_integrate(
            model=model,
            benchmark_fn=benchmark_fn,
            n_events=n,
            device=device,
        )

        res["trial"] = i
        res["method"] = method_name
        res["benchmark"] = BENCHMARK
        res["dim"] = DIM
        res["model_type"] = model_type
        res["n_blocks"] = n_blocks
        res["hidden"] = hidden
        res["eps_mix"] = eps_mix
        res["n_bins"] = n_bins if model_type == "spline" else None
        res["permute"] = permute if model_type == "spline" else None

        results.append(res)
        print(f"{method_name} | run {i}, N={n} -> {res}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved -> {out_path}")


def main():
    # RealNVP baseline
    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_A_kl.pt",
        method_name="nf_A_kl",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_sweep.json",
        model_type="realnvp",
        n_blocks=12,
        hidden=128,
        eps_mix=0.0,
    )

    # # RealNVP + floor
    # evaluate_checkpoint(
    #     checkpoint_path="results/checkpoints/nf_A_kl_floor001.pt",
    #     method_name="nf_A_kl_floor001",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor001_sweep.json",
    #     model_type="realnvp",
    #     n_blocks=12,
    #     hidden=128,
    #     eps_mix=0.01,
    # )

    # evaluate_checkpoint(
    #     checkpoint_path="results/checkpoints/nf_A_kl_floor005.pt",
    #     method_name="nf_A_kl_floor005",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor005_sweep.json",
    #     model_type="realnvp",
    #     n_blocks=12,
    #     hidden=128,
    #     eps_mix=0.05,
    # )

    # evaluate_checkpoint(
    #     checkpoint_path="results/checkpoints/nf_A_kl_floor010.pt",
    #     method_name="nf_A_kl_floor010",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor010_sweep.json",
    #     model_type="realnvp",
    #     n_blocks=12,
    #     hidden=128,
    #     eps_mix=0.10,
    # )

    # evaluate_checkpoint(
    #     checkpoint_path="results/checkpoints/nf_A_kl_floor010-setting2.pt",
    #     method_name="nf_A_kl_floor010_setting2",
    #     out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor010_setting2_sweep.json",
    #     model_type="realnvp",
    #     n_blocks=20,
    #     hidden=256,
    #     eps_mix=0.10,
    # )

    # Cube-native spline flow
    evaluate_checkpoint(
        checkpoint_path="results/checkpoints/nf_spline_kl.pt",
        method_name="nf_spline_kl",
        out_path=f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_spline_kl_sweep.json",
        model_type="spline",
        n_blocks=4,
        hidden=128,
        n_bins=16,
        permute="reverse",
    )


if __name__ == "__main__":
    main()