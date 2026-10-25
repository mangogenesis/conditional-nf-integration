from utils.plot_results import plot_compare
from benchmarks.registry import get_benchmark
BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)
DIM=2
JSON_PATHS = [
    f"results/uniform/{benchmark_fn.__name__}_d{DIM}_sweep.json",
    f"results/vegas/{benchmark_fn.__name__}_d{DIM}_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_untrained_sweep.json",
    f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor001_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor005_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor010_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_A_kl_floor010-setting2_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_B_raw_std_sweep.json",
    # f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_C_kl_raw_std_sweep.json",
    f"results/nf/{benchmark_fn.__name__}_d{DIM}_nf_spline_kl_sweep.json",
]

LABELS = [
    "Uniform MC",
    "VEGAS",
    # "NF untrained",
    "NF A: KL",
    # "NF A: KL_floor001",
    # "NF A: KL_floor005",
    # "NF A: KL_floor010",
    # "NF A: KL_floor010-setting2",
    # "NF B: raw_std",
    # "NF C: KL + raw_std",
    "NF_SPLINE A: KL",
]


def main():
    # Estimate
    plot_compare(
        json_paths=JSON_PATHS,
        labels=LABELS,
        metric="estimate",
        save_path="results/plots/final_estimate_comparison.png",
        title="Estimate vs events",
        ylabel="Integral estimate",
        logy=False,
    )

    # Variance
    plot_compare(
        json_paths=JSON_PATHS,
        labels=LABELS,
        metric="variance",
        save_path="results/plots/final_variance_comparison.png",
        title="Variance vs events",
        ylabel="Estimator variance",
        logy=True,
    )

    # Runtime
    plot_compare(
        json_paths=JSON_PATHS,
        labels=LABELS,
        metric="runtime",
        save_path="results/plots/final_runtime_comparison.png",
        title="Runtime vs events",
        ylabel="Runtime (s)",
        logy=False,
    )

    # ESS only for NF methods
    plot_compare(
        json_paths=JSON_PATHS[2:],
        labels=LABELS[2:],
        metric="ess",
        save_path="results/plots/final_ess_nf_comparison.png",
        title="ESS vs events (NF only)",
        ylabel="Effective sample size",
        logy=False,
    )


if __name__ == "__main__":
    main()