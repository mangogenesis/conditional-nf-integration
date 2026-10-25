import torch

from benchmarks.registry import get_benchmark
from nf.cube_flow import CubeRealNVP
from nf.train import train_nf, save_nf_checkpoint
from utils.plot_loss import plot_loss_curve

BENCHMARK = "double_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

def main():
    torch.manual_seed(0)

    model = CubeRealNVP(dim=2, n_blocks=12, hidden=128)

    out = train_nf(
        model=model,
        benchmark_fn=benchmark_fn,
        dim=2,
        steps=3000,
        batch_size=16384,
        lr=1e-4,
        loss_name="kl_raw_std_loss",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )

    save_nf_checkpoint(out, "results/checkpoints/nf_C_kl_raw_std.pt")
    plot_loss_curve(out["history"], "results/plots/loss_C_kl_raw_std.png", title="C: KL+std")


if __name__ == "__main__":
    main()