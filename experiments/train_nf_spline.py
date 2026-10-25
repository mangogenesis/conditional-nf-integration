import torch

from benchmarks.registry import get_benchmark
from nf.cube_spline_flow import CubeSplineFlow
from nf.train import train_nf, save_nf_checkpoint
from utils.plot_loss import plot_loss_curve

DIM = 2
BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)

def main():
    torch.manual_seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CubeSplineFlow(
        dim=DIM,
        n_blocks=4,
        hidden=128,
        n_bins=16,
        permute="reverse",
    ).to(device)

    out = train_nf(
        model=model,
        benchmark_fn=benchmark_fn,
        dim=DIM,
        steps=2000,
        batch_size=20000,
        lr=1e-5,
        loss_name="kl",
        device=device,
    )

    save_nf_checkpoint(out, "results/checkpoints/nf_spline_kl.pt")
    plot_loss_curve(out["history"], "results/plots/loss_spline_kl.png", title="Spline NF: KL")

if __name__ == "__main__":
    main()
