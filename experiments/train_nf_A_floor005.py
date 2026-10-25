import torch
torch.cuda.empty_cache()
from benchmarks.registry import get_benchmark
from nf.cube_flow import CubeRealNVP
from nf.train import train_nf, save_nf_checkpoint
from utils.plot_loss import plot_loss_curve

BENCHMARK = "easy_four_gaussian"
benchmark_fn = get_benchmark(BENCHMARK)
DIM = 2 
def main():
    torch.manual_seed(0)

    model = CubeRealNVP(dim=DIM, n_blocks=12, hidden=128,eps_mix=0.05)

    out = train_nf(
        model=model,
        benchmark_fn=benchmark_fn,
        dim=DIM,
        steps=2000,
        batch_size=20000,
        lr=1e-4,
        loss_name="kl",
        device = "cuda" if torch.cuda.is_available() else "cpu",
    )

    save_nf_checkpoint(out, "results/checkpoints/nf_A_kl_floor005.pt")
    plot_loss_curve(out["history"], "results/plots/loss_A_kl_floor005.png", title="A: KL_floor005")


if __name__ == "__main__":
    main()