import torch
from nf.cube_flow import CubeRealNVP


def main():
    model = CubeRealNVP(dim=2, n_blocks=24, hidden=256)
    x = model.sample(1000)

    print("sample shape:", x.shape)
    print("min:", x.min().item(), "max:", x.max().item())

    lp = model.log_prob(x)
    print("log_prob shape:", lp.shape)
    print("log_prob mean:", lp.mean().item())


if __name__ == "__main__":
    main()