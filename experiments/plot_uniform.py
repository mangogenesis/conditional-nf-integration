from utils.plot_results import plot_results_file


def main():
    plot_results_file(
        "results/uniform/gaussian_family_d2_sweep.json",
        out_dir="results/plots",
    )


if __name__ == "__main__":
    main()