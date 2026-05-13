import sys

import matplotlib.pyplot as plt
import pandas as pd


def main():
    input_file = "benchmark_results.csv"
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]

    output_file = "benchmark_total_time.png"
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    gpu_output_file = "benchmark_avg_gpu_time.png"
    if len(sys.argv) >= 4:
        gpu_output_file = sys.argv[3]

    data = pd.read_csv(input_file)

    required_columns = {"mode", "num_particles", "total_time_s"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {input_file}: {sorted(missing_columns)}"
        )

    modes = list(data["mode"].unique())

    fig, ax = plt.subplots(figsize=(10, 6))

    for mode in modes:
        mode_data = data[data["mode"] == mode].sort_values("num_particles")

        ax.plot(
            mode_data["num_particles"],
            mode_data["total_time_s"],
            label=mode,
            marker="o",
            linewidth=2
        )

    ax.set_title("Benchmark Total Time by Mode")
    ax.set_xlabel("Number of Particles")
    ax.set_ylabel("Total Time (s)")
    ax.set_xscale("log")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_file, dpi=200)
    print(f"Saved benchmark plot to {output_file}")

    gpu_required_columns = {
        "mode",
        "num_particles",
        "avg_gpu_malloc_s",
        "avg_gpu_h2d_s",
        "avg_gpu_kernel_s",
        "avg_gpu_d2h_s",
    }
    missing_gpu_columns = gpu_required_columns - set(data.columns)
    if missing_gpu_columns:
        raise ValueError(
            f"Missing required GPU timing columns in {input_file}: "
            f"{sorted(missing_gpu_columns)}"
        )

    gpu_data = data[data["mode"] != "cpu"].copy()
    gpu_data["avg_gpu_total_s"] = (
        gpu_data["avg_gpu_malloc_s"] +
        gpu_data["avg_gpu_h2d_s"] +
        gpu_data["avg_gpu_kernel_s"] +
        gpu_data["avg_gpu_d2h_s"]
    )

    gpu_modes = list(gpu_data["mode"].unique())
    gpu_fig, gpu_ax = plt.subplots(figsize=(10, 6))

    for mode in gpu_modes:
        mode_data = gpu_data[gpu_data["mode"] == mode].sort_values("num_particles")

        gpu_ax.plot(
            mode_data["num_particles"],
            mode_data["avg_gpu_total_s"],
            label=mode,
            marker="o",
            linewidth=2
        )

    gpu_ax.set_title("GPU Time by Mode")
    gpu_ax.set_xlabel("Number of Particles")
    gpu_ax.set_ylabel("GPU Time (s)")
    gpu_ax.set_xscale("log")
    gpu_ax.grid(True, linestyle="--", alpha=0.5)
    gpu_ax.legend()

    gpu_fig.tight_layout()
    gpu_fig.savefig(gpu_output_file, dpi=200)
    print(f"Saved GPU benchmark plot to {gpu_output_file}")


if __name__ == "__main__":
    main()
