import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

def compute_global_axis_limits(data):
    x_min, x_max = data["x"].min(), data["x"].max()
    y_min, y_max = data["y"].min(), data["y"].max()
    z_min, z_max = data["z"].min(), data["z"].max()

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    z_center = (z_min + z_max) / 2.0

    max_range = max(x_max - x_min, y_max - y_min, z_max - z_min)
    half_range = max(max_range * 0.55, 1.0)

    return (
        (x_center - half_range, x_center + half_range),
        (y_center - half_range, y_center + half_range),
        (z_center - half_range, z_center + half_range),
    )

def main():
    filename = "particles.csv"
    if len(sys.argv) >= 2:
        filename = sys.argv[1]

    output_file = "simulation.gif"
    if len(sys.argv) >= 3:
        output_file = sys.argv[2]

    data = pd.read_csv(filename)
    steps = sorted(data["step"].unique())

    has_charge = "charge" in data.columns
    x_limits, y_limits, z_limits = compute_global_axis_limits(data)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_index):
        ax.clear()

        step = steps[frame_index]
        step_data = data[data["step"] == step]

        if has_charge:
            colors = ["red" if q > 0 else "blue" for q in step_data["charge"]]
        else:
            colors = "black"

        ax.scatter(
            step_data["x"],
            step_data["y"],
            step_data["z"],
            c=colors,
            s=8,
            alpha=0.8
        )

        ax.set_title(f"3D Particle Simulation - Step {step}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        ax.set_zlim(*z_limits)
        ax.set_box_aspect((1, 1, 1))

        if has_charge:
            legend_elements = [
                Line2D([0], [0], marker="o", color="w", label="Positive charge",
                       markerfacecolor="red", markersize=8),
                Line2D([0], [0], marker="o", color="w", label="Negative charge",
                       markerfacecolor="blue", markersize=8),
            ]
            ax.legend(handles=legend_elements, loc="upper right")

    anim = FuncAnimation(
        fig,
        update,
        frames=len(steps),
        interval=80,
        repeat=True
    )

    anim.save(output_file, writer="pillow", fps=20)
    print(f"Saved animation to {output_file}")

if __name__ == "__main__":
    main()
