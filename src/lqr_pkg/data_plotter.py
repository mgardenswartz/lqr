import os

import matplotlib.pyplot as plt
import pandas as pd

from data_processor import ensure_directory_exists


def plot_from_csv(
    data: pd.DataFrame,
    output_dir: str,
    filename: str,
    dpi: int
) -> None:
    ensure_directory_exists(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))
    time_column = data.columns[0]
    state_columns = data.columns[1:]

    for state in state_columns:
        ax.plot(
            data[time_column],
            data[state],
            label=state.replace('_', ' ').title()
        )
    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("State Value", fontsize=12)
    ax.set_title("State Trajectories Over Time", fontsize=14)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.legend()
    plt.tight_layout()

    output_file_full_path = os.path.join(output_dir, filename)
    fig.savefig(output_file_full_path, bbox_inches="tight", dpi=dpi)
    plt.close(fig)
