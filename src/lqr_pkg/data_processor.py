import os

import numpy as np
import pandas as pd

from config import TIME_DATA_LABEL, STATE_DATA_LABEL, DATA_DECIMAL_PLACES


def ensure_directory_exists(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def write_ode_output_to_csv(
    just_started: bool,
    output_dir: str,
    filename: str,
    current_time: float,
    array_to_save: np.ndarray
) -> None:
    ensure_directory_exists(output_dir)
    array_to_save_flattened = array_to_save.flatten()
    data = {TIME_DATA_LABEL: [current_time]}
    for i, val in enumerate(array_to_save_flattened):
        data[f"{STATE_DATA_LABEL}_{i}"] = [val]

    output_data = pd.DataFrame(data)
    output_data = output_data.round(DATA_DECIMAL_PLACES)
    output_file_full_path = os.path.join(output_dir, filename)

    output_data.to_csv(
        path_or_buf=output_file_full_path,
        index=False,
        header=just_started,
        mode='a' if not just_started else 'w'
    )
