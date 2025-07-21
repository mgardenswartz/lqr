import sys
import math
import os
import time
import logging

import numpy as np
import pandas as pd

import data_processor
import dynamics
import data_plotter
from config import OUTPUT_PARENT_DIR_NAME, OUTPUT_DATA_FILENAME, \
    FIGURE_SAVE_DPI

logging.basicConfig(level=logging.INFO)


def main() -> None:
    x_0 = np.array([
        [-math.pi/4],  # deflection # rad
        [0]  # speed (rad/s)
    ])
    t_0 = 0
    t_f = 30
    controller_time_step_size = 1e-2
    solver_time_step_size = 1e-3  # Must be less than above

    data_processor.write_ode_output_to_csv(
        just_started=True,
        output_dir=OUTPUT_PARENT_DIR_NAME,
        filename=OUTPUT_DATA_FILENAME,
        current_time=t_0,
        array_to_save=x_0
    )

    # An inverted pendulum linearized about the vertical orientation    
    g = 9.81  # m/s2
    b = 0.005  # N-s
    L = 0.5  # m
    m = 0.1  # kg
    A = np.array([
        [  0,            1],
        [g/L,  -b/(m*L**2)]
    ])
    B = np.array([
        [0],
        [1]
    ])

    # LQR Costs (Bryson's Rule)
    torque_max = 1  # N-m
    Q = np.eye(2)
    R = np.array([
        [1/torque_max**2]
    ])
    K = dynamics.CLTI_LQR_gain(A=A, B=B, Q=Q, R=R)
    print("LQR gain K:", K)

    def my_inverted_pendulum_ode(t, x):
        return dynamics.P_controller_CLTI_dynamics(t=t, x=x, A=A, B=B, K=K)

    current_state = x_0.flatten()
    current_time = t_0
    num_time_steps = math.ceil(t_f / controller_time_step_size)
    for step in range(num_time_steps):
        start_time = time.perf_counter()
        current_state = \
        dynamics.rk4_solver(
            current_time=current_time,
            current_state=current_state,
            integration_time_window=[current_time, current_time + controller_time_step_size],
            time_step_size=solver_time_step_size,
            ode_func=my_inverted_pendulum_ode
        )
        end_time = time.perf_counter()
        logging.debug(f"Integrating ODE took {end_time-start_time} seconds.")

        current_time += controller_time_step_size

        start_time = time.perf_counter()
        data_processor.write_ode_output_to_csv(
            just_started=False,
            output_dir=OUTPUT_PARENT_DIR_NAME,
            filename=OUTPUT_DATA_FILENAME,
            current_time=current_time,
            array_to_save=current_state
        )
        end_time = time.perf_counter()
        logging.debug(f"Writing output took {end_time-start_time} seconds.")

        progress = step / num_time_steps * 100
        print_progress(progress=progress)
    print_progress(progress=100)
    print("\nSimulation completed.")

    output_data_full_path = os.path.join(OUTPUT_PARENT_DIR_NAME, OUTPUT_DATA_FILENAME)
    output_data = pd.read_csv(output_data_full_path)
    data_plotter.plot_from_csv(
        data=output_data,
        output_dir=OUTPUT_PARENT_DIR_NAME,
        filename="my_figure.svg",
        dpi=FIGURE_SAVE_DPI
    )


def print_progress(
    progress: float
) -> None:
    print(
        f"Progress: {progress:.3f}%",
        end="\r",
        flush=True
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("User terminated program.")
        sys.exit()
