# This was written by AI and should addressed separately from the rest of the codebase.

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.linalg import solve_continuous_are
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def create_lqr_animation(
    csv_path: str,
    output_dir: str,
    output_filename: str = "lqr_cost_animation.mp4",
    fps: int = 30,
    dpi: int = 150
) -> None:
    """
    Generates a 3D real-time animation of a system's state trajectory on its
    LQR cost function landscape.

    Args:
        csv_path: Path to the state data CSV file.
        output_dir: Directory to save the animation and plot.
        output_filename: Filename for the output video.
        fps: Frames per second for the output video.
        dpi: Dots per inch for the output video.
    """
    logging.info("Starting LQR animation generation...")

    # 1. Load State Data from CSV
    try:
        df = pd.read_csv(csv_path)
        sim_times = df['Time'].values
        states = df[['State_0', 'State_1']].values
    except FileNotFoundError:
        logging.error(f"Error: State data file not found at '{csv_path}'")
        return

    # 2. Define System and LQR Parameters (from your main.py)
    k_spring = 25.42  # N/m
    m = 1.0  # kg

    A = np.array([
        [ 0,           1],
        [-k_spring/m,  0]
    ])
    B = np.array([[0], [1/m]])
    Q = np.array([
        [1e-6, 0.0],
        [0.0, 64.0]
    ])
    R = np.array([[100.0]])

    # 3. Solve for the P matrix of the Value Function (V = x'Px)
    try:
        P = solve_continuous_are(A, B, Q, R)
        logging.info("Successfully solved the Continuous Algebraic Riccati Equation for P.")
    except Exception as e:
        logging.error(f"Failed to solve ARE: {e}")
        return

    def value_function(x):
        """Calculates the LQR cost V(x) = x'Px."""
        return x.T @ P @ x

    # 4. Set up the 3D Plot
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')

    # Create the cost function surface
    pos_range = np.max(np.abs(states[:, 0])) * 1.2
    vel_range = np.max(np.abs(states[:, 1])) * 1.2
    x1_grid = np.linspace(-pos_range, pos_range, 100)
    x2_grid = np.linspace(-vel_range, vel_range, 100)
    X1, X2 = np.meshgrid(x1_grid, x2_grid)

    V = np.zeros(X1.shape)
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            x_grid = np.array([X1[i, j], X2[i, j]])
            V[i, j] = value_function(x_grid)

    surf = ax.plot_surface(X1, X2, V, cmap='viridis', alpha=0.6, edgecolor='none')

    # Plot static elements
    origin_star, = ax.plot([0], [0], [0], '*', color='purple', markersize=15, label="Desired State (Origin)")

    # Initialize animated elements
    state_point, = ax.plot([], [], [], 'o', color='red', markersize=8, label="Current State")
    breadcrumb_line, = ax.plot([], [], [], '-', color='red', linewidth=2, alpha=0.7, label="Trajectory")
    time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

    # 5. Configure Plot Appearance
    ax.set_xlabel("Position (rad)")
    ax.set_ylabel("Velocity (rad/s)")
    ax.set_zlabel("Cost V(x)")
    ax.set_title("LQR Cost Landscape and State Trajectory")
    fig.colorbar(surf, shrink=0.5, aspect=5, label="Cost")
    ax.legend(loc="upper right")

    # 6. Animation Setup
    global_min_time = sim_times[0]
    global_max_time = sim_times[-1]
    sim_duration = global_max_time - global_min_time
    total_frames = int(np.ceil(sim_duration * fps))

    logging.info(f"Simulation duration: {sim_duration:.2f}s, Total frames: {total_frames}")

    def init():
        """Initializes the animation."""
        state_point.set_data_3d([], [], [])
        breadcrumb_line.set_data_3d([], [], [])
        time_text.set_text('')
        return [state_point, breadcrumb_line, time_text]

    def update(frame):
        """Updates the animation for each frame."""
        current_time = global_min_time + (frame / fps)
        idx = np.searchsorted(sim_times, current_time, side="right") - 1
        idx = max(0, idx)

        start_time_thresh = current_time - 10.0
        start_idx = np.searchsorted(sim_times, max(global_min_time, start_time_thresh), side="left")
        
        history_states = states[start_idx:idx+1]
        if len(history_states) > 0:
            history_pos = history_states[:, 0]
            history_vel = history_states[:, 1]
            history_cost = np.array([value_function(st) for st in history_states])
            breadcrumb_line.set_data_3d(history_pos, history_vel, history_cost)
        else:
            breadcrumb_line.set_data_3d([], [], [])

        current_state = states[idx]
        current_pos, current_vel = current_state[0], current_state[1]
        current_cost = value_function(current_state)

        state_point.set_data_3d([current_pos], [current_vel], [current_cost])
        time_text.set_text(f'Time: {current_time:.2f}s')

        return [state_point, breadcrumb_line, time_text]

    # 7. Create and Save the Animation
    ani = animation.FuncAnimation(
        fig, update, frames=total_frames, init_func=init, blit=True, interval=1000/fps
    )

    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=1800)

    output_full_path = os.path.join(output_dir, output_filename)
    os.makedirs(output_dir, exist_ok=True)

    progress_bar = tqdm(total=total_frames, desc="Rendering Video", unit="frame")
    try:
        logging.info(f"Saving animation to '{output_full_path}'...")
        ani.save(
            output_full_path,
            writer=writer,
            dpi=dpi,
            progress_callback=lambda i, n: progress_bar.update(1)
        )
    except FileNotFoundError:
        logging.error("FFmpeg not found. Please install FFmpeg and ensure it's in your system's PATH.")
    except Exception as e:
        logging.error(f"An error occurred during animation saving: {e}")
    finally:
        progress_bar.close()
        plt.close(fig)

    logging.info("Animation processing finished.")


if __name__ == '__main__':
    OUTPUT_PARENT_DIR_NAME = "output"
    OUTPUT_DATA_FILENAME = "states.csv"
    
    csv_file_path = os.path.join(OUTPUT_PARENT_DIR_NAME, OUTPUT_DATA_FILENAME)

    create_lqr_animation(
        csv_path=csv_file_path,
        output_dir=OUTPUT_PARENT_DIR_NAME
    )
