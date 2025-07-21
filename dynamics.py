import numpy as np
import math

import scipy.integrate
from control import lqr
from typing import Callable


def my_ODE_1(
    t: float,
    y: np.ndarray
) -> np.ndarray:
    a = -1
    dy_dt = a * y
    return dy_dt


def P_controller_CLTI_dynamics(
    t: float,
    x: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    u = -K @ x
    x_dot = A @ x + B @ u
    return x_dot


def CLTI_LQR_gain(
    A: np.ndarray,
    B: np.ndarray,
    Q: np.ndarray,
    R: np.ndarray,
    N: np.ndarray | None = None
) -> np.ndarray:
    if N is not None:
        K, _, _ = lqr(A, B, Q, R, N)
    else:
        K, _, _ = lqr(A, B, Q, R)
    return K


def integrate_first_order_ode(
    current_time: float,
    current_state: np.ndarray,
    time_step_delta: float,
    derivative_func: Callable[[float, np.ndarray], np.ndarray],
    solver: str
) -> scipy.integrate.OdeSolution | None:
    original_shape = np.shape(a=current_state)
    y0 = np.asarray(a=current_state).ravel()

    def wrapped_derivative(
        t: float,
        y: np.ndarray
    ) -> np.ndarray:
        y_reshaped = y.reshape(original_shape)
        dy_dt = derivative_func(
            t,
            y_reshaped
        )
        return np.asarray(a=dy_dt).ravel()

    t_span = [current_time, current_time + time_step_delta]
    solution = scipy.integrate.solve_ivp(
        fun=wrapped_derivative,
        t_span=t_span,
        y0=y0,
        method=solver
    )
    solution.y = solution.y[:, -1].reshape(original_shape)
    return solution

#         if not current_solution.success:
#             raise ValueError(f"The ODE solver has encountered an error at simulation time {current_time}.")
        # current_solution: scipy.integrate.OdeSolution = \
        # dynamics.integrate_first_order_ode(
        #     current_time=current_time,
        #     current_state=self.position,
        #     time_step_delta=self.config.sim_time_step_size_seconds,
        #     derivative_func=self.dynamics,
        #     solver=self.config.numeric_solver
        # )

def rk4_solver(
    current_time: float,
    current_state: np.ndarray,  # Assumed 1-D, so flatened already
    integration_time_window: list[float, float],
    time_step_size: float,
    ode_func: Callable[[float, np.ndarray], np.ndarray] # Assumed output is 1-D
) -> np.ndarray:
    t1, t2 = integration_time_window
    num_time_steps = math.ceil((t2 - t1) / time_step_size)
    
    state_data = np.zeros((num_time_steps + 1, current_state.shape[0]))
    state_data[0, :] = current_state
    t = current_time
    dt = time_step_size

    for step in range(num_time_steps):
        x = state_data[step, :]

        k_1 = ode_func(t=t,          x=x                 )
        k_2 = ode_func(t=t + dt / 2, x=x + k_1 * (dt / 2)) 
        k_3 = ode_func(t=t + dt / 2, x=x + k_2 * (dt / 2))
        k_4 = ode_func(t=t + dt,     x=x + k_3 * dt      )

        state_data[step + 1, :] = x + (1 / 6) * dt * (k_1 + 2 * k_2 + 2 * k_3 + k_4) 

    return state_data[-1, :]
