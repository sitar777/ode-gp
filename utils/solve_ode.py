"""Utility for solving ODE"""

from scipy.integrate import solve_ivp


def solve_ode(sample_function: callable, x: list, y0: list):
    """ODE solver"""

    # TODO to ODE class
    num_sol = solve_ivp(sample_function, [x[0], x[-1]], y0, t_eval=x)
    return num_sol.y[0, :]
