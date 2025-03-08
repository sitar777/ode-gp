import numpy as np

import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import simplify

from utils import plot_graph, solve_ode


def exact_solution(x):
    """we know the exact solution"""
    return 0.2*x**2 + 2*np.cos(4*x) - 2


def sample_function(x, y) -> np.ndarray:
    """example ODE to solve"""
    return np.array(
        [
            -y[0] - y[1] - 1.6 + 0.4*x + 0.2*x**2 - 8*np.sin(4*x) - 30*np.cos(4*x),
            y[0],
        ]
    )

if __name__ == '__main__':
    y0 = [0, 0]
    x_vals = np.linspace(0, 8, 100)
    y_sol = solve_ode(sample_function, x_vals, y0)

    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "*"],
        unary_operators=["cos", "sin"],
        populations=200,
        model_selection="best",
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 20"
            # Stop early if we find a good and simple equation
        ),
    )

    model.fit(x_vals.reshape(-1, 1), y_sol)
    print(model.sympy())

    best_idx = model.equations_.query(
        f"loss < {2 * model.equations_.loss.min()}"
    ).score.idxmax()

    y_prediction = model.predict(x_vals.reshape(-1, 1), index=best_idx)

    solution_plot = plt.figure(1)
    plt.plot(x_vals, y_sol, 'g', label='y(x)_num')
    plt.plot(x_vals, exact_solution(x_vals), 'r', label='y(x)_exact')
    plt.plot(x_vals, y_prediction, 'b', label=simplify(model.sympy()))
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.grid()

    graph_plot = plt.figure(2)
    plot_graph(model.sympy())

    plt.show()
