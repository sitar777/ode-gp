"""Main script"""

import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import simplify

from utils import plot_graph, solve_ode
from examples import ode_2 as ode


if __name__ == '__main__':
    y_sol = solve_ode(ode.function, ode.x_vals, ode.initial_condition)

    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "*"],
        unary_operators=ode.unary_operators,
        populations=300,
        model_selection="best",
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 20"
            # Stop early if we find a good and simple equation
        ),
    )

    model.fit(ode.x_vals.reshape(-1, 1), y_sol)

    best_idx = model.equations_.query(
        f"loss < {2 * model.equations_.loss.min()}"
    ).score.idxmax()

    y_prediction = model.predict(ode.x_vals.reshape(-1, 1), index=best_idx)

    sympy_simplified = simplify(model.sympy())

    # Plotting 2 plots simultneously
    solution_plot = plt.figure(1)
    plt.plot(ode.x_vals, y_sol, 'g', label='y(x)_num')
    plt.plot(ode.x_vals, ode.exact_solution(ode.x_vals), 'r', label='y(x)_exact')
    plt.plot(ode.x_vals, y_prediction, 'b', label=sympy_simplified)
    plt.legend(loc='best')
    plt.xlabel('x')
    plt.grid()

    graph_plot = plt.figure(2)
    plot_graph(sympy_simplified)

    plt.show()
