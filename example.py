import numpy as np

import matplotlib.pyplot as plt
from pysr import PySRRegressor
from sympy import simplify

from utils import plot_graph, solve_ode, ODE


def exact_solution_1(x):
    """we know the exact solution"""
    return 0.2*x**2 + 2*np.cos(4*x) - 2


# TODO function to class
def sample_function_1(x, y) -> np.ndarray:
    """example ODE to solve"""
    return np.array(
        [
            y[1],
            -y[1] - y[0] - 1.6 + 0.4*x + 0.2*x**2 - 8*np.sin(4*x) - 30*np.cos(4*x),
        ]
    )


ode_1 = ODE(
    function=sample_function_1,
    initial_condition=[0, 0],
    x_range=[0, 8],
    exact_solution=exact_solution_1,
    unary_operators=['sin', 'cos']
)


def exact_solution_2(x):
    """we know the exact solution"""
    return 1 + 0.4*x + np.sin(5*x)*np.exp(-0.3*x)


def sample_function_2(x, y):
    """example ODE to solve"""
    return np.array(
        [
            y[1],
            -0.3*y[1] - 25*y[0] + 25.12 + 10*x - 1.5*np.cos(5*x)*np.exp(-0.3*x),
        ]
    )


ode_2 = ODE(
    function=sample_function_2,
    initial_condition=[1, 5.4],
    x_range=[0, 8],
    exact_solution=exact_solution_2,
    unary_operators=['sin', 'cos', 'exp'],
)


def exact_solution_3(x):
    """we know the exact solution"""
    return 3*np.cos(2*np.log(x)) + 2*np.sin(np.log(x))


def sample_function_3(x, y):
    """example ODE to solve"""
    return np.array(
        [
            y[1],
            (6*np.sin(np.log(x)) - x*y[1] - 4*y[0])/x**2,
        ]
    )


ode_3 = ODE(
    function=sample_function_3,
    initial_condition=[3, 2],
    x_range=[1, 45],
    exact_solution=exact_solution_3,
    unary_operators=['sin', 'cos', 'log'],
)


if __name__ == '__main__':
    ode = ode_2

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
