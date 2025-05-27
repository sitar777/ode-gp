"""Main script"""

from pysr import PySRRegressor
from sympy import simplify

from utils import round_expr, configure_regressor
from examples import ode_1 as ode


if __name__ == '__main__':
    model = configure_regressor(ode)
    # Model training
    model.fit(ode.x_vals.reshape(-1, 1), ode.numerical_solution)

    # Best index search
    best_idx = model.equations_.query(
        f"loss < {2 * model.equations_.loss.min()}"
    ).score.idxmax()

    # Getting values predicted by regressor
    y_prediction = model.predict(ode.x_vals.reshape(-1, 1), index=best_idx)

    # Simplifying the solution representaion
    # https://docs.sympy.org/latest/modules/simplify/simplify.html#simplify
    sympy_simplified = round_expr(simplify(model.sympy()), 4, skip=False)

    # Plotting solution
    ode.plot_results(y_prediction, sympy_simplified)
