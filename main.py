"""Main script"""

from pysr import PySRRegressor
from sympy import simplify

from utils import round_expr
from examples import ode_1 as ode


if __name__ == '__main__':
    # Info about regressor parameters
    # https://astroautomata.com/PySR/api/#pysrregressor-parameters
    model = PySRRegressor(
        niterations=200,
        binary_operators=["+", "*"],
        unary_operators=ode.unary_operators,
        populations=300,
        model_selection="best",
        early_stop_condition=(
            "stop_if(loss, complexity) = loss < 1e-6 && complexity < 15"
            # Stop early if we find a good and simple equation
        ),
    )

    y_sol = ode.solution
    # Model training
    model.fit(ode.x_vals.reshape(-1, 1), y_sol)

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
