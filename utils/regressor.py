"""Utils for regression"""
from typing import TYPE_CHECKING

from pysr import PySRRegressor

if TYPE_CHECKING:
    from .ode import ODE
    from .pde import PDE


def configure_regressor(problem: 'ODE | PDE', optimal_complexity=15) -> PySRRegressor:
    """Configuring regressor"""

    # Info about regressor parameters
    # https://astroautomata.com/PySR/api/#pysrregressor-parameters
    return PySRRegressor(
        niterations=200,
        binary_operators=problem.binary_operators,
        unary_operators=problem.unary_operators,
        populations=300,
        model_selection="best",
        early_stop_condition=(
            f"stop_if(loss, complexity) = loss < 1e-6 && complexity < {optimal_complexity}"
            # Stop early if we find a good and simple equation
        ),
    )
