"""Utils for regression"""
from typing import TYPE_CHECKING

from pysr import PySRRegressor

if TYPE_CHECKING:
    from .ode import ODE


def configure_regressor(ode: 'ODE') -> PySRRegressor:
    """Configuring regressor"""

    # Info about regressor parameters
    # https://astroautomata.com/PySR/api/#pysrregressor-parameters
    return PySRRegressor(
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
