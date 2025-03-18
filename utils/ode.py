import numpy as np
from typing import Optional


class ODE:

    """Container class for ODE"""
    def __init__(
        self,
        function: callable,
        initial_condition: list,
        x_range: list,
        unary_operators: list[str],
        points_number: int = 100,
        exact_solution: Optional[callable] = None
    ):
        self.function: callable = function
        self.initial_condition: list = initial_condition
        self.exact_solution: callable = exact_solution
        self.unary_operators = unary_operators
        self.x_range: list = x_range
        self.points_number = points_number


    @property
    def x_vals(self):
        """Returns x values for range"""
        return np.linspace(self.x_range[0], self.x_range[1], self.points_number)
