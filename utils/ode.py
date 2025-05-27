"""ODE container class"""

import numpy as np
from functools import cached_property
from utils import plot_graph
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error



class ODE:
    """Container class for ODE"""

    initial_condition: list
    x_range: list
    unary_operators: list[str]
    points_number: int | None = 100

    @cached_property
    def x_vals(self):
        """Returns x values for range"""
        return np.linspace(self.x_range[0], self.x_range[1], self.points_number)

    def function(self, x, y):
        """ODE to solve"""
        raise NotImplementedError

    def exact_solution(self, x):
        """Exact solution of ODE"""
        raise NotImplementedError

    @cached_property
    def solution(self):
        """ODE solution"""
        num_sol = solve_ivp(
            self.function,
            [self.x_vals[0], self.x_vals[-1]],
            self.initial_condition,
            t_eval=self.x_vals,
        )

        return num_sol.y[0, :]

    def plot_results(self, y_prediction, sympy_expression):
        """Fuction for displaying functions graphically"""

        # plotting function graph
        plt.figure(1)
        plt.scatter(self.x_vals, self.solution, label='y(x)_num', color='red')

        try:
            y_exact = self.exact_solution(self.x_vals)
            plt.plot(self.x_vals, y_exact, 'r', label='y(x)_exact')

            print(mean_squared_error(y_exact, y_prediction))
        except NotImplementedError:
            pass

        plt.plot(self.x_vals, y_prediction, 'b', label=sympy_expression)

        plt.legend(loc='best')
        plt.xlabel('x')
        plt.grid()

        # Plotting tree
        plt.figure(2)
        plot_graph(sympy_expression)
        plt.show()
