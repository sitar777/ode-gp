import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils import plot_graph


class PDE:
    """Container class for PDE"""

    initial_condition: list
    x_range: list
    t_range: list
    unary_operators: list[str]
    points_number_x: int | None = 100
    points_number_t: int | None = 100
    binary_operators: list[str] = ["+", "*"]

    @property
    def x_vals(self):
        return np.linspace(self.x_range[0], self.x_range[1], self.points_number_x)

    @property
    def t_vals(self):
        return np.linspace(self.t_range[0], self.t_range[1], self.points_number_t)

    @property
    def feature_matrix(self):
        x, t = np.meshgrid(self.x_vals, self.t_vals)
        return np.column_stack([x.ravel(), t.ravel()])

    @property
    def target(self):
        return self.numerical_solution.ravel()

    def initial_condition(self, x):
        raise NotImplementedError

    @property
    def numerical_solution(self):
        raise NotImplementedError

    def exact_solution(self, x, t):
        raise NotImplementedError

    def plot_results(self, y_prediction, sympy_expression):
        fig = plt.figure(figsize=(12, 5))
        x, t = np.meshgrid(self.x_vals, self.t_vals)

        ax1 = fig.add_subplot(121, projection='3d')
        ax1.plot_surface(x, t, y_prediction, cmap='plasma')
        ax1.set_xlabel('x')
        ax1.set_ylabel('t')
        ax1.set_zlabel('u')
        ax1.set_title(f'Predicted: {sympy_expression}')

        try:
            u_exact = self.exact_solution(x, t)
            mse = mean_squared_error(u_exact.ravel(), y_prediction.ravel())
            fig.suptitle(f'MSE (predicted vs exact): {mse:.4g}', y=0.97)

            ax2 = fig.add_subplot(122, projection='3d')
            ax2.plot_surface(x, t, u_exact, cmap='coolwarm')
            ax2.set_xlabel('x')
            ax2.set_ylabel('t')
            ax2.set_zlabel('u')
            ax2.set_title('Exact solution')
        except NotImplementedError:
            pass

        plt.tight_layout()
        plt.show()

        # Plotting tree
        plt.figure(2)
        plot_graph(sympy_expression)
        plt.show()
