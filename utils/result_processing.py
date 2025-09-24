"""Result processing utils"""

from sympy import Number, Float


def round_expr(expr, num_digits, skip=True):
    """Round constants in expression"""
    if skip:
        return expr
    return expr.xreplace({n: n.evalf(num_digits) for n in expr.atoms(Number)})
