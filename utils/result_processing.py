"""Result processing utils"""

from sympy import Number


def round_expr(expr, num_digits, skip=True):
    """Round constants in expression"""
    if skip:
        return expr
    return expr.xreplace({n : round(n, num_digits) for n in expr.atoms(Number)})
