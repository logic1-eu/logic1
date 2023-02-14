"""We are preliminarily using sympy expressions as Terms and sympy symbols as
   variables without introducing own classes. We collect a few helpers here.
"""


import sympy
from typing import Any

Variable = sympy.Symbol


def is_constant(term: Any) -> bool:
    """
    >>> is_constant(1)
    True
    >>> is_constant(sympy.S('1'))
    True
    >>> is_constant(sympy.S('x'))
    False
    >>> is_constant(sympy.S('x + 1'))
    False
    """
    return isinstance(term, (int, sympy.Integer))


def is_term(x: Any) -> bool:
    """
    >>> is_term(1)
    True
    >>> is_term(sympy.S('1'))
    True
    >>> is_term(sympy.S('x'))
    True
    >>> is_term(sympy.S('x + 1'))
    True
    """
    return isinstance(x, (int, sympy.Expr))


def is_variable(term: Any) -> bool:
    """
    >>> is_variable(1)
    False
    >>> is_variable(sympy.S('1'))
    False
    >>> is_variable(sympy.S('x'))
    True
    >>> is_variable(sympy.S('x + 1'))
    False
    """
    return isinstance(term, Variable)
