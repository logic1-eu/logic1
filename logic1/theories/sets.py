from __future__ import annotations

import sympy

from ..atomlib import atomic
from ..firstorder import *


Term = sympy.Symbol
Variable = sympy.Symbol


class TermMixin():

    @staticmethod
    def term_type() -> type[Term]:
        return Term

    @staticmethod
    def variable_type() -> type[Variable]:
        return Variable


class Eq(TermMixin, atomic.Eq):
    """Equations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> EQ(x, y)
    Eq(x, y)

    >>> EQ(x, 0)
    Traceback (most recent call last):
    ...
    TypeError: 0 is not a Term

    >>> EQ(x + x, y)
    Traceback (most recent call last):
    ...
    TypeError: 2*x is not a Term
    """


EQ = Eq.interactive_new


class Ne(TermMixin, atomic.Ne):
    """Inequations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> NE(y, x)
    Ne(y, x)

    >>> NE(x, y + 1)
    Traceback (most recent call last):
    ...
    TypeError: y + 1 is not a Term
    """


NE = Ne.interactive_new


def qe(f: Formula) -> BooleanFormula:
    pass
