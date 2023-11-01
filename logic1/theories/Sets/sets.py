from __future__ import annotations


import logging
import sympy
from typing import TypeAlias

from logic1 import atomlib
from logic1.firstorder.boolean import T, F
from logic1.support.decorators import classproperty

logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


def show_progress(flag: bool = True) -> None:
    if flag:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.CRITICAL)


Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol


class TermMixin():

    @staticmethod
    def term_type() -> type[Term]:
        return Term

    @staticmethod
    def variable_type() -> type[Variable]:
        return Variable


class Eq(TermMixin, atomlib.sympy.Eq):
    """Equations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> Eq(x, y)
    Eq(x, y)

    >>> Eq(x, 0)
    Traceback (most recent call last):
    ...
    ValueError: 0 is not a Term

    >>> Eq(x + x, y)
    Traceback (most recent call last):
    ...
    ValueError: 2*x is not a Term
    """

    # Class variables
    func: type[Eq]

    @classproperty
    def complement_func(cls):
        """The complement relation Ne of Eq.
        """
        return Ne

    @classproperty
    def converse_func(cls):
        """The converse relation Eq of Eq.
        """
        return Eq

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, Variable):
                raise ValueError(f"{arg!r} is not a Term")
        super().__init__(*args)

    def simplify(self):
        c = self.lhs.compare(self.rhs)
        if c == 0:
            return T
        if c == 1:
            return Eq(self.rhs, self.lhs)
        assert c == -1
        return self


class Ne(TermMixin, atomlib.sympy.Ne):
    """Inequations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> Ne(y, x)
    Ne(y, x)

    >>> Ne(x, y + 1)
    Traceback (most recent call last):
    ...
    ValueError: y + 1 is not a Term
    """

    # Class variables
    func: type[Ne]

    @classproperty
    def complement_func(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classproperty
    def converse_func(cls):
        """The converse relation Me of Ne.
        """
        return Ne

    def __init__(self, *args):
        for arg in args:
            if not isinstance(arg, Variable):
                raise ValueError(f"{arg!r} is not a Term")
        super().__init__(*args)

    def simplify(self):
        c = self.lhs.compare(self.rhs)
        if c == 0:
            return F
        if c == 1:
            return Ne(self.rhs, self.lhs)
        assert c == -1
        return self


oo = atomlib.sympy.oo


class C(atomlib.sympy.C):

    # Class variables
    @classproperty
    def complement_func(cls):
        return C_


class C_(atomlib.sympy.C_):

    # Class variables
    @classproperty
    def complement_func(cls):
        return C
