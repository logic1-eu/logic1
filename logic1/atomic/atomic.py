from __future__ import annotations

import sympy

from ..formulas import formula
from ..formulas.containers import Variables

Term = sympy.Expr


class AtomicFormula(formula.AtomicFormula):
    """Atomic Formula with Sympy Terms. All terms are sympy.Expr.
    """

    @classmethod
    def interactive_new(cls, *args):
        args_ = []
        for arg in args:
            args_.append(sympy.Integer(arg) if isinstance(arg, int) else arg)
        return cls(*args_)

    def subs(self: Self, substitution: dict) -> Self:
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)

    def vars(self: Self, assume_quantified: set = set()) -> Variables:
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return Variables(free=all_vars - assume_quantified,
                         bound=all_vars & assume_quantified)


class BinaryAtomicFormula(AtomicFormula):
    @property
    def lhs(self: Self) -> Term:
        """The left-hand side of the BinaryAtomicFormula."""
        return self.args[0]

    @property
    def rhs(self: Self) -> Term:
        """The right-hand side of the BinaryAtomicFormula."""
        return self.args[1]

    # Override BooleanFormula._sprint() to prevent recursion into terms
    def _sprint(self: Self, mode: str) -> str:
        if mode == 'latex':
            symbol = self._latex_symbol
            lhs = sympy.latex(self.lhs)
            rhs = sympy.latex(self.rhs)
            spacing = self._latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self._text_symbol
            lhs = self.lhs.__str__()
            rhs = self.rhs.__str__()
            spacing = self._text_symbol_spacing
        return f'{lhs}{spacing}{symbol}{spacing}{rhs}'


class Eq(BinaryAtomicFormula):
    """
    >>> from sympy.abc import x
    >>> Eq(x, x)
    Eq(x, x)
    """
    _text_symbol = '='
    _latex_symbol = '='

    _sympy_func = sympy.Eq

    @staticmethod
    def dualize(conditional: bool = True):
        if conditional:
            return Ne
        return Eq

    def __init__(self, lhs, rhs):
        self.func = Eq
        self.args = (lhs, rhs)


EQ = Eq.interactive_new


class Ne(BinaryAtomicFormula):
    """
    >>> Ne(1, 0)
    Ne(1, 0)
    """
    _text_symbol = '!='
    _latex_symbol = '\\neq'

    _sympy_func = sympy.Ne

    @staticmethod
    def dualize(conditional: bool = True):
        if conditional:
            return Eq
        return Ne

    def __init__(self: Self, lhs: Term, rhs: Term) -> None:
        self.func = Ne
        self.args = (lhs, rhs)


NE = Ne.interactive_new


class Ge(BinaryAtomicFormula):

    _text_symbol = '>='
    _latex_symbol = '\\geq'

    _sympy_func = sympy.Ge

    def __init__(self: Self, lhs: Term, rhs: Term) -> None:
        self.func = Ge
        self.args = (lhs, rhs)


GE = Ge.interactive_new


class Le(BinaryAtomicFormula):

    _text_symbol = '<='
    _latex_symbol = '\\leq'

    _sympy_func = sympy.Le

    def __init__(self: Self, lhs: Term, rhs: Term) -> None:
        self.func = Le
        self.args = (lhs, rhs)


LE = Le.interactive_new


class Gt(BinaryAtomicFormula):

    _text_symbol = '>'
    _latex_symbol = '>'

    _sympy_func = sympy.Gt

    def __init__(self: Self, lhs: Term, rhs: Term) -> None:
        self.func = Gt
        self.args = (lhs, rhs)


GT = Gt.interactive_new


class Lt(BinaryAtomicFormula):

    _text_symbol = '<'
    _latex_symbol = '<'

    _sympy_func = sympy.Lt

    def __init__(self: Self, lhs: Term, rhs: Term) -> None:
        self.func = Lt
        self.args = (lhs, rhs)


LT = Lt.interactive_new
