from __future__ import annotations

from typing import Tuple

import sympy

from ..firstorder import atomic
from ..support.containers import GetVars
from ..support.renaming import rename


Term = sympy.Expr
Variable = sympy.Symbol


class TermMixin():

    @staticmethod
    def term_type() -> type[Term]:
        return Term

    @staticmethod
    def term_get_vars(term: Term) -> set[Variable]:
        return sympy.S(term).atoms(Variable)

    @staticmethod
    def term_to_latex(term: Term) -> str:
        return sympy.latex(term)

    @staticmethod
    def variable_type() -> type[Variable]:
        return Variable

    @staticmethod
    def rename_var(variable: Variable) -> Variable:
        return rename(variable)


class AtomicFormula(TermMixin, atomic.AtomicFormula):
    """Atomic Formula with Sympy Terms. All terms are sympy.Expr.
    """

    args: Tuple[Term, ...]

    @classmethod
    def interactive_new(cls, *args) -> Self:
        args_ = []
        for arg in args:
            arg_ = (sympy.Integer(arg) if isinstance(arg, int) else arg)
            if not isinstance(arg_, cls.term_type()):
                raise TypeError(f"{arg} is not a Term")
            args_.append(arg_)
        return cls(*args_)

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return GetVars(free=all_vars - assume_quantified,
                         bound=all_vars & assume_quantified)

    def subs(self, substitution: dict) -> Self:
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(AtomicFormula):

    args: Tuple[Term, Term]

    @property
    def lhs(self) -> Term:
        """The left-hand side of the BinaryAtomicFormula."""
        return self.args[0]

    @property
    def rhs(self) -> Term:
        """The right-hand side of the BinaryAtomicFormula."""
        return self.args[1]

    def __init__(self, lhs: Term, rhs: Term) -> None:
        self.func = self.__class__
        self.args = (lhs, rhs)

    # Override BooleanFormula._sprint() to prevent recursion into terms
    def _sprint(self, mode: str) -> str:
        if mode == 'latex':
            symbol = self.__class__.latex_symbol
            lhs = sympy.latex(self.lhs)
            rhs = sympy.latex(self.rhs)
            spacing = self.__class__.latex_symbol_spacing
        else:
            assert mode == 'text'
            symbol = self.__class__.text_symbol
            lhs = self.lhs.__str__()
            rhs = self.rhs.__str__()
            spacing = self.__class__.text_symbol_spacing
        return f'{lhs}{spacing}{symbol}{spacing}{rhs}'


class Eq(BinaryAtomicFormula):
    """
    >>> from sympy import exp, I, pi
    >>> Eq(exp(I * pi, evaluate=False), -1)
    Eq(exp(I*pi), -1)
    """
    text_symbol = '='
    latex_symbol = '='

    sympy_func = sympy.Eq

    @staticmethod
    def to_complementary(conditional: bool = True) \
            -> type[BinaryAtomicFormula]:
        if conditional:
            return Ne
        return Eq

    @staticmethod
    def to_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Eq
        return Eq


EQ = Eq.interactive_new


class Ne(BinaryAtomicFormula):
    """
    >>> Ne(1, 0)
    Ne(1, 0)
    """
    text_symbol = '!='
    latex_symbol = '\\neq'

    sympy_func = sympy.Ne

    @staticmethod
    def to_complementary(conditional: bool = True) \
            -> type[BinaryAtomicFormula]:
        if conditional:
            return Eq
        return Ne

    @staticmethod
    def to_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ne
        return Ne


NE = Ne.interactive_new


class Ge(BinaryAtomicFormula):

    text_symbol = '>='
    latex_symbol = '\\geq'

    sympy_func = sympy.Ge

    @staticmethod
    def to_complementary(conditional: bool = True) \
            -> type[BinaryAtomicFormula]:
        if conditional:
            return Lt
        return Ge

    @staticmethod
    def to_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Le
        return Ge


GE = Ge.interactive_new


class Le(BinaryAtomicFormula):

    text_symbol = '<='
    latex_symbol = '\\leq'

    sympy_func = sympy.Le

    @staticmethod
    def to_complementary(conditional: bool = True) \
            -> type[BinaryAtomicFormula]:
        if conditional:
            return Gt
        return Le

    @staticmethod
    def to_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Ge
        return Le


LE = Le.interactive_new


class Gt(BinaryAtomicFormula):

    text_symbol = '>'
    latex_symbol = '>'

    sympy_func = sympy.Gt

    @staticmethod
    def to_complementary(conditional: bool = True) \
            -> type[BinaryAtomicFormula]:
        if conditional:
            return Le
        return Gt

    @staticmethod
    def to_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Lt
        return Gt


GT = Gt.interactive_new


class Lt(BinaryAtomicFormula):

    text_symbol = '<'
    latex_symbol = '<'

    sympy_func = sympy.Lt

    @staticmethod
    def to_complementary(conditional: bool = True) \
            -> type[BinaryAtomicFormula]:
        if conditional:
            return Ge
        return Lt

    @staticmethod
    def to_dual(conditional: bool = True) -> type[BinaryAtomicFormula]:
        if conditional:
            return Gt
        return Lt


LT = Lt.interactive_new
