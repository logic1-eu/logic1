from __future__ import annotations

import sympy

from typing import Any, Final, TypeAlias, Union

from .. import firstorder
from ..support.containers import GetVars
from ..support.renaming import rename
from . import generic

Card: TypeAlias = Union[int, sympy.core.numbers.Infinity]

Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol

oo: Final = sympy.oo


class TermMixin:

    @staticmethod
    def term_get_vars(term: Term) -> set[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_get_vars`.
        """
        return sympy.S(term).atoms(Variable)

    @staticmethod
    def term_to_latex(term: Term) -> str:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_to_latex`.
        """
        return sympy.latex(term)

    @staticmethod
    def variable_type() -> type[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.variable_type`.
        """
        return Variable

    @staticmethod
    def rename_var(variable: Variable) -> Variable:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.rename_var`.
        """
        return rename(variable)


class AtomicFormula(TermMixin, firstorder.AtomicFormula):
    """Atomic Formula with Sympy Terms. All terms are :class:`sympy.Expr`.
    """

    # Instance methods
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        all_vars = set()
        for term in self.args:
            all_vars |= term.atoms(sympy.Symbol)
        return GetVars(free=all_vars - assume_quantified,
                       bound=all_vars & assume_quantified)

    def subs(self, substitution: dict) -> AtomicFormula:
        """Implements the abstract method :meth:`.firstorder.Formula.subs`.
        """
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(generic.BinaryAtomicFormulaMixin, AtomicFormula):
    """A class whose instances are binary formulas in the sense that both
    their m-arity and their p-arity is 2.
    """

    def __init__(self, *args):
        if len(args) != 2:
            raise ValueError(f'bad number of arguments for binary relation')
        args_ = []
        for arg in args:
            arg_ = sympy.Integer(arg) if isinstance(arg, int) else arg
            if not isinstance(arg_, Term):
                raise ValueError(f"{arg!r} is not a Term")
            args_.append(arg_)
        super().__init__(*args_)

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


class IndexedConstantAtomicFormula(AtomicFormula):
    r"""A class whose instances form a family of atomic formulas with m-arity
    0. Their p-arity is 1, where the one argument of the constructor is the
    index.
    """
    @property
    def index(self) -> Any:
        """The index of the :class:`IndexedConstantAtomicFormula`.
        """
        return self.args[0]

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        return GetVars()
