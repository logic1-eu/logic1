from __future__ import annotations

from abc import ABCMeta, abstractmethod
import sympy
from typing import Final, TypeAlias

from .. import firstorder
from ..support.renaming import rename
from . import generic

Index: TypeAlias = sympy.Integer | sympy.core.numbers.Infinity
Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol

oo: Final = sympy.oo


class TermMixin:

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

    def __le__(self, other: firstorder.Formula) -> bool:
        match other:
            case AtomicFormula():
                if len(self.args) != len(other.args):
                    return len(self.args) < len(other.args)
                for i in range(len(self.args)):
                    if self.args[i] != other.args[i]:
                        return self.args[i].sort_key() <= other.args[i].sort_key()  # type: ignore
                L = self.relations()
                return L.index(self.op) <= L.index(other.op)
            case _:
                return True

    @abstractmethod
    def relations(self) -> list[ABCMeta]:
        ...

    def subs(self, substitution: dict) -> AtomicFormula:
        """Implements the abstract method :meth:`.firstorder.Formula.subs`.
        """
        args = (arg.subs(substitution, simultaneous=True) for arg in self.args)  # type: ignore
        return self.op(*args)


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
