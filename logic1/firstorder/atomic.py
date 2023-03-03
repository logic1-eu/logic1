from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, final
from typing_extensions import Self

import pyeda.inter  # type: ignore
import sympy

from .formula import Formula
from .quantified import Ex, All
from ..support.containers import GetVars
from ..support.decorators import classproperty


class AtomicFormula(Formula):

    # Class variables
    latex_symbol_spacing = ' '
    print_precedence = 99
    text_symbol_spacing = ' '

    @classproperty
    def func(cls):
        return cls

    complement_func: type[AtomicFormula]

    # Instance variables
    args: tuple

    # Static methods on terms
    @staticmethod
    @abstractmethod
    def term_type() -> Any:
        ...

    @staticmethod
    @abstractmethod
    def term_get_vars(term: Any) -> set:
        ...

    @staticmethod
    @abstractmethod
    def term_to_latex(term: Any) -> str:
        ...

    @staticmethod
    @abstractmethod
    def term_to_sympy(term: Any) -> sympy.Basic:
        ...

    # Static methods on variables
    @staticmethod
    @abstractmethod
    def variable_type() -> Any:
        ...

    @staticmethod
    @abstractmethod
    def rename_var(variable: Any) -> Any:
        ...

    # Instance methods
    def __init__(self, *args) -> None:
        self.args = args

    @final
    def _count_alternations(self) -> tuple[int, set]:
        return (-1, {Ex, All})

    @final
    def get_any_atom(self) -> Self:
        return self

    @final
    def get_qvars(self) -> set:
        return set()

    @abstractmethod
    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        ...

    @abstractmethod
    def _sprint(self, mode: str) -> str:
        ...

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        ...

    @final
    def to_cnf(self) -> Self:
        """ Convert to Conjunctive Normal Form.

        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import a, b
        >>> Eq(a, 0).to_cnf()
        Eq(a, 0)
        """
        return self

    @final
    def to_complement(self, conditional: bool = True) -> AtomicFormula:
        # Do not pass on but check conditional in order to avoid construction
        # in case of False.
        if conditional:
            return self.complement_func(*self.args)
        return self

    @final
    def _to_distinct_vars(self, badlist: set) -> Self:
        return self

    @final
    def to_dnf(self) -> Self:
        """ Convert to Disjunctive Normal Form.

        >>> from logic1.atomlib.sympy import Eq
        >>> from sympy.abc import a, b
        >>> Eq(a, 0).to_dnf()
        Eq(a, 0)
        """
        return self

    def to_nnf(self, implicit_not: bool = False, to_positive: bool = True) \
            -> Formula:
        return self.to_complement() if implicit_not else self

    @final
    def _to_pnf(self) -> dict:
        """Prenex normal form. self must be in negation normal form.
        """
        return {Ex: self, All: self}

    @final
    def _to_pyeda(self, d: dict, c: list = [0]) -> pyeda.boolalg.expr:
        if self in d:
            return d[self]
        if self.to_complement() in d:
            return pyeda.boolalg.expr.Not(d[self.to_complement()])
        d[self] = pyeda.boolalg.expr.exprvar('a', c[0])
        c[0] += 1
        return d[self]

    @final
    def to_sympy(self, **kwargs) -> sympy.Basic:
        """Override Formula.to_sympy() to prevent recursion into terms
        """
        sympy_terms = (self.__class__.term_to_sympy(arg) for arg in self.args)
        return self.__class__.sympy_func(*sympy_terms, **kwargs)

    @final
    def transform_atoms(self, transformation: Callable) -> Self:
        return transformation(self)
