from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, final

import pyeda.inter  # type: ignore

from .boolean import Not
from .formula import Formula
from .quantified import Ex, All
from ..support.containers import GetVars


class AtomicFormula(Formula):

    print_precedence = 99

    latex_symbol_spacing = ' '
    text_symbol_spacing = ' '

    func: type[AtomicFormula]

    # Terms
    @staticmethod
    @abstractmethod
    def term_type():
        ...

    @staticmethod
    @abstractmethod
    def term_get_vars(term: Any) -> set:
        ...

    @staticmethod
    @abstractmethod
    def term_to_latex(term: Any) -> str:
        ...

    # Variables
    @staticmethod
    @abstractmethod
    def variable_type():
        ...

    @staticmethod
    @abstractmethod
    def rename_var(variable: Any) -> Any:
        ...

    # Relations
    @staticmethod
    @abstractmethod
    def to_complementary(conditional: bool = True) -> type[AtomicFormula]:
        ...

    @staticmethod
    @abstractmethod
    def to_dual(conditional: bool = True) -> type[AtomicFormula]:
        ...

    # Instance methods
    @final
    def _count_alternations(self) -> tuple:
        return (-1, {Ex, All})

    @final
    def get_any_atom(self) -> AtomicFormula:
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

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b
        >>> EQ(a, 0).to_cnf()
        Eq(a, 0)
        """
        return self

    @final
    def to_complement(self) -> Self:
        return self.func.to_complementary()(*self.args)

    @final
    def _to_distinct_vars(self, badlist: set) -> Self:
        return self

    @final
    def to_dnf(self) -> Self:
        """ Convert to Disjunctive Normal Form.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b
        >>> EQ(a, 0).to_dnf()
        Eq(a, 0)
        """
        return self

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
        if implicit_not:
            if to_positive:
                try:
                    tmp = self.func.to_complementary()(*self.args)
                except AttributeError:
                    pass
                else:
                    return tmp
            return Not(self)
        return self

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
    def to_sympy(self, **kwargs):
        """Override Formula.to_sympy() to prevent recursion into terms
        """
        return self.__class__.sympy_func(*self.args, **kwargs)

    @final
    def transform_atoms(self, transformation: Callable) -> Self:
        return transformation(self)
