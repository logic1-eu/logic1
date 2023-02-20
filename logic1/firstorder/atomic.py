from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, final

import pyeda.inter  # type: ignore

from .formula import Formula, Ex, All, Not
from ..support.containers import Variables


class AtomicFormula(Formula):

    print_precedence = 99
    text_symbol_spacing = ' '
    latex_symbol_spacing = ' '

    func: type[AtomicFormula]

    @staticmethod
    @abstractmethod
    def get_variables_from_term(term: Any) -> set:
        ...

    @staticmethod
    @abstractmethod
    def rename_variable(variable: Any) -> Any:
        ...

    @staticmethod
    @abstractmethod
    def term_to_latex(term: Any) -> str:
        ...

    @staticmethod
    @abstractmethod
    def term_type():
        ...

    @staticmethod
    @abstractmethod
    def to_complementary(conditional: bool = True) -> type[AtomicFormula]:
        ...

    @staticmethod
    @abstractmethod
    def to_dual(conditional: bool = True) -> type[AtomicFormula]:
        ...

    @staticmethod
    @abstractmethod
    def variable_type():
        ...

    @final
    def _count_alternations(self) -> tuple:
        return (-1, {Ex, All})

    @final
    def get_any_atomic_formula(self) -> AtomicFormula:
        return self

    @final
    def qvars(self) -> set:
        return set()

    @final
    def sympy(self, **kwargs):
        """Override Formula.sympy() to prevent recursion into terms
        """
        return self.__class__.sympy_func(*self.args, **kwargs)

    @final
    def to_complement(self) -> Self:
        return self.func.to_complementary()(*self.args)

    @final
    def _to_distinct_vars(self, badlist: set) -> Self:
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
    def to_dnf(self) -> Self:
        """ Convert to Disjunctive Normal Form.

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b
        >>> EQ(a, 0).to_dnf()
        Eq(a, 0)
        """
        return self

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
    def transform_atoms(self, transformation: Callable) -> Self:
        return transformation(self)

    @abstractmethod
    def vars(self, assume_quantified: set = set()) -> Variables:
        ...
