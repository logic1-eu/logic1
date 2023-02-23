from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, ClassVar, final
from typing_extensions import Self

import pyeda.inter  # type: ignore
import sympy

from .formula import Formula
from .quantified import Ex, All
from ..support.containers import GetVars


class AtomicFormula(Formula):

    latex_symbol_spacing: ClassVar[str] = ' '
    print_precedence: ClassVar[int] = 99
    text_symbol_spacing: ClassVar[str] = ' '

    func: type[AtomicFormula]

    # Terms
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

    # Variables
    @staticmethod
    @abstractmethod
    def variable_type() -> Any:
        ...

    @staticmethod
    @abstractmethod
    def rename_var(variable: Any) -> Any:
        ...

    # Relations
    @staticmethod
    @abstractmethod
    def rel_complement(conditional: bool = True) -> type[AtomicFormula]:
        """Returns the complement R' of a relation R derived from AtomicFormula
        if conditional is True, else R.

        Assume that R is defined on a Cartesian product P. Then R' = P - R. For
        instance,

        >>> from logic1.atomlib.sympy import Le, Eq
        >>> Le.rel_complement()
        <class 'logic1.atomlib.sympy.Gt'>
        >>> Eq.rel_complement()
        <class 'logic1.atomlib.sympy.Ne'>
        >>> Le.rel_complement(9 ** 2 < 80)
        <class 'logic1.atomlib.sympy.Le'>

        Compare the notions of the converse relation R ** (-1) of R, and of the
        dual relation (R') ** (-1), which equals (R ** (-1))'.
        """
        ...

    # Instance methods
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

        >>> from logic1.atomlib.sympy import EQ
        >>> from sympy.abc import a, b
        >>> EQ(a, 0).to_cnf()
        Eq(a, 0)
        """
        return self

    @final
    def to_complement(self, conditional: bool = True) -> AtomicFormula:
        # Do not pass on but check conditional in order to avoid construction
        # in case of False.
        if conditional:
            return self.func.rel_complement()(*self.args)
        return self

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
