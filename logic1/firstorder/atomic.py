from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, TypeAlias

from .formula import Formula
from ..support.decorators import classproperty

from ..support.tracing import trace  # noqa


class AtomicFormula(Formula):

    @classproperty
    def func(cls):
        """A class property yielding this class or the derived subclass itself.
        """
        return cls

    # The following would be an abstract class property, which is not available
    # at the moment.
    complement_func: type[AtomicFormula]  #: :meta private:

    # Instance variables
    args: tuple

    # Static methods on terms
    @staticmethod
    @abstractmethod
    def term_get_vars(term: Any) -> set:
        """Extract the set of variables occurring in `term`.
        """
        ...

    @staticmethod
    @abstractmethod
    def term_to_latex(term: Any) -> str:
        """Convert `term` to LaTeX.
        """
        ...

    # Static methods on variables
    @staticmethod
    @abstractmethod
    def variable_type() -> Any:
        """The Python type of variables in terms in subclasses of
        :class:`AtomicFormula`.
        """
        ...

    @staticmethod
    @abstractmethod
    def rename_var(variable: Any) -> Any:
        """Return a fresh variable for replacing `variable` in the course of
        renaming.

        Compare :meth:`.Formula.to_distinct_vars`, :meth:`.Formula.to_pnf`.
        """
        ...

    def __init__(self, *args) -> None:
        self.args = args

    def __str__(self) -> str:
        # Overloading __str__ here breaks an infinite recursion in the
        # inherited Formula.__str__. Nicer string representation are provided
        # by various theory modules.
        return repr(self)

    def as_latex(self) -> str:
        return f'\\verb!{repr(self)}!'

    def get_qvars(self) -> set:
        """Implements the abstract method :meth:`Formula.get_qvars`.
        """
        return set()

    def to_complement(self) -> AtomicFormula:
        """Returns an :class:`AtomicFormula` equivalent to ``~ self``.
        """
        return self.complement_func(*self.args)


class Term(ABC):

    @classmethod
    @abstractmethod
    def fresh_variable(cls, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence _G0001,
        _G0002, ..., G9999, G10000, ... This naming convention is inspired by
        Lisp's gensym(). If the optional argument :data:`suffix` is specified,
        the sequence _G0001_<suffix>, _G0002_<suffix>, ... is used instead.
        """
        ...

    @abstractmethod
    def get_vars(self) -> set[Variable]:
        """Extract the set of variables occurring in `self`.
        """
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """Convert `self` to LaTeX.
        """
        ...


Variable: TypeAlias = Term
