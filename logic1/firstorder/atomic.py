from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Self, TypeAlias

from .formula import Formula
from ..support.decorators import classproperty

from ..support.tracing import trace  # noqa


class AtomicFormula(Formula):

    @classproperty
    def func(cls):
        """A class property yielding this class or the derived subclass itself.
        """
        return cls

    @classproperty
    def complement_func(cls):
        # Should be an abstract class property
        raise NotImplementedError

    args: tuple

    def __init__(self, *args) -> None:
        self.args = args

    def __str__(self) -> str:
        # Overloading __str__ here breaks an infinite recursion in the
        # inherited Formula.__str__. Nicer string representations are provided
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
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """Convert `self` to LaTeX.
        """
        ...

    @abstractmethod
    def get_vars(self) -> set[Self]:  # mypy complains about -> set[Variable]
        """Extract the set of variables occurring in `self`.
        """
        ...

    @staticmethod
    @abstractmethod
    def sort_key(term: Any) -> Any:
        ...


Variable: TypeAlias = Term
