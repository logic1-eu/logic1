from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Self, TypeAlias

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

    @abstractmethod
    def _bvars(self, quantified: set) -> Iterator[Variable]:
        ...

    @abstractmethod
    def _fvars(self, quantified: set) -> Iterator[Variable]:
        ...

    def simplify(self) -> Formula:
        return self

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        ...

    def to_complement(self) -> AtomicFormula:
        """Returns an :class:`AtomicFormula` equivalent to ``~ self``.
        """
        return self.complement_func(*self.args)


class Term(ABC):

    @abstractmethod
    def fresh(self) -> Variable:
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """Convert `self` to LaTeX.
        """
        ...

    @staticmethod
    @abstractmethod
    def sort_key(term: Any) -> Any:
        ...

    @abstractmethod
    def vars(self) -> Iterator[Variable]:
        ...


Variable: TypeAlias = Term
