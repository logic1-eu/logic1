from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator, Self, TypeAlias

from .formula import Formula
from ..support.decorators import classproperty

from ..support.tracing import trace  # noqa


class AtomicFormula(Formula):

    @classproperty
    def complement(cls):
        """The complement operator of an atomic formula.

        Should be an abstract class property
        """
        raise NotImplementedError()

    def __init__(self, *args) -> None:
        self.args = args

    def __str__(self) -> str:
        # Overloading __str__ here breaks an infinite recursion in the
        # inherited Formula.__str__. Nicer string representations are provided
        # by various theory modules.
        return repr(self)

    def as_latex(self) -> str:
        """Provides verbatim output of :code:`repr(self)` as a default
        implementation. This is expected to be overridden in subclasses.

        .. seealso::
            :meth:`.Formula.as_latex` -- LaTeX representation
        """
        return f'\\verb!{repr(self)}!'

    @abstractmethod
    def _bvars(self, quantified: set) -> Iterator[Variable]:
        ...

    @abstractmethod
    def _fvars(self, quantified: set) -> Iterator[Variable]:
        ...

    def simplify(self) -> Formula:
        """Provides identity as a default implementation of simplification.
        This is expected to be overridden in subclasses.

        .. seealso::
            :meth:`.Formula.simplify` -- simplification
        """
        return self

    @abstractmethod
    def subs(self, substitution: dict) -> Self:
        """
        .. seealso::
            :meth:`.Formula.subs` -- substitution
        """
        ...

    def to_complement(self) -> Self:
        """Returns an :class:`AtomicFormula` equivalent to ``Not(self)``.

        .. seealso::
            :attr:`complement` -- complement relation
        """
        return self.complement(*self.args)


class Term(ABC):

    @abstractmethod
    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far.
        """
        ...

    @abstractmethod
    def as_latex(self) -> str:
        """LaTeX representation.

        .. seealso::
            :meth:`.Formula.as_latex` -- LaTeX representation
        """
        ...

    @staticmethod
    @abstractmethod
    def sort_key(term: Any) -> Any:
        """A sort key suitable for ordering terms. Note that ``<``, ``<=`` etc.
        are reserved as constructors for instances of :class:`.AtomicFormula`.
        """
        ...

    @abstractmethod
    def vars(self) -> Iterator[Variable]:
        """All variables occurring in self.

        .. seealso::
            * :meth:`.Formula.bvars` -- all occurring bound variables
            * :meth:`.Formula.fvars` -- all occurring free variables
            * :meth:`.Formula.qvars` -- all occurring quantified variables
        """
        ...


Variable: TypeAlias = Term
