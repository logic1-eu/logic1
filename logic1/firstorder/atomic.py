from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, final, Iterator
from typing_extensions import Self

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

    @final
    def _count_alternations(self) -> tuple[int, set]:
        return (-1, {Ex, All})

    @final
    def get_qvars(self) -> set:
        """Implements the abstract method :meth:`Formula.get_qvars`.
        """
        return set()

    @final
    def matrix(self) -> tuple[Formula, list[QuantifierBlock]]:
        return self, []

    @final
    def to_complement(self) -> AtomicFormula:
        """Returns an :class:`AtomicFormula` equivalent to ``~ self``.
        """
        return self.complement_func(*self.args)

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> Formula:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        return self.to_complement() if _implicit_not else self

    @final
    def transform_atoms(self, transformation: Callable) -> Self:
        """Implements the abstract method :meth:`Formula.transform_atoms`.
        """
        return transformation(self)


# The following imports are intentionally late to avoid circularity.
from .quantified import Ex, All, QuantifierBlock
