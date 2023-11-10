from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable, final, Iterator, TYPE_CHECKING
from typing_extensions import Self

from .formula import Formula
from .quantified import Ex, All
from ..support.decorators import classproperty

if TYPE_CHECKING:
    import sympy

from ..support.tracing import trace  # noqa


class AtomicFormula(Formula):

    # Class variables
    latex_symbol_spacing = ' '
    """A class variable holding LaTeX spacing that goes around infix operators.

    This is used with :meth:`Formula.to_latex`, which is in turn used for the
    output in Jupyter notebooks.
    """

    text_symbol_spacing = ' '
    """A class variable holding spacing that goes around infix operators in
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    print_precedence = 99
    """A class variable holding the precedence of `latex_symbol` and
    `text_symbol` with LaTeX and string conversions in subclasses.

    This is compared with the corresponding `print_precedence` of other classes
    for placing parentheses.
    """

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
    def term_type() -> Any:
        """The Python type of terms of the respective subclass of
        :class:`AtomicFormula`.
        """
        ...

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

    @staticmethod
    @abstractmethod
    def term_to_sympy(term: Any) -> sympy.Basic:
        """Convert `term` to SymPy.
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

    # Instance methods
    def __init__(self, *args) -> None:
        self.args = args

    def atoms(self) -> Iterator[AtomicFormula]:
        yield self

    @final
    def _count_alternations(self) -> tuple[int, set]:
        return (-1, {Ex, All})

    def depth(self) -> int:
        return 0

    @final
    def get_qvars(self) -> set:
        """Implements the abstract method :meth:`Formula.get_qvars`.
        """
        return set()

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
    def to_sympy(self, **kwargs) -> sympy.core.basic.Basic:
        """Override :meth:`.Formula.to_sympy` to prevent recursion into terms.
        """
        sympy_terms = (self.__class__.term_to_sympy(arg) for arg in self.args)
        return self.__class__.sympy_func(*sympy_terms, **kwargs)

    @final
    def transform_atoms(self, transformation: Callable) -> Self:
        """Implements the abstract method :meth:`Formula.transform_atoms`.
        """
        return transformation(self)
