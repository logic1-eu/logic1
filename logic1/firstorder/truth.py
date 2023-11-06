from __future__ import annotations

from typing import final, Optional
from typing_extensions import Self

from ..support.containers import GetVars
from ..support.decorators import classproperty

from .boolean import BooleanFormula, Not
from .formula import Formula
from .quantified import Ex, All


class TruthValue(BooleanFormula):
    r"""A class whose instances are formulas corresponding to :math:`\top` and
    :math:`\bot`.
    """

    # Class variables
    print_precedence = 99
    """A class variable holding the precedence of the operators of instances of
    :class:`TruthValue` in LaTeX and string conversions.

    This is compared with the corresponding `print_precedence` of other classes
    for placing parentheses.
    """

    print_style = 'constant'
    """A class variable indicating the use of operators of instances of
    :class:`TruthValue` as constants in LaTeX and string conversions.
    """

    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[TruthValue]  #: :meta private:
    dual_func: type[TruthValue]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[()]  #: :meta private:

    # Instance methods
    def _count_alternations(self) -> tuple[int, set]:
        return (-1, {Ex, All})

    def depth(self) -> int:
        return 0

    def get_qvars(self) -> set:
        """Implements the abstract method :meth:`Formula.get_qvars()`.
        """
        return set()

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`Formula.get_vars`.
        """
        return GetVars()

    def to_cnf(self) -> Self:
        """ Convert to Conjunctive Normal Form.

        >>> F.to_dnf()
        F
        """
        return self

    def _to_dnf(self) -> Self:
        """ Convert to Disjunctive Normal Form.

        >>> T.to_dnf()
        T
        """
        return self

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> Formula:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        if to_positive:
            return self.dual_func() if _implicit_not else self
        if _implicit_not:
            return Not(self)
        return self

    def to_sympy(self):
        """Raises :exc:`NotImplementedError` since SymPy does not
        know quantifiers.
        """
        raise NotImplementedError(f'sympy does not know {self.func}')


@final
class _T(TruthValue):
    """The constant Formula that is always true.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _T to be a
    subclass itself.

    >>> _T() is _T()
    True
    """

    # Class variables
    latex_symbol = '\\top'
    """A class variable holding a LaTeX symbol for :class:`_T`.

    This is used with :meth:`Formula.to_latex`, which is in turn used for the
    output in Jupyter notebooks.
    """

    text_symbol = 'T'
    """A class variable holding a representation of :class:`_T` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    @classproperty
    def func(cls):
        """A class property yielding the class :class:`_T` itself.
        """
        return cls

    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`_F`, which implements
        the dual operator :math:`\bot` or :math:`\top`.
        """
        return _F

    _instance: Optional[_T] = None

    # Class methods
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Instance methods
    def __init__(self) -> None:
        self.args = ()

    def __repr__(self) -> str:
        return 'T'


T = _T()
"""Support use as a constant without parentheses.
"""


@final
class _F(TruthValue):
    """The constant Formula that is always false.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _F to be a
    subclass itself.

    >>> _F() is _F()
    True
    """

    # Class variables
    latex_symbol = '\\bot'
    """A class variable holding a LaTeX symbol for :class:`_T`.

    This is used with :meth:`Formula.to_latex`, which is in turn used for the
    output in Jupyter notebooks.
    """

    text_symbol = 'F'
    """A class variable holding a representation of :class:`_F` suitable for
    string representation.

    This is used for string conversions, e.g., explicitly with the constructor
    of :class:`str` or implicitly with :func:`print`.
    """

    @classproperty
    def func(cls):
        """A class property yielding the class :class:`_F` itself.
        """
        return cls

    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`_T`, which implements
        the dual operator :math:`\top` or :math:`\bot`.
        """
        return (lambda: _T)()

    _instance: Optional[_F] = None

    # Class methods
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Instance methods
    def __init__(self) -> None:
        self.args = ()

    def __repr__(self) -> str:
        return 'F'


F = _F()
"""Support use as a constant without parentheses.
"""
