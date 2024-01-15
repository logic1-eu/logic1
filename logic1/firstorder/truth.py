from __future__ import annotations

from typing import final, Optional

from ..support.containers import GetVars
from ..support.decorators import classproperty

from .boolean import BooleanFormula, Not
from .formula import Formula
from .quantified import Ex, All


class TruthValue(BooleanFormula):
    r"""A class whose instances are formulas corresponding to :math:`\top` and
    :math:`\bot`.
    """
    # The following would be abstract class variables, which are not available
    # at the moment.
    func: type[TruthValue]  #: :meta private:
    dual_func: type[TruthValue]  #: :meta private:

    # Similarly the following would be an abstract instance variable:
    args: tuple[()]  #: :meta private:

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

    def to_nnf(self, to_positive: bool = True,
               _implicit_not: bool = False) -> Formula:
        """Implements the abstract method :meth:`Formula.to_nnf`.
        """
        if to_positive:
            return self.dual_func() if _implicit_not else self
        if _implicit_not:
            return Not(self)
        return self


@final
class _T(TruthValue):
    """The constant Formula that is always true.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _T to be a
    subclass itself.

    >>> _T() is _T()
    True
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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

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

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.args = ()

    def __repr__(self) -> str:
        return 'F'


F = _F()
"""Support use as a constant without parentheses.
"""
