from __future__ import annotations

from typing import final, Optional
from typing_extensions import Self

from ..support.containers import GetVars
from ..support.decorators import classproperty

from .boolean import BooleanFormula, Not
from .formula import Formula
from .quantified import Ex, All


class TruthValue(BooleanFormula):

    # Class variables
    print_precedence = 99
    print_style = 'constant'

    func: type[TruthValue]
    dual_func: type[TruthValue]

    # Instance variables
    args: tuple[()]

    # Instance methods
    def _count_alternations(self) -> tuple[int, set]:
        return (-1, {Ex, All})

    def get_qvars(self) -> set:
        return set()

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        return GetVars()

    def to_cnf(self) -> Self:
        """ Convert to Conjunctive Normal Form.

        >>> F.to_dnf()
        F
        """
        return self

    def to_dnf(self) -> Self:
        """ Convert to Disjunctive Normal Form.

        >>> T.to_dnf()
        T
        """
        return self

    def to_nnf(self, implicit_not: bool = False, to_positive: bool = True)\
            -> Formula:
        if to_positive:
            return self.dual_func() if implicit_not else self
        if implicit_not:
            return Not(self)
        return self

    def _to_pnf(self) -> dict:
        """Prenex normal form. self must be in negation normal form.
        """
        return {Ex: self, All: self}

    def to_sympy(self):
        raise NotImplementedError(f'sympy does not know {self.func}')


@final
class _T(TruthValue):
    """The constant Formula that is always true.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _T to be a
    subclass itself.
    """

    # Class variables
    latex_symbol = '\\top'
    text_symbol = 'T'

    @classproperty
    def func(cls):
        return cls

    @classproperty
    def dual_func(cls):
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


@final
class _F(TruthValue):
    """The constant Formula that is always false.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _F to be a
    subclass itself.
    """

    # Class variables
    latex_symbol = '\\bot'
    text_symbol = 'F'

    @classproperty
    def func(cls):
        return cls

    @classproperty
    def dual_func(cls):
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
