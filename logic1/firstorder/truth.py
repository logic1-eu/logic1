from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar, final, Optional
from typing_extensions import Self

from ..support.containers import GetVars

from .boolean import BooleanFormula, Not
from .formula import Formula
from .quantified import Ex, All


class TruthValue(BooleanFormula):

    print_precedence: ClassVar[int] = 99
    print_style: ClassVar[str] = 'constant'

    func: type[TruthValue]
    args: tuple[()]

    @staticmethod
    @abstractmethod
    def to_dual(conditional: bool = True):
        ...

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
            return self.func.to_dual(conditional=implicit_not)()
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
    latex_symbol: ClassVar[str] = '\\top'
    text_symbol: ClassVar[str] = 'T'

    _instance: ClassVar[Optional[_T]] = None

    func: type[_T]

    @staticmethod
    def to_dual(conditional: bool = True) -> type[_T] | type[_F]:
        if conditional:
            return _F
        return _T

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        self.func = _T
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
    latex_symbol: ClassVar[str] = '\\bot'
    text_symbol: ClassVar[str] = 'F'

    _instance: ClassVar[Optional[_F]] = None

    func: type[_F]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def to_dual(conditional: bool = True) -> type[_T] | type[_F]:
        if conditional:
            return _T
        return _F

    def __init__(self) -> None:
        self.func = _F
        self.args = ()

    def __repr__(self) -> str:
        return 'F'


F = _F()
