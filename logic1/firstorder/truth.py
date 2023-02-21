from __future__ import annotations

from abc import abstractmethod
from typing import final

from ..support.containers import GetVars

from .formula import Formula
from .quantified import Ex, All
from .boolean import BooleanFormula, Not


class TruthValue(BooleanFormula):

    print_precedence = 99
    print_style = 'constant'

    func: type[TruthValue]

    @staticmethod
    @abstractmethod
    def to_dual(conditional: bool = True):
        ...

    # Instance methods
    def _count_alternations(self) -> tuple:
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

    def to_nnf(self, implicit_not: bool = False,
               to_positive: bool = True) -> Formula:
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
    latex_symbol = '\\top'
    text_symbol = 'T'

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def to_dual(conditional: bool = True):
        if conditional:
            return _F
        return _T

    def __init__(self):
        self.func = _T
        self.args = ()

    def __repr__(self):
        return 'T'


T = _T()


@final
class _F(TruthValue):
    """The constant Formula that is always false.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _F to be a
    subclass itself.
    """
    latex_symbol = '\\bot'
    text_symbol = 'F'

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @staticmethod
    def to_dual(conditional: bool = True):
        if conditional:
            return _T
        return _F

    def __init__(self):
        self.func = _F
        self.args = ()

    def __repr__(self):
        return 'F'


F = _F()
