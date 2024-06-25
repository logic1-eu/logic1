from abc import ABCMeta
import logging
import sympy
from typing import Final, Optional, TypeAlias

from ... import atomlib
from ...firstorder import T, F
from ...support.decorators import classproperty


logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


Term: TypeAlias = sympy.Expr
Variable: TypeAlias = sympy.Symbol


_modulus = None


def mod() -> Optional[int]:
    return _modulus


def set_mod(modulus: Optional[int]) -> Optional[int]:
    global _modulus
    save_modulus = _modulus
    _modulus = modulus
    return save_modulus


class BinaryAtomicFormula(atomlib.sympy.BinaryAtomicFormula):

    def __str__(self) -> str:
        SYMBOL: Final = {Eq: '==', Ne: '!='}
        SPACING: Final = ' '
        return f'{self.lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{self.rhs}'

    def as_latex(self) -> str:
        SYMBOL: Final = {Eq: '=', Ne: '\\neq'}
        SPACING: Final = ' '
        lhs = sympy.latex(self.lhs)
        rhs = sympy.latex(self.rhs)
        return f'{lhs}{SPACING}{SYMBOL[self.op]}{SPACING}{rhs}'

    def relations(self) -> list[ABCMeta]:
        return [Eq, Ne]


class Eq(atomlib.generic.EqMixin, BinaryAtomicFormula):

    @classmethod
    def complement(cls):
        """The complement relation Ne of Eq.
        """
        return Ne

    @classmethod
    def converse(cls):
        """The converse relation Eq of Eq.
        """
        return Eq

    def simplify(self):
        lhs = self.args[0] - self.args[1]
        lhs = lhs.expand(modulus=_modulus)
        if lhs == sympy.Integer(0):
            return T
        if not lhs.free_symbols:
            return F
        return Eq(lhs, 0)


class Ne(atomlib.generic.NeMixin, BinaryAtomicFormula):

    @classmethod
    def complement(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classmethod
    def converse(cls):
        """The converse relation Ne of Ne.
        """
        return Ne

    def simplify(self):
        lhs = self.args[0] - self.args[1]
        lhs = lhs.expand(modulus=_modulus)
        if lhs == sympy.Integer(0):
            return F
        if not lhs.free_symbols:
            return T
        return Ne(lhs, 0)
