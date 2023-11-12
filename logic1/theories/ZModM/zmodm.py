import logging
from typing import Optional

import sympy

from ... import atomlib
from ...firstorder import T, F
from ...support.decorators import classproperty


logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


def show_progress(flag: bool = True) -> None:
    if flag:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.CRITICAL)


Term = sympy.Expr
Variable = sympy.Symbol


_modulus = None


def mod() -> Optional[int]:
    return _modulus


def set_mod(modulus: Optional[int]) -> Optional[int]:
    global _modulus
    save_modulus = _modulus
    _modulus = modulus
    return save_modulus


class Eq(atomlib.generic.EqMixin, atomlib.sympy.BinaryAtomicFormula):

    @classproperty
    def complement_func(cls):
        """The complement relation Ne of Eq.
        """
        return Ne

    @classproperty
    def converse_func(cls):
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


class Ne(atomlib.generic.NeMixin, atomlib.sympy.BinaryAtomicFormula):

    @classproperty
    def complement_func(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classproperty
    def converse_func(cls):
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
