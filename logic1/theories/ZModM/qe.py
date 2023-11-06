from typing import Optional

from ... import abc
from ...firstorder import Formula, Or
from .zmodm import Eq, Ne, mod, set_mod, Variable
from .pnf import pnf as _pnf


class QuantifierElimination(abc.qe.QuantifierElimination):
    """Quantifier elimination
    """

    def __call__(self, f, modulus: Optional[int] = None):
        if modulus is not None:
            save_modulus = set_mod(modulus)
            result = self.qe(f)
            set_mod(save_modulus)
            return result
        assert isinstance(mod(), int)
        return self.qe(f)

    def pnf(self, f: Formula) -> Formula:
        return _pnf(f)

    def qe1p(self, v: Variable, f: Formula) -> Formula:
        modulus = mod()
        assert isinstance(modulus, int)
        return Or(*(f.subs({v: i}) for i in range(modulus))).simplify()

    @staticmethod
    def is_valid_atom(f: Formula) -> bool:
        return isinstance(f, (Eq, Ne))

    def simplify(self, f: Formula) -> Formula:
        return f.simplify()


qe = quantifier_elimination = QuantifierElimination()
