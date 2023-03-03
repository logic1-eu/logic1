from __future__ import annotations

from itertools import combinations
import logging

import sympy

from logic1 import atomlib
from logic1 import abc
from logic1.firstorder.boolean import And, Or
from logic1.firstorder.formula import Formula
from logic1.support.decorators import classproperty

logging.basicConfig(
    format='%(levelname)s[%(relativeCreated)0.0f ms]: %(message)s',
    level=logging.CRITICAL)


def show_progress(flag: bool = True) -> None:
    if flag:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.CRITICAL)


Term = sympy.Symbol
Variable = sympy.Symbol


class TermMixin():

    @staticmethod
    def term_type() -> type[Term]:
        return Term

    @staticmethod
    def variable_type() -> type[Variable]:
        return Variable


class Eq(TermMixin, atomlib.sympy.Eq):
    """Equations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> Eq(x, y)
    Eq(x, y)

    >>> Eq(x, 0)
    Traceback (most recent call last):
    ...
    ValueError: 0 is not a Term

    >>> Eq(x + x, y)
    Traceback (most recent call last):
    ...
    ValueError: 2*x is not a Term
    """

    # Class variables
    func: type[Eq]

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

    def __init__(self, *args):
        super().__init__(*args)


class Ne(TermMixin, atomlib.sympy.Ne):
    """Inequations with only variables as terms.

    This implements that fact that the language of sets has no functions and,
    in particular, no constants.

    >>> from sympy.abc import x, y
    >>> Ne(y, x)
    Ne(y, x)

    >>> Ne(x, y + 1)
    Traceback (most recent call last):
    ...
    ValueError: y + 1 is not a Term
    """

    # Class variables
    func: type[Ne]

    @classproperty
    def complement_func(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classproperty
    def converse_func(cls):
        """The converse relation Me of Ne.
        """
        return Ne


oo = atomlib.sympy.oo


class _C(atomlib.sympy._C):

    # Class variables
    @classproperty
    def complement_func(cls):
        return _C_bar


C = _C.interactive_new


class _C_bar(atomlib.sympy._C_bar):

    # Class variables
    @classproperty
    def complement_func(cls):
        return _C


C_bar = _C_bar.interactive_new


class QuantifierElimination(abc.qe.QuantifierElimination):
    """Quantifier elimination for the theory of Sets.

    >>> from logic1 import *
    >>> from sympy.abc import a, u, v, w, x, y, z
    >>> f = All(u, Ex(w, All(x, Ex(y, Ex(v, \
            (Eq(u, v) | Ne(v, w)) & ~ Equivalent(Eq(u, x), Ne(u, w)) & Eq(y, a))))))
    >>> qe(f)
    C_bar(2)

    >>> g = Ex(x, Ex(y, Ex(z, Ne(x, y) & Ne(x, z) & Ne(y, z) \
            & All(u, Eq(u, x) | Eq(u, y) | Eq(u, z)))))
    >>> qe(g)
    And(C(2), C(3), C_bar(4))

    >>> h = Ex(w, Ex(x, Ne(w, x))) >> Ex(w, Ex(x, Ex(y, Ex(z, \
            Ne(w, x) & Ne(w, y) & Ne(w, z) & Ne(x, y) & Ne(x, z) & Ne(y, z)))))
    >>> qe(h)
    Or(And(C(2), C(3), C(4)), C_bar(2))
    """

    # Instance methods
    def __call__(self, f):
        return self.qe(f)

    def qe1p(self, v: Variable, f: Formula) -> Formula:
        def eta(Z: set, k: int) -> Formula:
            args = []
            for k_choice in combinations(Z, k):
                args1 = []
                for z in Z:
                    args_inner = []
                    for chosen_z in k_choice:
                        args_inner.append(Eq(z, chosen_z))
                    args1.append(Or(*args_inner))
                f1 = And(*args1)
                args2 = []
                for i in range(k):
                    for j in range(i + 1, k):
                        args2.append(Ne(k_choice[i], k_choice[j]))
                f2 = And(*args2)
                args.append(And(f1, f2))
            return Or(*args)

        logging.info(f'{self.qe1p.__qualname__}: eliminating Ex {v} ({f})')
        eqs, nes = self._split_by_relation(f)
        if eqs:
            solution = eqs[0].rhs if eqs[0].rhs != v else eqs[0].lhs
            return f.subs({v: solution})
        Z = And(*nes).get_vars().free - {v}
        m = len(Z)
        phi_prime = Or(*(And(eta(Z, k), C(k + 1)) for k in range(1, m + 1)))
        logging.info(f'{self.qe1p.__qualname__}: result is {phi_prime}')
        return phi_prime

    # Static methods
    @staticmethod
    def is_valid_atom(f: Formula) -> bool:
        return isinstance(f, (Eq, Ne, _C, _C_bar))

    @staticmethod
    def _split_by_relation(f) -> tuple[list[type[Eq]], list[type[Ne]]]:
        eqs = []
        nes = []
        args = f.args if f.func is And else (f,)
        for atom in args:
            if atom.func is Eq:
                eqs.append(atom)
            else:
                assert atom.func is Ne, f'bad atom {atom} - {atom.func}'
                nes.append(atom)
        return eqs, nes


qe = quantifier_elimination = QuantifierElimination()
