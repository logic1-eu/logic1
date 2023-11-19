from __future__ import annotations

import operator

from sage.all import Integer, latex, PolynomialRing, ZZ  # type: ignore
from sage.rings.polynomial.multi_polynomial_libsingular import (  # type: ignore
    MPolynomial_libsingular)
from typing import Optional, Self, TypeAlias

from ... import firstorder
from ...firstorder import Formula, T, F
from ...atomlib import generic
from ...support.containers import GetVars
from ...support.decorators import classproperty

from ...support.tracing import trace  # noqa


Term: TypeAlias = MPolynomial_libsingular
Variable: TypeAlias = MPolynomial_libsingular


class _Ring:

    _instance: Optional[_Ring] = None

    def __call__(self, obj):
        return self.sage_ring(obj)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.sage_ring = PolynomialRing(ZZ, 'one_unused_variable',
                                        implementation='singular')
        self.stack = []

    def __repr__(self):
        return str(self.sage_ring)

    def add_var(self, v: str) -> Variable:
        return self.add_vars(v)[0]

    def add_vars(self, *args) -> tuple[Variable, ...]:
        if len(args) == 0:
            # The code below is correct also for len(args) == 0, but I do now
            # want to recreate polynomial rings without a good reason.
            return ()
        strings = list(args)
        gens_as_str = [str(gen) for gen in self.sage_ring.gens()]
        for arg in strings:
            if not isinstance(arg, str):
                raise ValueError(f'{arg} is not a string')
            if arg in gens_as_str:
                raise ValueError(f'{arg} is already a variable') from None
        self.sage_ring = PolynomialRing(ZZ, gens_as_str + strings,
                                        implementation='singular')
        gens = self.sage_ring.gens()[len(gens_as_str):]
        return gens

    def get_vars(self) -> tuple[Variable]:
        return self.sage_ring.gens()

    def pop(self) -> PolynomialRing:
        self.sage_ring = self.stack.pop()
        return self.sage_ring

    def push(self) -> list[PolynomialRing]:
        self.stack.append(self.sage_ring)
        return self.stack

    def set_vars(self, *args) -> tuple[Variable, ...]:
        self.sage_ring = PolynomialRing(ZZ, 'one_unused_variable',
                                        implementation='singular')
        gens = self.add_vars(*args)
        return gens


ring = _Ring()


class TermMixin():

    @staticmethod
    def term_get_vars(term: Term) -> set[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_get_vars`.
        """
        return set(term.variables())

    @staticmethod
    def term_to_latex(term: Term) -> str:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.term_to_latex`.
        """
        return str(latex(term))

    @staticmethod
    def variable_type() -> type[Variable]:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.variable_type`.
        """
        return Variable

    @staticmethod
    def rename_var(variable: Variable) -> Variable:
        """Implements the abstract method
        :meth:`.firstorder.AtomicFormula.rename_var`.
        """
        i = 0
        vars_as_str = tuple(str(v) for v in ring.get_vars())
        v_as_str = str(variable)
        while v_as_str in vars_as_str:
            i += 1
            v_as_str = str(variable) + "_R" + str(i)
        v = ring.add_var(v_as_str)
        return v


class AtomicFormula(TermMixin, firstorder.AtomicFormula):
    """Atomic Formula with Sage Terms. All terms are
    :class:`sage.symbolic.expression.Expression`.
    """

    def get_vars(self, assume_quantified: set = set()) -> GetVars:
        """Implements the abstract method :meth:`.firstorder.Formula.get_vars`.
        """
        all_vars = set()
        for term in self.args:
            all_vars.update(set(term.variables()))
        return GetVars(free=all_vars - assume_quantified,
                       bound=all_vars & assume_quantified)

    def subs(self, d: dict) -> Self:
        """Implements the abstract method :meth:`.firstorder.Formula.subs`.
        """
        sage_keywords = {str(v): t for v, t in d.items()}
        args = (arg.subs(**sage_keywords) for arg in self.args)
        return self.func(*args)


class BinaryAtomicFormula(generic.BinaryAtomicFormulaMixin, AtomicFormula):

    def __init__(self, lhs, rhs, chk: bool = True):
        if chk:
            args_ = []
            for arg in (lhs, rhs):
                assert isinstance(arg, (int, Integer, MPolynomial_libsingular)), arg
                args_.append(ring(arg))
            super().__init__(*args_)
        else:
            super().__init__(lhs, rhs)

    def _sprint(self, mode: str) -> str:
        match mode:
            case 'latex':
                symbol = self.__class__.latex_symbol
                lhs = str(latex(self.lhs))
                rhs = str(latex(self.rhs))
                spacing = self.__class__.latex_symbol_spacing
            case 'text':
                symbol = self.__class__.text_symbol
                lhs = self.lhs.__str__()
                rhs = self.rhs.__str__()
                spacing = self.__class__.text_symbol_spacing
            case _:
                assert False
        return f'{lhs}{spacing}{symbol}{spacing}{rhs}'


class Eq(generic.EqMixin, BinaryAtomicFormula):

    sage_func = operator.eq
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

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        if lhs.is_zero():
            return T
        if lhs.is_constant():
            return F
        return Eq(lhs, 0, chk=False)


class Ne(generic.NeMixin, BinaryAtomicFormula):

    sage_func = operator.ne  #: :meta private:
    func: type[Ne]

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

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        if lhs.is_zero():
            return F
        if lhs.is_constant():
            return T
        return Ne(lhs, 0, chk=False)


class Ge(generic.GeMixin, BinaryAtomicFormula):

    sage_func = operator.ge  #: :meta private:
    func: type[Ge]

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Lt` of :class:`Ge`.
        """
        return Lt

    @classproperty
    def converse_func(cls):
        """The converse relation Le of Ge.
        """
        return Le

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        if lhs.is_constant():
            return T if lhs >= 0 else F
        return Ge(lhs, 0, chk=False)


class Le(generic.LeMixin, BinaryAtomicFormula):

    sage_func = operator.le
    func: type[Le]

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Lt` of :class:`Ge`.
        """
        return Gt

    @classproperty
    def converse_func(cls):
        """The converse relation Le of Ge.
        """
        return Ge

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        if lhs.is_constant():
            return T if lhs <= 0 else F
        return Le(lhs, 0, chk=False)


class Gt(generic.GtMixin, BinaryAtomicFormula):

    sage_func = operator.gt
    func: type[Gt]

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Le` of :class:`Gt`.
        """
        return Le

    @classproperty
    def converse_func(cls):
        """The converse relation Lt of Gt.
        """
        return Lt

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        if lhs.is_constant():
            return T if lhs > 0 else F
        return Gt(lhs, 0, chk=False)


class Lt(generic.LtMixin, BinaryAtomicFormula):

    sage_func = operator.lt
    func: type[Lt]

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Ge` of :class:`Lt`.
        """
        return Ge

    @classproperty
    def converse_func(cls):
        """The converse relation Gt of Lt.
        """
        return Gt

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        if lhs.is_constant():
            return T if lhs < 0 else F
        return Lt(lhs, 0, chk=False)


# The type alias `RcfAtomicFormula` supports :code:`cast(RcfAtomicFormula,
# ...)`. Furthermore, :code:`type[RcfAtomicFormula]` can be used for
# annotating, e.g., :attr:`dual_func`. The tuple `RcfAtomicFormulas` supports
# :code:`assert isinstance(..., RcfAtomicFormulas)`.
RcfAtomicFormula: TypeAlias = Eq | Ne | Ge | Le | Gt | Lt

RcfAtomicFormulas = (Eq, Ne, Ge, Le, Gt, Lt)
