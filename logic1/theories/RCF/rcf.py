from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import operator
from sage.all import Integer, latex, PolynomialRing, ZZ  # type: ignore
from sage.rings.polynomial.multi_polynomial_libsingular import (  # type: ignore
    MPolynomial_libsingular)
import sys
from types import FrameType
from typing import Final, Optional, Self, TypeAlias

from ... import firstorder
from ...firstorder import Formula, T, F
from ...atomlib import generic
from ...support.containers import GetVars
from ...support.decorators import classproperty

from ...support.tracing import trace  # noqa


@dataclass
class Term(firstorder.Term):

    poly: MPolynomial_libsingular

    @classmethod
    def fresh_variable(cls, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence _G0001,
        _G0002, ..., G9999, G10000, ... This naming convention is inspired by
        Lisp's gensym(). If the optional argument :data:`suffix` is specified,
        the sequence _G0001_<suffix>, _G0002_<suffix>, ... is used instead.
        """
        if suffix != '':
            suffix = f'_{suffix}'
        vars_as_str = tuple(str(v) for v in ring.get_vars())
        i = 1
        v_as_str = f'_G{i:04d}{suffix}'
        while v_as_str in vars_as_str:
            i += 1
            v_as_str = f'_G{i:04d}{suffix}'
        v = ring.add_var(v_as_str)
        return v

    def __add__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly + other.poly)
        return Term(self.poly + other)

    def __radd__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__add__`.
        return Term(other + self.poly)

    def __mul__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly * other.poly)
        return Term(self.poly * other)

    def __rmul__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__mul__`.
        return Term(other * self.poly)

    def __neg__(self) -> Term:
        return Term(- self.poly)

    def __sub__(self, other: object) -> Term:
        if isinstance(other, Term):
            return self + (- other)
        return Term(self.poly - other)

    def __rsub__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__sub__`.
        return Term(other - self.poly)

    def __pow__(self, other: object) -> Term:
        return Term(self.poly ** other)

    def __xor__(self, other: object) -> Term:
        return self ** other

    def get_vars(self) -> set[Variable]:
        """Extract the set of variables occurring in `self`.
        """
        print('getting_vars')


Variable: TypeAlias = Term


class _Ring:

    _instance: Optional[_Ring] = None

    def __call__(self, obj):
        return self.sage_ring(obj)

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.sage_ring = PolynomialRing(ZZ, 'unused_', implementation='singular')
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
        added_as_str = list(args)
        old_as_str = [str(gen) for gen in self.sage_ring.gens()]
        for v in added_as_str:
            if not isinstance(v, str):
                raise ValueError(f'{v} is not a string')
            if v in old_as_str:
                raise ValueError(f'{v} is already a variable') from None
        new_as_str = sorted(old_as_str + added_as_str)
        self.sage_ring = PolynomialRing(ZZ, new_as_str, implementation='singular')
        added_as_gen = (self.sage_ring(v) for v in added_as_str)
        added_as_var = (Variable(g) for g in added_as_gen)
        return tuple(added_as_var)

    def get_vars(self) -> tuple[Variable]:
        gens = self.sage_ring.gens()
        gens = (g for g in gens if str(g) != 'unused_')
        vars_ = (Variable(g) for g in gens)
        return tuple(vars_)

    def import_vars(self, force: bool = False):
        critical = []
        gens = self.get_vars()
        frame = inspect.currentframe()
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        try:
            assert isinstance(frame, FrameType)
            name = frame.f_globals['__name__']
            assert name == '__main__', f'import_vars called from {name}'
            for gen in gens:
                try:
                    expr = frame.f_globals[str(gen)]
                    if expr != gen:
                        critical.append(str(gen))
                except KeyError:
                    pass
            for gen in gens:
                if force or str(gen) not in critical:
                    frame.f_globals[str(gen)] = gen
            if not force:
                if len(critical) == 1:
                    print(f'{critical[0]} has another value already, '
                          f'use force=True to overwrite ',
                          file=sys.stderr)
                elif len(critical) > 1:
                    print(f'{", ".join(critical)} have other values already, '
                          f'use force=True to overwrite ',
                          file=sys.stderr)
        finally:
            # Compare Note here:
            # https://docs.python.org/3/library/inspect.html#inspect.Traceback
            del frame

    def pop(self) -> PolynomialRing:
        self.sage_ring = self.stack.pop()
        return self.sage_ring

    def push(self) -> list[PolynomialRing]:
        self.stack.append(self.sage_ring)
        return self.stack

    def set_vars(self, *args) -> tuple[Variable, ...]:
        self.sage_ring = PolynomialRing(ZZ, 'unused_', implementation='singular')
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


@functools.total_ordering
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

    def __le__(self, other: Formula) -> bool:
        match other:
            case BinaryAtomicFormula():
                if self.lhs != other.lhs:
                    return not self.lhs <= other.lhs
                if self.rhs != other.rhs:
                    return not self.rhs <= other.rhs
                L = [Eq, Ne, Le, Lt, Ge, Gt]
                return L.index(self.func) <= L.index(other.func)
            case _:
                assert not isinstance(other, AtomicFormula)
                return True

    def __str__(self) -> str:
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs}{SPACING}{SYMBOL[self.func]}{SPACING}{self.rhs}'

    def as_latex(self) -> str:
        SYMBOL: Final = {
            Eq: '=', Ne: '\\neq', Ge: '\\geq', Le: '\\leq', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        lhs = str(latex(self.lhs))
        rhs = str(latex(self.rhs))
        return f'{lhs}{SPACING}{SYMBOL[self.func]}{SPACING}{rhs}'


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
