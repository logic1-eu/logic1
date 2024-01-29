from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
import operator
from sage.all import Integer, latex, PolynomialRing, ZZ  # type: ignore[import-untyped]
from sage.rings.polynomial.multi_polynomial_libsingular import (  # type: ignore[import-untyped]
    MPolynomial_libsingular as Polynomial)
import sys
from types import FrameType
from typing import Any, Final, Iterator, Optional, Self, TypeAlias

from ... import firstorder
from ...firstorder import Formula, T, F
from ...support.decorators import classproperty

from ...support.tracing import trace  # noqa


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
                raise ValueError(f'{v} is already a variable')
        new_as_str = sorted(old_as_str + added_as_str)
        self.sage_ring = PolynomialRing(ZZ, new_as_str, implementation='singular')
        added_as_gen = (self.sage_ring(v) for v in added_as_str)
        added_as_var = (Variable(g) for g in added_as_gen)
        return tuple(added_as_var)

    def get_vars(self) -> tuple[Variable, ...]:
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


@dataclass
class Term(firstorder.Term):

    poly: Polynomial

    @classmethod
    def fresh_variable(cls, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        vars_as_str = tuple(str(v) for v in ring.get_vars())
        i = 1
        v_as_str = f'G{i:04d}{suffix}'
        while v_as_str in vars_as_str:
            i += 1
            v_as_str = f'G{i:04d}{suffix}'
        v = ring.add_var(v_as_str)
        return v

    def __add__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly + other.poly)
        return Term(self.poly + other)

    def __eq__(self, other: Term | Polynomial | Integer | int) -> Eq:  # type: ignore[override]
        # discuss: we have Eq.__bool__ but this way we cannot compare Terms
        # with arbitrary objects in a boolean context. Same for __ne__.
        if isinstance(other, Term):
            return Eq(self, other)
        return Eq(self, Term(other))

    def __ge__(self, other: Term | Polynomial | Integer | int) -> Ge:
        if isinstance(other, Term):
            return Ge(self, other)
        return Ge(self, Term(other))

    def __gt__(self, other: Term | Polynomial | Integer | int) -> Gt:
        if isinstance(other, Term):
            return Gt(self, other)
        return Gt(self, Term(other))

    def __hash__(self) -> int:
        return hash((tuple(str(cls) for cls in self.__class__.mro()), self.poly))

    def __init__(self, arg: Polynomial | Integer | int) -> None:
        match arg:
            case Polynomial():
                self.poly = arg
            case Integer() | int():
                self.poly = ring(arg)
            case _:
                raise ValueError(
                    f'arguments must be polylnomial or integer; {arg} is {type(arg)}')

    def __le__(self, other: Term | Polynomial | Integer | int) -> Le:
        if isinstance(other, Term):
            return Le(self, other)
        return Le(self, Term(other))

    def __lt__(self, other: Term | Polynomial | Integer | int) -> Lt:
        if isinstance(other, Term):
            return Lt(self, other)
        return Lt(self, Term(other))

    def __mul__(self, other: object) -> Term:
        if isinstance(other, Term):
            return Term(self.poly * other.poly)
        return Term(self.poly * other)

    def __ne__(self, other: Term | Polynomial | Integer | int) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        return Ne(self, Term(other))

    def __neg__(self) -> Term:
        return Term(- self.poly)

    def __pow__(self, other: object) -> Term:
        return Term(self.poly ** other)

    def __repr__(self) -> str:
        return str(self.poly)

    def __radd__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__add__`.
        return Term(other + self.poly)

    def __rmul__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__mul__`.
        return Term(other * self.poly)

    def __rsub__(self, other: object) -> Term:
        # We know that other is not a :class:`Term`, see :meth:`__sub__`.
        return Term(other - self.poly)

    def __sub__(self, other: object) -> Term:
        if isinstance(other, Term):
            return self + (- other)
        return Term(self.poly - other)

    def __xor__(self, other: object) -> Term:
        return self ** other

    def as_latex(self) -> str:
        """Convert `self` to LaTeX.

        Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_Latex`.
        """
        return str(latex(self.poly))

    def _coefficient(self, d: dict[Variable, int]) -> Term:
        d_poly = {key.poly: value for key, value in d.items()}
        return Term(self.poly.coefficient(d_poly))

    def _degree(self, x: Variable) -> int:
        return self.poly.degree(x.poly)

    def _derivative(self, x: Variable, n: int = 1) -> Term:
        return Term(self.poly.derivative(x.poly, n))

    def vars(self) -> Iterator[Variable]:
        # discuss "from" vs. "for"
        yield from (Term(g) for g in self.poly.variables())

    def _is_constant(self) -> bool:
        return self.poly.is_constant()

    def _is_zero(self) -> bool:
        return self.poly.is_zero()

    @staticmethod
    def sort_key(term: Term) -> Polynomial:
        return term.poly

    def subs(self, d: dict[Variable, Term]) -> Term:
        sage_keywords = {str(v.poly): t.poly for v, t in d.items()}
        return Term(self.poly.subs(**sage_keywords))


Variable: TypeAlias = Term


@functools.total_ordering
class AtomicFormula(firstorder.AtomicFormula):

    @classproperty
    def complement_func(cls) -> type[AtomicFormula]:
        D: Any = {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}
        return D[cls]

    @classproperty
    def converse_func(cls) -> type[AtomicFormula]:
        D: Any = {Eq: Eq, Ne: Ne, Le: Ge, Lt: Gt, Ge: Le, Gt: Lt}
        return D[cls]

    @classproperty
    def dual_func(cls) -> type[AtomicFormula]:
        return cls.complement_func.converse_func

    @property
    def lhs(self):
        return self.args[0]

    @property
    def rhs(self):
        return self.args[1]

    def __init__(self, lhs: Term | Polynomial | Integer | int,
                 rhs: Term | Polynomial | Integer | int):
        # Integer is a candidate for removal
        assert not isinstance(lhs, Integer)
        assert not isinstance(rhs, Integer)
        if not isinstance(lhs, Term):
            lhs = Term(lhs)
        if not isinstance(rhs, Term):
            rhs = Term(rhs)
        super().__init__(lhs, rhs)

    def __le__(self, other: Formula) -> bool:
        match other:
            case AtomicFormula():
                if self.lhs != other.lhs:
                    return not Term.sort_key(self.lhs) <= Term.sort_key(other.lhs)
                if self.rhs != other.rhs:
                    return not Term.sort_key(self.rhs) <= Term.sort_key(other.rhs)
                L = [Eq, Ne, Le, Lt, Ge, Gt]
                return L.index(self.func) <= L.index(other.func)
            case _:
                return True

    def __repr__(self) -> str:
        if self.lhs.poly.is_constant() and self.rhs.poly.is_constant():
            # Return Eq(1, 2) instead of 1 == 2, because the latter is not
            # suitable as input.
            return super().__repr__()
        return str(self)

    def __str__(self) -> str:
        SYMBOL: Final = {Eq: '==', Ne: '!=', Ge: '>=', Le: '<=', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.poly}{SPACING}{SYMBOL[self.func]}{SPACING}{self.rhs.poly}'

    def as_latex(self) -> str:
        SYMBOL: Final = {
            Eq: '=', Ne: '\\neq', Ge: '\\geq', Le: '\\leq', Gt: '>', Lt: '<'}
        SPACING: Final = ' '
        return f'{self.lhs.as_latex()}{SPACING}{SYMBOL[self.func]}{SPACING}{self.rhs.as_latex()}'

    def _bvars(self, quantified: set) -> Iterator[Variable]:
        yield from (v for v in self.lhs.vars() if v in quantified)
        yield from (v for v in self.rhs.vars() if v in quantified)

    def _fvars(self, quantified: set) -> Iterator[Variable]:
        yield from (v for v in self.lhs.vars() if v not in quantified)
        yield from (v for v in self.rhs.vars() if v not in quantified)

    def subs(self, d: dict[Variable, Term]) -> Self:
        """Implements the abstract method :meth:`.firstorder.atomic.AtomicFormula.subs`.
        """
        return self.func(self.lhs.subs(d), self.rhs.subs(d))


class Eq(AtomicFormula):

    sage_func = operator.eq
    func: type[Eq]

    def __bool__(self) -> bool:
        return self.lhs.poly == self.rhs.poly

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_zero():
            return T
        if lhs.is_constant():
            return F
        return Eq(Term(lhs), Term(0))


class Ne(AtomicFormula):

    sage_func = operator.ne  #: :meta private:
    func: type[Ne]

    def __bool__(self) -> bool:
        return self.lhs.poly != self.rhs.poly

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_zero():
            return F
        if lhs.is_constant():
            return T
        return Ne(Term(lhs), Term(0))


class Ge(AtomicFormula):

    sage_func = operator.ge  #: :meta private:
    func: type[Ge]

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs >= 0 else F
        return Ge(Term(lhs), Term(0))


class Le(AtomicFormula):

    sage_func = operator.le
    func: type[Le]

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs <= 0 else F
        return Le(Term(lhs), Term(0))


class Gt(AtomicFormula):

    sage_func = operator.gt
    func: type[Gt]

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs > 0 else F
        return Gt(Term(lhs), Term(0))


class Lt(AtomicFormula):

    sage_func = operator.lt
    func: type[Lt]

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs < 0 else F
        return Lt(Term(lhs), Term(0))
