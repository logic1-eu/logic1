from __future__ import annotations

import functools
import inspect
import operator
from sage.all import Integer, latex, PolynomialRing, ZZ  # type: ignore[import-untyped]
from sage.rings.polynomial.multi_polynomial_libsingular import (  # type: ignore[import-untyped]
    MPolynomial_libsingular as Polynomial)
from types import FrameType
from typing import Any, ClassVar, Final, Iterable, Iterator, Optional, Self, TypeAlias

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

    def add_var(self, var: str) -> None:
        new_vars = [str(g) for g in self.sage_ring.gens()]
        assert var not in new_vars
        new_vars.append(var)
        new_vars.sort()
        self.sage_ring = PolynomialRing(ZZ, new_vars, implementation='singular')

    def ensure_vars(self, vars_: Iterable[str]) -> None:

        def sort_key(s: str) -> tuple[str, int]:
            base = s.rstrip('0123456789')
            index = s[len(base):]
            n = int(index) if index else -1
            return base, n

        new_vars = [str(g) for g in self.sage_ring.gens()]
        have_appended = False
        for v in vars_:
            if v not in new_vars:
                have_appended = True
                new_vars.append(v)
        if have_appended:
            new_vars.sort(key=sort_key)
            self.sage_ring = PolynomialRing(ZZ, new_vars, implementation='singular')

    def get_vars(self) -> tuple[Polynomial, ...]:
        gens = self.sage_ring.gens()
        gens = (g for g in gens if str(g) != 'unused_')
        return tuple(gens)

    def pop(self) -> None:
        self.sage_ring = self.stack.pop()

    def push(self) -> None:
        self.stack.append(self.sage_ring)
        self.sage_ring = PolynomialRing(ZZ, 'unused_', implementation='singular')


ring = _Ring()


class _VariableSet:

    _instance: ClassVar[Optional[_VariableSet]] = None

    wrapped_ring: PolynomialRing

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        vars_ = set(str(g) for g in self.wrapped_ring.get_vars())
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in vars_:
            i += 1
            v = f'G{i:04d}{suffix}'
        self.wrapped_ring.add_var(v)
        return Term(self.wrapped_ring(v))

    def get(self, *args) -> tuple[Variable, ...]:
        return tuple(self[name] for name in args)

    def __getitem__(self, index: str) -> Variable:
        match index:
            case str():
                self.wrapped_ring.ensure_vars((index,))
                return Term(self.wrapped_ring(index))
            case _:
                raise ValueError(f'expecting string as index; {index} is {type(index)}')

    def imp(self, *args) -> None:
        """Import variables into global namespace.
        """
        vars_ = self.get(*args)
        frame = inspect.currentframe()
        assert isinstance(frame, FrameType)
        frame = frame.f_back
        try:
            assert isinstance(frame, FrameType)
            module = frame.f_globals['__name__']
            assert module == '__main__', \
                f'expecting imp to be called from the top level of module __main__; ' \
                f'context is module {module}'
            function = frame.f_code.co_name
            assert function == '<module>', \
                f'expecting imp to be called from the top level of module __main__; ' \
                f'context is function {function} in module {module}'
            for v in vars_:
                frame.f_globals[str(v)] = v
        finally:
            # Compare Note here:
            # https://docs.python.org/3/library/inspect.html#inspect.Traceback
            del frame

    def __init__(self, ring_: PolynomialRing) -> None:
        self.wrapped_ring = ring_

    def __new__(cls, ring_: PolynomialRing):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def pop(self) -> None:
        self.wrapped_ring.pop()

    def push(self) -> None:
        self.wrapped_ring.push()

    def _stack(self) -> list[PolynomialRing]:
        return self.wrapped_ring.stack

    def __repr__(self):
        vars_ = self.wrapped_ring.get_vars()
        s = ', '.join(str(g) for g in (*vars_, '...'))
        return f'{{{s}}}'


VV = _VariableSet(ring)


class Term(firstorder.Term):

    wrapped_ring: _Ring = ring
    wrapped_variable_set: _VariableSet = VV

    _poly: Polynomial

    @property
    def poly(self):
        if self.wrapped_ring.sage_ring is not self._poly.parent():
            self.wrapped_ring.ensure_vars(str(g) for g in self._poly.parent().gens())
            self._poly = self.wrapped_ring(self._poly)
        return self._poly

    @poly.setter
    def poly(self, value):
        self._poly = value

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

    def fresh(self) -> Variable:
        assert self._is_variable()
        return self.wrapped_variable_set.fresh(suffix=f'_{str(self)}')

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
                self.poly = self.wrapped_ring(arg)
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

    def _is_variable(self) -> bool:
        return self.poly in self.poly.parent().gens()

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
        """Complement relation.
        """
        D: Any = {Eq: Ne, Ne: Eq, Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}
        return D[cls]

    @classproperty
    def converse_func(cls) -> type[AtomicFormula]:
        """Converse relation.
        """
        D: Any = {Eq: Eq, Ne: Ne, Le: Ge, Lt: Gt, Ge: Le, Gt: Lt}
        return D[cls]

    @classproperty
    def dual_func(cls) -> type[AtomicFormula]:
        """Dual relation.
        """
        return cls.complement_func.converse_func

    @property
    def lhs(self) -> Term:
        _lhs = self.args[0]
        assert isinstance(_lhs, Term)
        return _lhs

    @property
    def rhs(self) -> Term:
        assert len(self.args) == 2
        _rhs = self.args[1]
        assert isinstance(_rhs, Term)
        return _rhs

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

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs >= 0 else F
        return Ge(Term(lhs), Term(0))


class Le(AtomicFormula):

    sage_func = operator.le

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs <= 0 else F
        return Le(Term(lhs), Term(0))


class Gt(AtomicFormula):

    sage_func = operator.gt

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs > 0 else F
        return Gt(Term(lhs), Term(0))


class Lt(AtomicFormula):

    sage_func = operator.lt

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs.poly - self.rhs.poly
        if lhs.is_constant():
            return T if lhs < 0 else F
        return Lt(Term(lhs), Term(0))
