from __future__ import annotations

from collections.abc import Container, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from fractions import Fraction
from typing import Any, Final, Iterable, Iterator, Mapping, Optional, Self, Tuple

from flint import Ordering, fmpq, fmpq_mpoly, fmpq_mpoly_ctx
from gmpy2 import mpq

from logic1 import firstorder

POLYLIB: Final[str] = "FLINT"

TERM_ORDER: Final[Ordering] = Ordering.deglex


type Constant = float | fmpq | Fraction| int | mpq

_CONSTANT_TYPES: Final[tuple[type, ...]] = (float, fmpq, Fraction, int, mpq)


CACHE_SIZE: Final[Optional[int]] = 2**16


def _caches():
    from logic1.theories.RCF.node.xopt import Node
    from logic1.theories.RCF.simplify import Simplify
    from logic1.theories.RCF.substitution import _SubstValue
    return [_SubstValue.as_term, Simplify._simpl_at, Node.subs_into_formula]

def cache_clear():
    for cache in _caches():
        cache.cache_clear()

def cache_info():
    return {cache.__wrapped__: cache.cache_info() for cache in _caches()}


def as_fmpq(arg: Constant) -> fmpq:
    if isinstance(arg, float):
        return mpq_to_fmpq(mpq(arg))
    elif isinstance(arg, fmpq):
        return arg
    elif isinstance(arg, Fraction):
        return fmpq(arg.numerator, arg.denominator)
    elif isinstance(arg, int):
        return fmpq(arg)
    elif isinstance(arg, mpq):
        return mpq_to_fmpq(arg)
    else:
        constant_types = ', '.join(c.__name__ for c in _CONSTANT_TYPES)
        raise ValueError(f'expected one of {constant_types}; {arg} is {type(arg)}')

def fmpq_to_mpq(r: fmpq) -> mpq:
    return mpq(int(r.numerator), int(r.denominator))

def mpq_to_fmpq(q: mpq) -> fmpq:
    return fmpq(int(q.numerator), int(q.denominator))


def init_env(ring_vars: list[str]) -> None:
    VV._used.update(ring_vars)

def init_env_arg() -> list[str]:
    return list(VV._used)


class FmpqMpolyPoly():

    # All fmpq_mpoly must have the same context. `name` is a name in that
    # context, whose exponent is always 0.

    name: str
    terms: dict[int, fmpq_mpoly]

    def __init__(self, terms: dict[int, fmpq_mpoly], name: str) -> None:
        self.terms = terms
        self.name = name

    def __repr__(self) -> str:
        return f'FmpqMpolyPoly({self.name}, {self.terms})'

    def __str__(self) -> str:
        summands = sorted(self.terms.items(), reverse=True)
        return ' + '.join(f'({coeff!s}) {self.name}^{exp}' for exp, coeff in summands) or '0'

    def as_fmpq_mpoly(self) -> fmpq_mpoly:
        if not self.terms:
            ctx = fmpq_mpoly_ctx.get((self.name,), TERM_ORDER)
            poly = ctx.constant(0)
            return poly

        ctx = next(iter(self.terms.values())).context()
        idx = ctx.variable_to_index(self.name)
        their_terms = {}
        for exp, poly in self.terms.items():
            for exp_vec, coeff in poly.terms():
                their_exp_vec = exp_vec[:idx] + (exp,) + exp_vec[idx+1:]
                their_terms[their_exp_vec] = coeff
        poly = ctx.from_dict(their_terms)
        return poly

    @classmethod
    def from_fmpq_mpoly(cls, poly: fmpq_mpoly, name: str) -> Self:
        """Convert a univariate fmpq_mpoly to an FmpqMpolyPoly. If `name` does
        not occur in the context of `poly`, we first project `poly` to a context
        containing `name`, which is then used as the common context for the
        resulting FmpqMpolyPoly.
        """
        ctx = poly.context()
        zero = ctx.constant(fmpq(0))

        names = ctx.names()
        if name not in names:
            new_names = sorted(names + (name,), key=TermContext.sort_key)
            new_ctx = fmpq_mpoly_ctx.get(new_names, TERM_ORDER)
            new_poly = poly.project_to_context(new_ctx)
            return cls({0: new_poly}, name)

        idx = ctx.variable_to_index(name)
        terms: dict[int, fmpq_mpoly] = {}
        for exp_vec, coeff in poly.terms():
            exp = exp_vec[idx]
            my_exp_vec = exp_vec[:idx] + (0,) + exp_vec[idx+1:]
            my_coeff = terms.get(exp, zero) + ctx.term(coeff=coeff, exp_vec=my_exp_vec)
            terms[exp] = my_coeff
        return cls(terms, name)

    def pseudo_quo_rem(self, divisor: FmpqMpolyPoly, full_delta: bool = True) \
            -> tuple[FmpqMpolyPoly, FmpqMpolyPoly, fmpq_mpoly]:
        """Pseudo-quotient and pseudo-remainder following Knuth Vol. 2.

        Computes (Q, R, lc_pow) satisfying:

            lc_pow * self  =  Q * divisor + R

        where deg(R) < deg(divisor).

        If full_delta=True (default), lc_pow = lc(divisor)^delta where
        delta = max(deg(self) - deg(divisor) + 1, 0), matching Knuth exactly.
        If full_delta=False, lc_pow = lc(divisor)^s where s is the number of
        reduction steps actually performed (s <= delta).

        Examples:

        1. Classic example from Knuth Vol. 2 §4.6.1:

        >>> ctx = fmpq_mpoly_ctx.get(('x',), TERM_ORDER)
        >>> x, = ctx.gens()
        >>> A = FmpqMpolyPoly.from_fmpq_mpoly(
        ...     x**8 - x**6 + 2*x**5 - 2*x**4 + 2*x**3 - 2*x**2 + 2*x - 1, 'x')
        >>> B = FmpqMpolyPoly.from_fmpq_mpoly(
        ...     3*x**6 - 5*x**4 + 3*x**3 - x, 'x')
        >>> Q, R, lc_pow = A.pseudo_quo_rem(B)
        >>> print(Q)
        (9) x^2 + (6) x^0
        >>> print(R)
        (27) x^5 + (-24) x^4 + (45) x^3 + (-54) x^2 + (60) x^1 + (-27) x^0

        lc(B) = 3, so lc_pow = 3^3 = 27:

        >>> lc_pow == ctx.constant(27)
        True

        Fundamental identity lc_pow * A == Q*B + R:

        >>> lc_pow * A.as_fmpq_mpoly() == Q.as_fmpq_mpoly() * B.as_fmpq_mpoly() + R.as_fmpq_mpoly()
        True

        With full_delta=False the loop exits after 2 steps, so lc_pow = 3^2 = 9:

        >>> Q2, R2, lc_pow2 = A.pseudo_quo_rem(B, full_delta=False)
        >>> print(Q2)
        (3) x^2 + (2) x^0
        >>> print(R2)
        (9) x^5 + (-8) x^4 + (15) x^3 + (-18) x^2 + (20) x^1 + (-9) x^0
        >>> lc_pow2 == ctx.constant(9)
        True
        >>> lc_pow2 * A.as_fmpq_mpoly() == Q2.as_fmpq_mpoly() * B.as_fmpq_mpoly() + R2.as_fmpq_mpoly()
        True

        Q and R scale by lc_B^(delta-s) = lc_pow // lc_pow2 between the two modes:

        >>> all(R.terms[e] == R2.terms[e] * (lc_pow // lc_pow2) for e in R2.terms)
        True

        2. Parametric example with multivariate coefficients in QQ[a, b][x]:

        >>> ctx2 = fmpq_mpoly_ctx.get(('a', 'b', 'x'), TERM_ORDER)
        >>> a, b, x = ctx2.gens()
        >>> A3 = FmpqMpolyPoly.from_fmpq_mpoly(a*b*x**2 + (a + b)*x + 1, 'x')
        >>> B3 = FmpqMpolyPoly.from_fmpq_mpoly(a*x + b, 'x')
        >>> Q3, R3, lc_pow3 = A3.pseudo_quo_rem(B3)
        >>> print(Q3)
        (a^2*b) x^1 + (-a*b^2 + a^2 + a*b) x^0

        lc(B3) = a, delta = 2, so lc_pow = a^2:

        >>> lc_pow3 == a**2
        True

        Fundamental identity:

        >>> lc_pow3 * A3.as_fmpq_mpoly() == Q3.as_fmpq_mpoly() * B3.as_fmpq_mpoly() + R3.as_fmpq_mpoly()
        True

        R3 has degree 0; its coefficient is a (b - a) (b^2 - 1):

        >>> print(R3)
        (a*b^3 - a^2*b - a*b^2 + a^2) x^0

        3. Trivial case deg(A) < deg(B): Q = 0, R = A, lc_pow = 1:

        >>> Q0, R0, lc_pow0 = B.pseudo_quo_rem(A)
        >>> Q0.terms, R0.terms == B.terms, lc_pow0 == ctx.constant(1)
        ({}, True, True)
        """
        assert self.name == divisor.name
        assert divisor.terms

        name = self.name
        deg_A = max(self.terms, default=-1)
        deg_B = max(divisor.terms)

        ctx = next(iter(divisor.terms.values())).context()

        zero = ctx.constant(fmpq(0))
        one = ctx.constant(fmpq(1))

        # Trivial case
        if deg_A < deg_B:
            return FmpqMpolyPoly({}, name), FmpqMpolyPoly(dict(self.terms), name), one

        assert next(iter(self.terms.values())).context() is ctx

        lc_B = divisor.terms[deg_B]
        delta = deg_A - deg_B + 1

        R: dict[int, fmpq_mpoly] = dict(self.terms)
        Q: dict[int, fmpq_mpoly] = {}
        steps = 0

        def scale_dict(d: dict[int, fmpq_mpoly], c: fmpq_mpoly) -> None:
            for e in list(d):
                d[e] = c * d[e]

        def add_term(d: dict[int, fmpq_mpoly], e: int, val: fmpq_mpoly) -> None:
            new_c = d.get(e, zero) + val
            if new_c == zero:
                d.pop(e, None)
            else:
                d[e] = new_c

        for _ in range(delta):
            deg_R = max(R, default=-1)
            if deg_R < deg_B:
                break
            lc_R = R[deg_R]
            k = deg_R - deg_B
            scale_dict(R, lc_B)
            scale_dict(Q, lc_B)
            add_term(Q, k, lc_R)
            for e_B, c_B in divisor.terms.items():
                add_term(R, e_B + k, -(lc_R * c_B))
            steps += 1

        # If full_delta, apply the remaining lc_B factors to Q and R
        extra = delta - steps
        if full_delta and extra > 0:
            lc_extra = lc_B ** extra
            scale_dict(Q, lc_extra)
            scale_dict(R, lc_extra)

        lc_pow = lc_B ** (delta if full_delta else steps)

        return FmpqMpolyPoly(Q, name), FmpqMpolyPoly(R, name), lc_pow


class TermContext:

    _poly_context: fmpq_mpoly_ctx

    def __eq__(self, other: object) -> bool:
        """We efficiently check for identical poly_contexts, because equal
        fmpq_mpoly contexts are always identical. However, note that equal
        TermContexts are not necessarily identical.
        """
        if not isinstance(other, TermContext):
            return False
        return self._poly_context is other._poly_context

    def __init__(self, names: Iterable[str]):
        sorted_names = sorted(names, key=TermContext.sort_key)
        self._poly_context = fmpq_mpoly_ctx.get(sorted_names, TERM_ORDER)

    def __or__(self, other: TermContext) -> TermContext:
        """The ring generated by the union of the generators of self and other.
        """
        if self._poly_context is other._poly_context:
            return self
        else:
            return TermContext(set(self._poly_context.names()) | set(other._poly_context.names()))

    def __repr__(self) -> str:
        return f'TermContext({self._poly_context.names()})'

    def coerce(self, term: Term) -> Term:
        """Coerce term to the TermContext self.
        """
        # The following assertion fires in particular when term contains
        # variables created between VV._stash() and VV._drop(). This is useful
        # because we would return a semantically meaningless term:
        assert not (bad_names := set(term._poly.context().names()) - VV._used), \
               f'{term} uses a context with unknown variables {bad_names}'
        return Term.from_raw(self.coerce_poly(term._poly))

    def coerce_poly(self, poly: fmpq_mpoly) -> fmpq_mpoly:
        """Coerce poly to the _flint_context of the TermContext self.
        """
        return poly.project_to_context(self._poly_context)

    @classmethod
    def from_raw(cls, context: fmpq_mpoly_ctx) -> Self:
        self = cls.__new__(cls)
        self._poly_context = context
        return self

    def get_names(self) -> tuple[str, ...]:
        return self._poly_context.names()

    def get_var_by_index(self, index: int) -> Variable:
        gen = self._poly_context.gen(index)
        return Variable.from_raw(gen)

    def get_var_by_name(self, name: str) -> Variable:
        index = self._poly_context.variable_to_index(name)
        return self.get_var_by_index(index)

    def get_vars(self) -> tuple[Variable, ...]:
        return tuple(Variable.from_raw(gen) for gen in self._poly_context.gens())

    def ordering(self) -> Ordering:
        return self._poly_context.ordering()

    @staticmethod
    def sort_key(s: str) -> tuple[Any, ...]:  # MyPy has issues with a more specific return type
        base = s.rstrip('0123456789')
        index = s[len(base):]
        n = int(index) if index else -1
        return base, n


class VariableSet(firstorder.VariableSet['Variable']):
    """The infinite set of all variables belonging to the theory of Real Closed
    Fields. Variables are uniquely identified by their name, which is a
    :external:class:`.str`. This class is a singleton, whose single instance is
    assigned to :data:`.VV`.

    .. seealso::
        Final methods inherited from parent class:

        * :meth:`.firstorder.term.VariableSet.get`
            -- obtain several variables simultaneously
        * :meth:`.firstorder.term.VariableSet.imp`
            -- import variables into global namespace
    """

    _stack: list[set[str]]

    # required by the abstract parent class
    @property
    def stack(self) -> list[set[str]]:
        return self._stack

    @property
    def _used(self) -> set[str]:
        return self._stack[-1]

    def __getitem__(self, name: str) -> Variable:
        """Implements abstract method :meth:`.firstorder.term.VariableSet.__getitem__`.
        """
        if not isinstance(name, str):
            raise ValueError(f'expecting string as index; {name} is {type(name)}')
        tcontext = TermContext([name])
        self._used.add(name)
        return tcontext.get_var_by_index(0)

    def __init__(self) -> None:
        self._reset()

    def __repr__(self) -> str:
        names = sorted(self._used, key=TermContext.sort_key)
        s = ', '.join(name for name in (*names, '...'))
        return f'{{{s}}}'

    def _drop(self) -> None:
        if len(self._stack) <= 1:
            raise ValueError('illegal _drop at bottom of stack')
        self._stack.pop()

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in self._used:
            i += 1
            v = f'G{i:04d}{suffix}'
        return self[v]

    def merge(self) -> None:
        if len(self._stack) <= 1:
            raise ValueError('illegal merge at bottom of stack')
        self.stack[-2].update(self._stack[-1])
        self.stack.pop()

    def pop(self) -> None:
        self._drop()

    def push(self) -> None:
        self.stash()

    def _reset(self) -> None:
        self._stack = [set()]

    def stash(self) -> None:
        self._stack.append(set())


VV: Final = VariableSet()
"""
The unique instance of :class:`.VariableSet`.
"""


class DEFINITE(Enum):
    """Information whether a certain term has positive or negative definiteness
    properties; typically as a result of a heuristic test as in
    :meth:`.Term.is_definite`.
    """

    # This is an ordered Enum, the order of the following properties should not
    # be changed.
    UNKNOWN = auto()
    """It has not been derived that any the other cases holds.
    """

    ZERO = auto()
    """The polynomial is the zero polynomial.
    """

    POSITIVE = auto()
    """The polynomial positive definite, i.e., positive for all real choices of
    variables.
    """

    POSITIVE_SEMI = auto()
    """The polynomial positive semi-definite, i.e., non-negative for all real
    choices of variables.
    """

    NEGATIVE = auto()
    """The polynomial negative definite, i.e., negative for all real choices of
    variables.
    """

    NEGATIVE_SEMI = auto()
    """The polynomial negative semi-definite, i.e., non-positive for all real
    choices of variables.
    """

    # The following is an implementation of OrderedEnum as described in
    # https://docs.python.org/3/howto/enum.html#orderedenum
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    @staticmethod
    def add(x: DEFINITE, y: DEFINITE) -> DEFINITE:
        """Compute DEFINITE of a sum from DEFINITE of the summands.

        >>> l = list(DEFINITE)

        >>> for x in l:
        ...     for y in l:
        ...             print(f'{x.name} + {y.name} = {DEFINITE.add(x,y).name}')
        ...
        UNKNOWN + UNKNOWN = UNKNOWN
        UNKNOWN + ZERO = UNKNOWN
        UNKNOWN + POSITIVE = UNKNOWN
        UNKNOWN + POSITIVE_SEMI = UNKNOWN
        UNKNOWN + NEGATIVE = UNKNOWN
        UNKNOWN + NEGATIVE_SEMI = UNKNOWN
        ZERO + UNKNOWN = UNKNOWN
        ZERO + ZERO = ZERO
        ZERO + POSITIVE = POSITIVE
        ZERO + POSITIVE_SEMI = POSITIVE_SEMI
        ZERO + NEGATIVE = NEGATIVE
        ZERO + NEGATIVE_SEMI = NEGATIVE_SEMI
        POSITIVE + UNKNOWN = UNKNOWN
        POSITIVE + ZERO = POSITIVE
        POSITIVE + POSITIVE = POSITIVE
        POSITIVE + POSITIVE_SEMI = POSITIVE
        POSITIVE + NEGATIVE = UNKNOWN
        POSITIVE + NEGATIVE_SEMI = UNKNOWN
        POSITIVE_SEMI + UNKNOWN = UNKNOWN
        POSITIVE_SEMI + ZERO = POSITIVE_SEMI
        POSITIVE_SEMI + POSITIVE = POSITIVE
        POSITIVE_SEMI + POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE_SEMI + NEGATIVE = UNKNOWN
        POSITIVE_SEMI + NEGATIVE_SEMI = UNKNOWN
        NEGATIVE + UNKNOWN = UNKNOWN
        NEGATIVE + ZERO = NEGATIVE
        NEGATIVE + POSITIVE = UNKNOWN
        NEGATIVE + POSITIVE_SEMI = UNKNOWN
        NEGATIVE + NEGATIVE = NEGATIVE
        NEGATIVE + NEGATIVE_SEMI = NEGATIVE
        NEGATIVE_SEMI + UNKNOWN = UNKNOWN
        NEGATIVE_SEMI + ZERO = NEGATIVE_SEMI
        NEGATIVE_SEMI + POSITIVE = UNKNOWN
        NEGATIVE_SEMI + POSITIVE_SEMI = UNKNOWN
        NEGATIVE_SEMI + NEGATIVE = NEGATIVE
        NEGATIVE_SEMI + NEGATIVE_SEMI = NEGATIVE_SEMI

        This addition is commutative:
        >>> all(DEFINITE.add(x, y) is DEFINITE.add(y, x) for x in l for y in l)
        True

        DEFINITE.zero is a (unique) neutral element:
        >>> all(DEFINITE.add(x, DEFINITE.ZERO) is x for x in l)
        True
        """
        x, y = sorted([x, y])
        if x is DEFINITE.UNKNOWN:
            return DEFINITE.UNKNOWN
        if x is DEFINITE.ZERO:
            return y
        if x is DEFINITE.POSITIVE:
            if y is DEFINITE.POSITIVE or y is DEFINITE.POSITIVE_SEMI:
                return DEFINITE.POSITIVE
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.UNKNOWN
        if x is DEFINITE.POSITIVE_SEMI:
            if y is DEFINITE.POSITIVE_SEMI:
                return DEFINITE.POSITIVE_SEMI
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.UNKNOWN
        if x is DEFINITE.NEGATIVE:
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.NEGATIVE
        assert x is DEFINITE.NEGATIVE_SEMI, (x, y)
        assert y is DEFINITE.NEGATIVE_SEMI, (x, y)
        return DEFINITE.NEGATIVE_SEMI

    @staticmethod
    def from_constant(q: int | mpq | fmpq) -> DEFINITE:
        """Compute DEFINITE of a number.

        >>> print(DEFINITE.from_constant(mpq(42)))
        DEFINITE.POSITIVE

        >>> print(DEFINITE.from_constant(mpq(-4711)))
        DEFINITE.NEGATIVE

        >>> print(DEFINITE.from_constant(mpq(0)))
        DEFINITE.ZERO
        """
        assert isinstance(q, (int, mpq, fmpq)), q
        if q > 0:
            return DEFINITE.POSITIVE
        if q < 0:
            return DEFINITE.NEGATIVE
        assert q == 0, q
        return DEFINITE.ZERO

    @staticmethod
    def mul(x: DEFINITE, y: DEFINITE) -> DEFINITE:
        """Compute DEFINITE of a product from DEFINITE of the factors.

        >>> l = list(DEFINITE)

        The multiplication table:
        >>> for x in l:
        ...     for y in l:
        ...             print(f'{x.name} * {y.name} = {DEFINITE.mul(x,y).name}')
        ...
        UNKNOWN * UNKNOWN = UNKNOWN
        UNKNOWN * ZERO = ZERO
        UNKNOWN * POSITIVE = UNKNOWN
        UNKNOWN * POSITIVE_SEMI = UNKNOWN
        UNKNOWN * NEGATIVE = UNKNOWN
        UNKNOWN * NEGATIVE_SEMI = UNKNOWN
        ZERO * UNKNOWN = ZERO
        ZERO * ZERO = ZERO
        ZERO * POSITIVE = ZERO
        ZERO * POSITIVE_SEMI = ZERO
        ZERO * NEGATIVE = ZERO
        ZERO * NEGATIVE_SEMI = ZERO
        POSITIVE * UNKNOWN = UNKNOWN
        POSITIVE * ZERO = ZERO
        POSITIVE * POSITIVE = POSITIVE
        POSITIVE * POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE * NEGATIVE = NEGATIVE
        POSITIVE * NEGATIVE_SEMI = NEGATIVE_SEMI
        POSITIVE_SEMI * UNKNOWN = UNKNOWN
        POSITIVE_SEMI * ZERO = ZERO
        POSITIVE_SEMI * POSITIVE = POSITIVE_SEMI
        POSITIVE_SEMI * POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE_SEMI * NEGATIVE = NEGATIVE_SEMI
        POSITIVE_SEMI * NEGATIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE * UNKNOWN = UNKNOWN
        NEGATIVE * ZERO = ZERO
        NEGATIVE * POSITIVE = NEGATIVE
        NEGATIVE * POSITIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE * NEGATIVE = POSITIVE
        NEGATIVE * NEGATIVE_SEMI = POSITIVE_SEMI
        NEGATIVE_SEMI * UNKNOWN = UNKNOWN
        NEGATIVE_SEMI * ZERO = ZERO
        NEGATIVE_SEMI * POSITIVE = NEGATIVE_SEMI
        NEGATIVE_SEMI * POSITIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE_SEMI * NEGATIVE = POSITIVE_SEMI
        NEGATIVE_SEMI * NEGATIVE_SEMI = POSITIVE_SEMI

        This multiplication is commutative:
        >>> all(DEFINITE.mul(x, y) is DEFINITE.mul(y, x) for x in l for y in l)
        True

        DEFINITE.POSITIVE is a (unique) neutral element:
        >>> all(DEFINITE.mul(x, DEFINITE.POSITIVE) is x for x in l)
        True
        """
        x, y = sorted([x, y])
        if x is DEFINITE.UNKNOWN:
            if y is DEFINITE.ZERO:
                return DEFINITE.ZERO
            return DEFINITE.UNKNOWN
        if x is DEFINITE.ZERO:
            return DEFINITE.ZERO
        if x is DEFINITE.POSITIVE:
            return y
        if x is DEFINITE.POSITIVE_SEMI:
            if y is DEFINITE.POSITIVE_SEMI:
                return DEFINITE.POSITIVE_SEMI
            assert y is DEFINITE.NEGATIVE or y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.NEGATIVE_SEMI
        if x is DEFINITE.NEGATIVE:
            if y is DEFINITE.NEGATIVE:
                return DEFINITE.POSITIVE
            assert y is DEFINITE.NEGATIVE_SEMI, (x, y)
            return DEFINITE.POSITIVE_SEMI
        assert x is DEFINITE.NEGATIVE_SEMI, (x, y)
        assert y is DEFINITE.NEGATIVE_SEMI, (x, y)
        return DEFINITE.POSITIVE_SEMI

    @staticmethod
    def square(x: DEFINITE) -> DEFINITE:
        if x is DEFINITE.UNKNOWN:
            return DEFINITE.POSITIVE_SEMI
        return DEFINITE.mul(x, x)


@dataclass
class SortKey[τ: Term]:

    term: τ

    def __eq__(self, other: Self) -> bool:  # type: ignore[override]
        return self.cmp(other) == 0

    def __ge__(self, other: Self) -> bool:
        return self.cmp(other) >= 0

    def __gt__(self, other: Self) -> bool:
        return self.cmp(other) > 0

    def __hash__(self) -> int:
        return hash(self.term)

    def __le__(self, other: Self) -> bool:
        return self.cmp(other) <= 0

    def __lt__(self, other: Self) -> bool:
        return self.cmp(other) < 0

    def __ne__(self, other: Self) -> bool:  # type: ignore[override]
        return self.cmp(other) != 0

    def cmp(self, other: Self) -> int:
        tcontext = self.term.term_context() | other.term.term_context()
        p = tcontext.coerce_poly(self.term._poly)
        q = tcontext.coerce_poly(other.term._poly)
        return SortKey.cmp_sage_like(p, q)

    @staticmethod
    def cmp_sage_like(p: fmpq_mpoly, q: fmpq_mpoly) -> int:
        """Comparator intended to match Sage MPolynomial_libsingular ordering.
        """
        def exp_key(exp_vec: Tuple[int, ...]) -> Tuple[int, ...]:
            if TERM_ORDER == Ordering.deglex:
                return (sum(exp_vec), *exp_vec)
            if TERM_ORDER == Ordering.lex:
                return exp_vec
            if TERM_ORDER == Ordering.degrevlex:
                return (sum(exp_vec), *(-e for e in reversed(exp_vec)))
            raise NotImplementedError(f"Unsupported ordering: {TERM_ORDER!r}")

        ep: Optional[tuple[int, ...]]
        eq: Optional[tuple[int, ...]]

        itp = iter(p.terms())  # yields (exp_vec, coeff) in descending monomial order
        itq = iter(q.terms())
        ep, cp = next(itp, (None, None))
        eq, cq = next(itq, (None, None))
        while ep is not None or eq is not None:
            if ep is None:
                # p missing this monomial => coeff 0
                assert cq is not None
                assert cq != 0
                return -1 if 0 < cq else 1
            if eq is None:
                assert cp is not None
                assert cp != 0
                return -1 if cp < 0 else 1
            assert cp is not None and cq is not None and cp != 0 and cq != 0
            kp = exp_key(ep)
            kq = exp_key(eq)
            if kp == kq:
                if cp == cq:
                    ep, cp = next(itp, (None, None))
                    eq, cq = next(itq, (None, None))
                    continue
                return -1 if cp < cq else 1
            elif kp > kq:
                return -1 if cp < 0 else 1
            else:
                return -1 if 0 < cq else 1
        return 0


class Term(firstorder.Term['Term', 'Variable', int, SortKey['Term']]):

    _poly: fmpq_mpoly

    def __add__(self, other: Term | Constant) -> Term:
        if isinstance(other, Term):
            tcontext = self.term_context() | other.term_context()
            sum = tcontext.coerce_poly(self._poly) + tcontext.coerce_poly(other._poly)
            return Term.from_raw(sum)
        else:
            return self + Term(other)

    def __eq__(self, other: Term | Constant) -> Eq:  # type: ignore[override]
        # MyPy requires "other: object". However, with our use a a constructor,
        # it makes no sense to compare terms with general objects. We have
        # Eq.__bool__, which supports some comparisons in Boolean contexts.
        # Same for __ne__.
        lhs = self - other
        # Use _poly.leadling_coefficient() in order to support @lru_cache on
        # Term.lc().
        if lhs._poly.leading_coefficient() < 0:
            lhs = -lhs
        return Eq(lhs, 0)

    def __ge__(self, other: Term | Constant) -> Ge | Le:
        lhs = self - other
        if lhs.lc() < 0:
            return Le(-lhs, 0)
        else:
            return Ge(lhs, 0)

    def __getstate__(self) -> dict[str, Any]:
        poly = self._poly
        poly_as_dict = poly.to_dict()
        names = poly.context().names()
        d = {"poly_as_dict": poly_as_dict, "names": names}
        return d

    def __gt__(self, other: Term | Constant) -> Gt | Lt:
        lhs = self - other
        if lhs.lc() < 0:
            return Lt(-lhs, 0)
        else:
            return Gt(lhs, 0)

    def __hash__(self) -> int:
        # We use this low-level approach because
        # hash(self._summands_as_hashable()) was too slow.
        return hash(repr(self._poly))

    def __init__(self, arg: Constant) -> None:
        """
        >>> Term(0.5)
        1/2
        >>> Term(fmpq(1, 3))
        1/3
        >>> Term(Fraction(1, 4))
        1/4
        >>> Term(42)
        42
        >>> Term(mpq(1, 5))
        1/5
        """
        context = fmpq_mpoly_ctx.get((), TERM_ORDER)
        poly = context.constant(as_fmpq(arg))
        self._poly = poly

    def __iter__(self) -> Iterator[tuple[mpq, Term]]:
        """Iterate over the polynomial representation of the term, yielding
        pairs of coefficients and monomials.

        >>> from gmpy2 import mpq
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> [(abs(coef), power_product) for coef, power_product in t]
        [(mpq(1,1), x**2), (mpq(2,1), x*y), (mpq(1,1), y**2), (mpq(4,1), x),
         (mpq(4,1), y), (mpq(4,1), 1)]
        """
        poly_factory = self._poly.context().term
        for exp_vec, coeff in self._poly.terms():
            monomial = Term.from_raw(poly_factory(exp_vec=exp_vec))
            yield fmpq_to_mpq(coeff), monomial

    def __le__(self, other: Term | Constant) -> Ge | Le:
        lhs = self - other
        if lhs.lc() < 0:
            return Ge(-lhs, 0)
        else:
            return Le(lhs, 0)

    def __len__(self) -> int:
        return len(self._poly)

    def __lt__(self, other: Term | Constant) -> Gt | Lt:
        lhs = self - other
        if lhs.lc() < 0:
            return Gt(-lhs, 0)
        else:
            return Lt(lhs, 0)

    def __mul__(self, other: Term | Constant) -> Term:
        """
        >>> x, y = VV.get('x', 'y')
        >>> (x - y) * (x + y)
        x**2 - y**2
        """
        if isinstance(other, Term):
            tcontext = self.term_context() | other.term_context()
            product = tcontext.coerce_poly(self._poly) * tcontext.coerce_poly(other._poly)
            return Term.from_raw(product)
        else:
            return self * Term(other)

    def __ne__(self, other: Term | Constant) -> Ne:  # type: ignore[override]
        lhs = self - other
        if lhs.lc() < 0:
            lhs = -lhs
        return Ne(lhs, Term(0))

    def __neg__(self) -> Term:
        return Term.from_raw(-self._poly)

    def __pow__(self, exp: int) -> Term:
        return Term.from_raw(self._poly ** exp)

    def __radd__(self, other: Constant) -> Term:
        assert not isinstance(other, Term)
        return Term(other) + self

    def __repr__(self) -> str:
        return self._as_string(mul='*', pow='**')

    def __rmul__(self, other: Constant) -> Term:
        assert not isinstance(other, Term)
        return Term(other) * self

    def __rsub__(self, other: Constant) -> Term:
        assert not isinstance(other, Term)
        return Term(other) - self

    def __setstate__(self, state: dict[str, Any]) -> None:
        context = fmpq_mpoly_ctx.get(state["names"], TERM_ORDER)
        poly = context.from_dict(state["poly_as_dict"])
        self._poly = poly

    def __str__(self) -> str:
        return self._as_string(mul='*', pow='^')

    def __sub__(self, other: Term | Constant) -> Term:
        if isinstance(other, Term):
            tcontext = self.term_context() | other.term_context()
            difference = tcontext.coerce_poly(self._poly) - tcontext.coerce_poly(other._poly)
            return Term.from_raw(difference)
        else:
            return self - Term(other)

    def __truediv__(self, other: Term | Constant) -> Term:
        """True division. `self` must be divisible by `other`. Otherwise, flint
        will raise an error.
        """
        if isinstance(other, Term):
            tcontext = self.term_context() | other.term_context()
            quotient = tcontext.coerce_poly(self._poly) / tcontext.coerce_poly(other._poly)
            return Term.from_raw(quotient)
        else:
            return self / Term(other)

    def __xor__(self, other: object) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_constant(self) -> mpq:
        assert self.is_constant()
        return self.constant_coefficient()

    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.term.Term.as_latex`.

        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.as_latex()
        'x^{2} - 2 x y + y^{2} + 4 x - 4 y + 4'
        """
        return self._as_string(mul=' ', pow='^', exp_pre='{', exp_post='}')

    def _as_string(self, mul: str, pow: str, exp_pre: str = '', exp_post: str = '') -> str:

        def _format_first(c: mpq, d) -> str:
            m = _format_mon(d)
            if c == mpq(1):
                return m if m else f'{c}'
            elif c == mpq(-1):
                return f'-{m}' if m else f'{c}'
            elif c != mpq(0):
                return f'{c}{mul}{m}' if m else f'{c}'
            else:
                assert False, f'zero summand in {self!r}'

        def _format_next(c: mpq, d) -> str:
            m = _format_mon(d)
            if c == mpq(1):
                return f' + {m}' if m else f' + {c}'
            elif c == mpq(-1):
                return f' - {m}' if m else f' - {-c}'
            elif c > mpq(0):
                return f' + {c}{mul}{m}' if m else f' + {c}'
            elif c < mpq(0):
                return f' - {-c}{mul}{m}' if m else f' - {-c}'
            else:
                assert False, f'zero summand in {self!r}'

        def _format_var(v: Variable) -> str:
            return str(v._poly)

        def _format_mon(d: dict[Variable, int]) -> str:
            ret = ''
            for gen in self._poly.context().gens():
                v = Variable.from_raw(gen)
                e = d.get(v, 0)
                if e == 0:
                    continue
                elif e == 1:
                    ret += f'{mul}{_format_var(v)}'
                else:
                    ret += f'{mul}{_format_var(v)}{pow}{exp_pre}{e}{exp_post}'
            return ret.lstrip(mul)

        summands = list(self.summands())
        summands.reverse()
        if not summands:
            return '0'
        d, c = summands.pop()
        ret = [_format_first(c, d)]
        while summands:
            d, c = summands.pop()
            ret.append(_format_next(c, d))
        return ''.join(ret)

    def as_variable(self) -> Variable:
        return Variable.from_raw(self._poly)

    def coefficient(self, degrees: dict[Variable, int]) -> Term:
        """Return the coefficient of the variables with the degrees specified in
        the `degrees`. Mathematically, this is the coefficient in the base ring
        adjoined by the variables of this ring that are not listed in `degrees`.

        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.coefficient({x: 1, y: 1})
        -2
        >>> t.coefficient({x: 1})
        -2*y + 4
        """

        def subtract_if_subset(d1: dict[Variable, int], d2: dict[str, int]) \
                -> Optional[dict[Variable, int]]:
            """Check if d2 constraints are satisfied by d1, return remaining
            exponents. Returns d1 with keys matching d2 removed, if each key in
            d2 either has a zero exponent and does not exist in d1, or has a
            non-zero exponent and exists in d1 with the same exponent. Returns
            None otherwise.
            """
            # Map variable names in d1 back to actual Variable objects
            str_to_var = {str(var1): var1 for var1 in d1}

            # Verify that contraint on d2 keys is satisfied
            for name2, exp2 in d2.items():
                if name2 not in str_to_var:
                    if exp2 != 0:
                        return None
                else:
                    var1 = str_to_var[name2]
                    if d1[var1] != exp2:
                        return None

            # Return d1 with all variables mentioned in d2 removed
            return {var1: exp1 for var1, exp1 in d1.items() if str(var1) not in d2}

        # The result will have the same context as `self`. We reference
        # `degrees` only by the names of the variables.
        degrees_by_names = {str(var): exp for var, exp in degrees.items()}
        ret = Term(0)
        for exp_dict, coeff in self.summands():
            summand_dict = subtract_if_subset(exp_dict, degrees_by_names)
            if summand_dict is not None:
                summand = Term(coeff)
                for var, exp in summand_dict.items():
                    summand *= var ** exp
                ret += summand
        return ret

    def constant_coefficient(self) -> mpq:
        """Return the constant coefficient of this Term.
        """
        last = None
        for last in self._poly.terms():
            pass
        if last is None:
            return mpq(0)
        last_exp_vec, last_coeff = last
        if any(exp != 0 for exp in last_exp_vec):
            return mpq(0)
        return fmpq_to_mpq(last_coeff)

    def content(self) -> mpq:
        """Return the content of this term, which is defined as the positive gcd
        of its rational coefficients.

        >>> x, y = VV.get('x', 'y')
        >>> (mpq(2, 3) * x + mpq(4, 9) * y + mpq(8, 15)).content()
        mpq(2,45)
        """
        content = fmpq(0)
        for coeff in self._poly.coeffs():
            content = content.gcd(coeff)
        assert content > 0 or (content == 0 and self == 0)
        return fmpq_to_mpq(content)

    def degree(self, x: Variable) -> int:
        """Return the degree in `x` of this term. `x` is matched against
        variables of `self` by name. The index of the variables in their
        specific TermContext is not relevant. If no variable with the name of
        `x` occurs in the context of `self`, return 0.

        >>> x, y = VV.get('x', 'y')
        >>> (2*y*x**2 + x + 1).degree(x)
        2
        """
        context = self._poly.context()
        names = context.names()
        x_name = str(x._poly)
        if x_name not in names:
            return 0
        x_index = context.variable_to_index(x_name)
        return self._poly.degrees()[x_index]

    def derivative(self, x: Variable, n: int = 1) -> Term:
        """The `n`-th derivative of this term, with respect to `x`.

        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> (x ** 2 * y + x * z ** 2 + y ** 2 * z).derivative(x)
        2*x*y + z**2
        """
        poly = self._poly
        names = poly.context().names()
        x_name = str(x._poly)
        if x_name not in names:
            return Term(0)
        for _ in range(n):
            poly = poly.derivative(x_name)
        return Term.from_raw(poly)

    def factor(self) -> tuple[mpq, dict[Term, int]]:
        """A polynomial factorization of this term.

        A pair `(unit, D)`, where `unit` is a rational number, the
        keys of `D` are irreducible factors, and the corresponding values are
        their multiplicities. All irreducible factors are monic. Note that
        the return value is uniquely determined by this specification.

        >>> x, y = VV.get('x', 'y')
        >>> t = -x**2 + y**2
        >>> t.factor() == (mpq(-1,1), {x - y: 1, x + y: 1})
        True
        >>> t = (2*x + y)**3
        >>> t.factor() == (mpq(8,1), {x + 1/2*y: 3})
        True
        """
        unit, factors = self._poly.factor()
        D = dict()
        for factor, exp in factors:
            lc = factor.leading_coefficient()
            unit *= lc ** exp
            factor /= lc
            D[Term.from_raw(factor)] = exp
        return fmpq_to_mpq(unit), D

    @classmethod
    def from_raw(cls, poly: fmpq_mpoly) -> Self:
        self = cls.__new__(cls)
        self._poly = poly
        return self

    @classmethod
    def from_sage(self, sage_poly: Any) -> Term:
        """Convert a Sage polynomial to a Term. The context of the
        result is determined by the names of the variables of sage_poly, and
        their order in the term order of sage_poly. In particular, if
        sage_poly has no variables, the result is a constant term.
        """
        def rational_to_fmpq(r):
            return fmpq(int(r.numerator()), int(r.denominator()))

        sage_dict = sage_poly.dict().items()
        flint_dict = {exp_vec: rational_to_fmpq(coeff) for exp_vec, coeff in sage_dict}
        names = sage_poly.parent().variable_names()
        context = fmpq_mpoly_ctx.get(names, TERM_ORDER)
        flint_poly = context.from_dict(flint_dict)
        return Term.from_raw(flint_poly)

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.
        """
        return self._poly.is_constant()

    def is_definite(self, assume: Mapping[Variable, DEFINITE] = {}) -> DEFINITE:
        """A fast heuristic test for definitetess properties of this term. This
        is based on *trivial square sum* properties of coefficient signs and
        exponents.

        >>> x, y = VV.get('x', 'y')
        >>> print(Term(0).is_definite())
        DEFINITE.ZERO
        >>> f = x**2 + y**2
        >>> print(f.is_definite())
        DEFINITE.POSITIVE_SEMI
        >>> g = -x**2 - y**2 - 1
        >>> print(g.is_definite())
        DEFINITE.NEGATIVE
        >>> h = (x - y) ** 2
        >>> print(h.is_definite())
        DEFINITE.UNKNOWN
        >>> print(h.is_definite(assume={x: DEFINITE.POSITIVE, y: DEFINITE.NEGATIVE}))
        DEFINITE.POSITIVE
        >>> print(h.is_definite(assume={x: DEFINITE.NEGATIVE_SEMI, y: DEFINITE.POSITIVE_SEMI}))
        DEFINITE.POSITIVE_SEMI
        """
        # Start with the neutral element of DEFINITE.add().
        poly_result = DEFINITE.ZERO
        for exp_dict, coeff in self.summands():
            # Start with either POSITIVE or NEGATIVE, depending on the coefficient.
            mon_result = DEFINITE.from_constant(coeff)
            for var, exp in exp_dict.items():
                assert exp != 0
                var_exp_result = assume.get(var, DEFINITE.UNKNOWN)
                if exp % 2 == 0:
                    var_exp_result = DEFINITE.square(var_exp_result)
                mon_result = DEFINITE.mul(mon_result, var_exp_result)
            poly_result = DEFINITE.add(poly_result, mon_result)
            if poly_result is DEFINITE.UNKNOWN:
                return DEFINITE.UNKNOWN
        return poly_result

    def is_monomial(self) -> bool:
        """Check if this term is a monomial. A monomial is a summand without its
        coefficient, i.e., there is a bijection between monomials and exponent
        vectors. In particular, 1 is the only constant monomial.
        """
        coeffs = self._poly.coeffs()
        return len(coeffs) == 1 and coeffs[0] == 1

    def is_variable(self) -> bool:
        """Check if this term is a variable.
        """
        poly = self._poly
        return poly in poly.context().gens()

    def is_weakly_parametric_linear(self, X: Container[Variable]) -> bool:
        """Return :obj:`True` if this Term can be written as a_1 x_1 + ... +
        a_n x_n + r such that a_1, ..., a_n in QQ, x_1, ..., x_n in X, and r is
        a polynomial over QQ that does not contain any variable from X.

        >>> a, b, x, y = VV.get('a', 'b', 'x', 'y')
        >>> term = 2 * x - 3 * y + 4 * a**2 + 5 * a * b
        >>> term.is_weakly_parametric_linear({x, y})
        True
        >>> term.is_weakly_parametric_linear({a})
        False
        >>> term.is_weakly_parametric_linear({b})
        False
        """
        for m in self.monomials():
            if m in X:
                continue
            for v in m.vars():
                if v in X:
                    return False
        return True

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term represents the constant zero.
        """
        return self._poly.is_zero()

    def lc(self) -> mpq:
        return fmpq_to_mpq(self._poly.leading_coefficient())

    def monomial_coefficient(self, monomial: Term) -> mpq:
        """Return the rational coefficient of the monomial mon in self. If
        monomial is not a monomial of self, return 0.
        """
        # There is a more straighforward implementation using self.__iter__().
        # However, this method is performance critical, e.g. for substitution in
        # xopt elimination. Here we compare exponent vectors directly, while
        # comparison of monomial Terms involves subtraction, construction of an
        # AtomicFormula, and its evaluation via __bool__.

        tcontext = self.term_context() | monomial.term_context()
        self_poly = tcontext.coerce_poly(self._poly)
        monomial_poly = tcontext.coerce_poly(monomial._poly)
        monomial_terms = tuple(monomial_poly.terms())
        assert len(monomial_terms) == 1, f'{monomial!r} is not a monomial'
        monomial_exp_vec, monomial_coeff = monomial_terms[0]
        assert monomial_coeff == 1, f'{monomial!r} is not a monomial'
        for exp_vec, coeff in self_poly.terms():
            if exp_vec == monomial_exp_vec:
                return fmpq_to_mpq(coeff)
        return mpq(0)

    def monomials(self) -> list[Term]:
        """List of monomials of this term. A monomial is defined here as a
        summand of a polynomial *without* the coefficient.

        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> t.monomials()
        [x**2, x*y, y**2, x, y, 1]
        """
        return [monomial for _, monomial in self]

    def normalize(self) -> Term:
        """Divide this Term by its leading coefficient, so that the result is
        monic.
        """
        poly = self._poly
        return Term.from_raw(poly / poly.leading_coefficient())

    def primitive_part(self, positive: bool = False) -> Term:
        """Return the primitive part over ``Z``. This is ``self`` divided by its
        (positive) content, so that ``self.content() * self.primitive_part() ==
        self``. If ``positive`` is ``True``, the result is normalized to have a
        positive leading coefficient.
        """
        pp = self / self.content()
        if positive and pp.lc() < 0:
            pp = -pp
        return pp

    def pseudo_quo_rem(self, other: Term, x: Variable) -> tuple[Term, Term]:
        """Pseudo quotient and remainder of this term and other, both as
        univariate polynomials in `x` with polynomial coefficients in all other
        variables.

        >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
        >>> f = a * x**2 + b*x + c
        >>> g = c * x + b
        >>> q, r = f.pseudo_quo_rem(g, x); q, r
        (a*c*x - a*b + b*c, a*b**2 - b**2*c + c**3)
        >>> assert c**(2 - 1 + 1) * f == q * g + r
        """
        tcontext = self.term_context() | other.term_context() | x.term_context()
        v = str(x._poly)
        p1 = FmpqMpolyPoly.from_fmpq_mpoly(tcontext.coerce_poly(self._poly), v)
        p2 = FmpqMpolyPoly.from_fmpq_mpoly(tcontext.coerce_poly(other._poly), v)
        quo, rem, _ = p1.pseudo_quo_rem(p2)
        return Term.from_raw(quo.as_fmpq_mpoly()), Term.from_raw(rem.as_fmpq_mpoly())

    def _pseudo_quo_rem_sage(self, other: Term, x: Variable) -> tuple[Term, Term]:
        """Pseudo quotient and remainder of this term and other, both as
        univariate polynomials in `x` with polynomial coefficients in all other
        variables. Deprecated; use peudo_quo_rem() instead.

        >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
        >>> f = a * x**2 + b*x + c
        >>> g = c * x + b
        >>> q, r = f.pseudo_quo_rem(g, x); q, r
        (a*c*x - a*b + b*c, a*b**2 - b**2*c + c**3)
        >>> assert c**(2 - 1 + 1) * f == q * g + r
        """
        tcontext = self.term_context() | other.term_context() | x.term_context()
        p1 = tcontext.coerce(self).to_sage()
        p2 = tcontext.coerce(other).to_sage()
        v = tcontext.coerce(x).to_sage()
        p1_v = p1.polynomial(v)
        p2_v = p2.polynomial(v)
        quo_v, rem_v = p1_v.pseudo_quo_rem(p2_v)
        R = p1.parent()  # same as p2.parent() and v.parent(), and matches tcontext
        quo = R(quo_v)
        rem = R(rem_v)
        return Term.from_sage(quo), Term.from_sage(rem)

    def quo_rem(self, other: Term) -> tuple[Term, Term]:
        """Quotient and remainder of this term and `other`.

        >>> x, y = VV.get('x', 'y')
        >>> f = 2*y*x**2 + x + 1
        >>> f.quo_rem(x)
        (2*x*y + 1, 1)
        >>> f.quo_rem(y)
        (2*x**2, x + 1)
        >>> f.quo_rem(3*x)  # would yield (0, 2*x**2*y + x + 1) over ZZ
        (2/3*x*y + 1/3, 1)
        """
        tcontext = self.term_context() | other.term_context()
        p = tcontext.coerce_poly(self._poly)
        q = tcontext.coerce_poly(other._poly)
        quo, rem = divmod(p, q)
        return Term.from_raw(quo), Term.from_raw(rem)

    def _quo_rem_sage(self, other: Term) -> tuple[Term, Term]:
        """Quotient and remainder of this term and `other`. Deprecated; use
        quo_rem() instead.

        >>> x, y = VV.get('x', 'y')
        >>> f = 2*y*x**2 + x + 1
        >>> f._quo_rem_sage(x)
        (2*x*y + 1, 1)
        >>> f._quo_rem_sage(y)
        (2*x**2, x + 1)
        >>> f._quo_rem_sage(3*x)  # would yield (0, 2*x**2*y + x + 1) over ZZ
        (2/3*x*y + 1/3, 1)
        """
        tcontext = self.term_context() | other.term_context()
        p1 = tcontext.coerce(self).to_sage()
        p2 = tcontext.coerce(other).to_sage()
        quo, rem = p1.quo_rem(p2)
        return Term.from_sage(quo), Term.from_sage(rem)

    def reduce(self, G: Iterable[Term]) -> Term:
        """Reduce self modulo G using the standard algorithm (Buchberger/CLO
        Division Algorithm in k[x_1,...,x_n]). The result is the remainder of
        self modulo G with respect to TERM_ORDER. The result is unique modulo
        the ideal generated by G, but may depend on the order of G and
        TERM_ORDER when G is not a Gröbner basis.

        >>> x, y = VV.get('x', 'y')

        Univariate reduction:
        >>> (x**2 - 1).reduce([x - 1])
        0
        >>> (x**3).reduce([x**2 - 1])
        x

        Multivariate reduction to zero when `self` is in the ideal:
        >>> (x**2 - y**2).reduce([x - y, x + y])
        0

        Reducing modulo an empty list returns `self` unchanged:
        >>> (2*x*y + 3).reduce([])
        2*x*y + 3

        The remainder depends on the order of `G` when `G` is not a Gröbner
        basis:
        >>> f = x**2*y + x*y**2 + y**2
        >>> f.reduce([x*y - 1, y**2 - 1])
        x + y + 1
        >>> f.reduce([y**2 - 1, x*y - 1])
        2*x + 1
        """
        tcontext = self.term_context()
        for g in G:
            tcontext |= g.term_context()
        f_poly = tcontext.coerce_poly(self._poly)
        G_poly = tuple(tcontext.coerce_poly(g._poly) for g in G)
        rem, _ = self.reduce_poly(f_poly, G_poly, tcontext._poly_context)
        return Term.from_raw(rem)

    @staticmethod
    def reduce_poly(f: fmpq_mpoly, G: Sequence[fmpq_mpoly], ctx: fmpq_mpoly_ctx) \
            -> Tuple[fmpq_mpoly, list[fmpq_mpoly]]:
        """Reduce self modulo G.
        """
        N: Final = ctx.nvars()
        ZERO: Final = ctx.constant(0)
        qs = [ZERO] * len(G)
        r  = ZERO
        p  = f

        while p != ZERO:
            lm_p, lc_p = next(iter(p.terms()))

            divided = False
            for i, g in enumerate(G):
                if g == ZERO:
                    continue
                lm_g, lc_g = next(iter(g.terms()))

                # Check if lm_p is divisible by lm_g
                if all(lm_p[j] >= lm_g[j] for j in range(N)):
                    quot_exp = tuple(lm_p[j] - lm_g[j] for j in range(N))
                    quot_coeff = lc_p / lc_g
                    quot_term = ctx.term(coeff=quot_coeff, exp_vec=quot_exp)
                    qs[i] += quot_term
                    p -= quot_term * g
                    divided = True
                    break  # restart scan from g_0 (standard algorithm)

            if not divided:
                # Leading term of p is irreducible — move it to remainder
                lt_p = ctx.term(coeff=lc_p, exp_vec=lm_p)
                r += lt_p
                p -= lt_p

        return r, qs

    def _reduce_sage(self, G: Iterable[Term]) -> Term:
        """Reduce self modulo G via Sage. Deprecated; use reduce() instead.
        """
        tcontext = self.term_context()
        for g in G:
            tcontext |= g.term_context()
        self_sage = tcontext.coerce(self).to_sage()
        G_sage = tuple(tcontext.coerce(g).to_sage() for g in G)
        return Term.from_sage(self_sage.reduce(G_sage))

    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.term.Term.sort_key`.
        """
        return SortKey(self)

    def subs(self, mapping: Mapping[Variable, Term | Constant]) -> Term:
        """Simultaneous substitution of terms or constants for variables.

        >>> x, y, z = VV.get('x', 'y', 'z')
        >>> f = 2*y*x**2 + x + 1
        >>> f.subs({x: y, y: 2*z})
        4*y**2*z + y + 1
        """
        proper_term_mapping: dict[Variable, Term] = {}
        constant_mapping: dict[str | int, fmpq] = {}
        for var, substitute in mapping.items():
            if isinstance(substitute, Term):
                if substitute.is_constant():
                    constant_mapping[str(var._poly)] = mpq_to_fmpq(substitute.as_constant())
                else:
                    proper_term_mapping[var] = substitute
            else:
                constant_mapping[str(var._poly)] = as_fmpq(substitute)
        # First use fmpq_mpoly.subs for subsituting constants:
        pre_result = Term.from_raw(self._poly.subs(constant_mapping))
        # Then use arithmetic for the rest:
        result = Term(0)
        for exp_dict, coeff in pre_result.summands():
            summand = Term(coeff)
            for var, exp in exp_dict.items():
                substitute = proper_term_mapping.get(var, var)
                summand *= substitute ** exp
            result += summand
        return result

    def subs_linear_solution(self, x: Variable, minimal_polynomial: Term) -> Term:
        """Substitute the solution of the weakly parametric linear
        polynomial ``minimal_polynomial`` this weakly parametric linear
        polynomial.
        """
        # self = a * x + b
        a = self.monomial_coefficient(x)
        b = self - a * x
        assert x not in b.vars(), f'{tuple(b.vars())} contains {x!r}, {tuple((x - x).vars())=}'
        # minimal_polynomial = c * x + d
        c = minimal_polynomial.monomial_coefficient(x)
        d = minimal_polynomial - c * x
        assert x not in d.vars()
        result = a * (-d / c) + b
        return result

    def summands(self) -> Iterator[tuple[dict[Variable, int], mpq]]:
        """Iterate over the summands of self yielding pairs of dictionaries
        representing monomials, and coefficients.|

        >>> x, y = VV.get('x', 'y')
        >>> motzkin = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1
        >>> list(motzkin.summands())
        [({x: 4, y: 2}, mpq(1,1)), ({x: 2, y: 4}, mpq(1,1)),
         ({x: 2, y: 2}, mpq(-3,1)), ({}, mpq(1,1))]
        """
        poly = self._poly
        vars = tuple(Variable.from_raw(gen) for gen in poly.context().gens())
        for exp_vec, coeff in poly.terms():
            exp_dict = dict()
            for i, exp in enumerate(exp_vec):
                if exp:
                    exp_dict[vars[i]] = int(exp)
            yield exp_dict, fmpq_to_mpq(coeff)

    def _summands_as_hashable(self) -> tuple[tuple[tuple[tuple[str, int], ...], mpq], ...]:
        """Return the summands of self as a hashable tuple of pairs of tuples
        representing monomials, and coefficients. The summands ordered by the
        term order.

        >>> x, y = VV.get('x', 'y')
        >>> motzkin = x**4 * y**2 + x**2 * y**4 - 3 * x**2 * y**2 + 1
        >>> motzkin._summands_as_hashable()
        (((('x', 4), ('y', 2)), mpq(1,1)), ((('x', 2), ('y', 4)), mpq(1,1)),
         ((('x', 2), ('y', 2)), mpq(-3,1)), ((), mpq(1,1)))
        """
        poly = self._poly
        gen = poly.context().gen
        summands = []
        for exp_vec, coeff in poly.terms():
            exp_list = []
            for i, exp in enumerate(exp_vec):
                if exp:
                    exp_list.append((str(gen(i)), exp))
            summands.append((tuple(exp_list), fmpq_to_mpq(coeff)))
        return tuple(summands)

    def term_context(self) -> TermContext:
        """Return a TermContext of this term.
        """
        return TermContext.from_raw(self._poly.context())

    def to_sage(self) -> Any:
        """Convert this Term to a Sage MPolynomial_libsingular if
        """
        from sage.all import PolynomialRing, QQ, Rational

        def fmpq_to_rational(r: fmpq) -> Rational:
            return Rational((int(r.numerator), int(r.denominator)))

        flint_poly = self._poly
        flint_dict = dict(flint_poly.terms())
        sage_dict = {exp_vec: fmpq_to_rational(coeff) for exp_vec, coeff in flint_dict.items()}
        names = flint_poly.context().names()
        if names:
            sage_ring = PolynomialRing(QQ, names=names, order='deglex', implementation='singular')
        else:
            sage_ring = PolynomialRing(QQ, names=names, order='deglex')
        sage_poly = sage_ring(sage_dict)
        return sage_poly

    def vars(self) -> Iterator[Variable]:
        """An iterator that yields each variable of this term once. Implements
        the abstract method :meth:`.firstorder.term.Term.vars`.
        """
        poly = self._poly
        context = poly.context()
        if poly.is_zero():  # compare https://github.com/flintlib/python-flint/issues/368
            unused = set(context.names())
        else:
            unused = set(poly.unused_gens())
        for gen in context.gens():
            if str(gen) not in unused:
                yield Variable.from_raw(gen)

class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):


    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.term.Variable.fresh`.
        """
        return VV.fresh(suffix=f'_{str(self)}')

    @classmethod
    def from_raw(cls, poly: fmpq_mpoly) -> Self:
        assert poly in poly.context().gens(), f'{poly} is not a generator'
        return super().from_raw(poly)


from logic1.theories.RCF.atomic import Eq, Ge, Gt, Le, Lt, Ne