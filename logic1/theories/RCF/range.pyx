# cython: profile=True
# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE_NOGIL=1
# type: ignore

from __future__ import annotations
import functools
from typing import Collection, Final, Iterable, Iterator, Optional, Self

import cython

# from cython.cimports.gmpy2 import import_gmpy2, mpq, mpq_t, mpfr
from cython.cimports.gmpy2 import *

cdef extern from "gmp.h":  # noqa
    void mpq_add(mpq_t sum, const mpq_t addend1, const mpq_t addend2)
    int mpq_cmp(const mpq_t op1, const mpq_t op2)
    int mpq_equal(const mpq_t op1, const mpq_t op2)
    void mpq_init(mpq_t x)
    void mpq_mul(mpq_t product, const mpq_t multiplier, const mpq_t multiplicand)
    void mpq_set(mpq_t rop, const mpq_t op)
    void mpq_set_z(mpq_t rop, const mpz_t op)
    int mpq_sgn(const mpq_t op)

cdef extern from "mpfr.h":  # noqa
    int mpfr_add(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_add_q(mpfr_t rop, mpfr_t op1, mpq_t op2, mpfr_rnd_t rnd)
    int mpfr_equal_p(mpfr_t op1, mpfr_t op2)
    int mpfr_less_p(mpfr_t op1, mpfr_t op2)
    int mpfr_mul(mpfr_t rop, mpfr_t op1, mpfr_t op2, mpfr_rnd_t rnd)
    int mpfr_mul_q(mpfr_t rop, mpfr_t op1, mpq_t op2, mpfr_rnd_t rnd)
    int mpfr_sgn(mpfr_t op)

import_gmpy2()

MPQ_ZERO: Final[mpq] = mpq(0)


@cython.total_ordering
class EndPoint:

    finite: cython.bint = cython.declare(cython.bint, visibility='readonly')
    finite_value: Optional[mpq] = cython.declare(Optional[mpq], visibility='readonly')
    infinite_value: Optional[mpfr] = cython.declare(Optional[mpfr], visibility='readonly')

    def __add__(self, other: EndPoint) -> EndPoint:
        if self.finite and other.finite:
            f1 = MPQ(self.finite_value)
            f2 = MPQ(other.finite_value)
            sum_ = GMPy_MPQ_New(NULL)
            mpq_add(MPQ(sum_), f1, f2)
            return EndPoint_Finite(sum_)
        elif self.finite and not other.finite:
            f1 = MPQ(self.finite_value)
            i2 = MPFR(other.infinite_value)
            sum_ = GMPy_MPFR_New(0, NULL)
            mpfr_add_q(MPFR(sum_), i2, f1, MPFR_RNDN)
            return EndPoint_Infinite(sum_)
        elif not self.finite and other.finite:
            i1 = MPFR(self.infinite_value)
            f2 = MPQ(other.finite_value)
            sum_ = GMPy_MPFR_New(0, NULL)
            mpfr_add_q(MPFR(sum_), i1, f2, MPFR_RNDN)
            return EndPoint_Infinite(sum_)
        else:
            i1 = MPFR(self.infinite_value)
            i2 = MPFR(other.infinite_value)
            sum_ = GMPy_MPFR_New(0, NULL)
            mpfr_add(MPFR(sum_), i1, i2, MPFR_RNDN)
            return EndPoint_Infinite(sum_)

    def __cinit__(self, finite: cython.bint, finite_value: Optional[mpq],
                  infinite_value: Optional[mpfr]) -> None:
        self.finite = finite
        self.finite_value = finite_value
        self.infinite_value = infinite_value

    def __eq__(self, other: EndPoint) -> cython.bint:
        if self.finite != other.finite:
            return False
        if self.finite:
            return mpq_equal(MPQ(self.finite_value), MPQ(other.finite_value))
        else:
            return mpfr_equal_p(MPFR(self.infinite_value), MPFR(other.infinite_value))

    def __hash__(self) -> int:
        return hash((self.finite, self.finite_value, self.infinite_value))

    def __lt__(self, other: EndPoint) -> cython.bint:
        if self.finite and other.finite:
            return mpq_cmp(MPQ(self.finite_value), MPQ(other.finite_value)) < 0
        elif self.finite and not other.finite:
            return 0 < mpfr_sgn(MPFR(other.infinite_value))
        elif not self.finite and other.finite:
            return mpfr_sgn(MPFR(self.infinite_value)) < 0
        else:
            return mpfr_less_p(MPFR(self.infinite_value), MPFR(other.infinite_value))

    def __mul__(self, other: EndPoint) -> EndPoint:
        # Treat multiplication with zero as a special case, defining oo * 0 =
        # -oo * 0 = 0.
        if self.finite and mpq_equal(MPQ(self.finite_value), MPQ(MPQ_ZERO)):
            return self
        elif other.finite and mpq_equal(MPQ(other.finite_value), MPQ(MPQ_ZERO)):
            return other
        # Complete case distinction on the regular cases:
        if self.finite and other.finite:
            f1 = MPQ(self.finite_value)
            f2 = MPQ(other.finite_value)
            product = GMPy_MPQ_New(NULL)
            mpq_mul(MPQ(product), f1, f2)
            return EndPoint_Finite(product)
        elif self.finite and not other.finite:
            f1 = MPQ(self.finite_value)
            i2 = MPFR(other.infinite_value)
            product = GMPy_MPFR_New(0, NULL)
            mpfr_mul_q(MPFR(product), i2, f1, MPFR_RNDN)
            return EndPoint_Infinite(product)
        elif not self.finite and other.finite:
            i1 = MPFR(self.infinite_value)
            f2 = MPQ(other.finite_value)
            product = GMPy_MPFR_New(0, NULL)
            mpfr_mul_q(MPFR(product), i1, f2, MPFR_RNDN)
            return EndPoint_Infinite(product)
        else:
            i1 = MPFR(self.infinite_value)
            i2 = MPFR(other.infinite_value)
            product = GMPy_MPFR_New(0, NULL)
            mpfr_mul(MPFR(product), i1, i2, MPFR_RNDN)
            return EndPoint_Infinite(product)

    def __neg__(self) -> EndPoint:
        if self.finite:
            return EndPoint_Finite(-self.finite_value)
        else:
            return EndPoint_Infinite(-self.infinite_value)

    def __pow__(self, n: int) -> EndPoint:
        if self.finite:
            return EndPoint_Finite(self.finite_value ** n)
        else:
            return EndPoint_Infinite(self.infinite_value ** n)

    def __repr__(self) -> str:
        return f'EndPoint({self.finite=!r}, {self.finite_value=!r}, {self.infinite_value=!r})'

    def __str__(self) -> str:
        if self.finite:
            return str(self.finite_value)
        if self.infinite_value == mpfr('inf'):
            return 'oo'
        if self.infinite_value == -mpfr('inf'):
            return '-oo'
        raise ValueError(self)

    @cython.cfunc
    def sgn(self) -> cython.int:
        if self.finite:
            return mpq_sgn(MPQ(self.finite_value))
        else:
            return mpfr_sgn(MPFR(self.infinite_value))


@cython.ccall
def EndPoint_Finite(value: mpq) -> EndPoint:
    return EndPoint.__new__(EndPoint, True, value, None)


@cython.cfunc
def EndPoint_Infinite(value: mpfr) -> EndPoint:
    assert value == mpfr('inf') or value == -mpfr('inf'), value
    return EndPoint.__new__(EndPoint, False, None, value)


oo: Final[EndPoint] = EndPoint_Infinite(mpfr('inf'))
ZERO: Final[EndPoint] = EndPoint_Finite(mpq(0))


@cython.final
@cython.cclass
class _Range:
    r"""Non-empty range IVL \ EXC, where IVL is an interval with boundaries in
    Q extended by {-oo, oo}, and EXC is a finite subset of the interior of
    IVL. Raises ValueError if IVL \ EXC gets empty.
    """

    lopen: cython.bint = cython.declare(cython.bint, visibility='readonly')
    start: EndPoint = cython.declare(EndPoint, visibility='readonly')
    end: EndPoint = cython.declare(EndPoint, visibility='readonly')
    ropen: cython.bint = cython.declare(cython.bint, visibility='readonly')
    exc: set[EndPoint] = cython.declare(object, visibility='readonly')

    def __add__(self, other: _Range) -> _Range:
        r"""The Minkowski sum.

        >>> def EP(*args):
        ...     from gmpy2 import mpq
        ...     return EndPoint_Finite(mpq(*args))

        >>> print(_Range(True, EP(1), EP(3), False, {EP(2)})
        ...     + _Range(True, EP(4), EP(6), False, {EP(5)}))
        (5, 9]

        >>> print(_Range.from_constant(EP(1)) + _Range(True, EP(4), EP(6), False, {EP(5)}))
        (5, 7] \ {6}
        """
        start = self.start + other.start
        lopen = self.lopen or other.lopen
        end = self.end + other.end
        ropen = self.ropen or other.ropen
        if self.is_point():
            assert self.start.finite
            exc = {point + self.start for point in other.exc}
        elif other.is_point():
            assert other.start.finite
            exc = {point + other.start for point in self.exc}
        else:
            exc = set()
        return _Range(lopen=lopen, start=start, end=end, ropen=ropen, exc=exc)

    def __contains__(self, ep: EndPoint) -> cython.bint:
        assert ep.finite
        if ep < self.start or (self.lopen and ep == self.start):
            return False
        if self.end < ep or (self.ropen and ep == self.end):
            return False
        if ep in self.exc:
            return False
        return True

    def __mul__(self, other: _Range) -> _Range:
        return self.x__mul__(other)

    def x__mul__(self, other: _Range) -> _Range:
        if self.is_zero() or other.is_zero():
            return _Range.from_constant(ZERO)
        elif self.is_point():
            return other.transform(self.start.finite_value, MPQ_ZERO)
        elif other.is_point():
            return self.transform(other.start.finite_value, MPQ_ZERO)
        negate = False
        if self.end.sgn() <= 0:
            self = -self
            negate = not negate
        if other.end.sgn() <= 0:
            other = -other
            negate = not negate
        if self.start.sgn() > other.start.sgn():
            self, other = other, self
        s1 = self.start.sgn()
        s2 = self.end.sgn()
        t1 = other.start.sgn()
        t2 = other.end.sgn()
        assert s2 > 0
        assert t2 > 0
        assert s1 <= t1
        if s1 < 0 and t1 < 0:
            # ZERO is in the interior of both self and other
            l1 = self.lopen or other.ropen
            p1 = self.start * other.end
            l2 = self.ropen or other.lopen
            p2 = self.end * other.start
            if p1 < p2:
                lopen = l1
                start = p1
            elif p2 < p1:
                lopen = l2
                start = p2
            else:  # p1 == p2
                lopen = l1 and l2
                start = p1
            r1 = self.lopen or other.lopen
            q1 = self.start * other.start
            r2 = self.ropen or other.ropen
            q2 = self.end * other.end
            if q1 < q2:
                ropen = r2
                end = q2
            elif q1 > q2:
                ropen = r1
                end = q1
            else:  # q1 == q2
                ropen = r1 and r2
                end = q1
        elif s1 < 0 and t1 == 0:
            #          0
            #       (  |  )
            #          (     )
            lopen = self.lopen or other.ropen
            start = self.start * other.end
            ropen = self.ropen or other.ropen
            end = self.end * other.end
        elif s1 < 0 and t1 > 0:
            #          0
            #       (  |  )
            #            (       )
            lopen = self.lopen or other.ropen
            start = self.start * other.end
            ropen = self.ropen or other.ropen
            end = self.end * other.end
        elif s1 == 0 and t1 == 0:
            #       0
            #       (    )
            #       (       )
            lopen = ((self.lopen or other.lopen) and (self.lopen or other.ropen)
                     and (self.ropen and other.lopen))
            start = ZERO
            ropen = self.ropen or other.ropen
            end = self.end * other.end
        elif s1 == 0 and t1 > 0:
            lopen = self.lopen or (other.lopen and other.ropen)
            start = ZERO
            assert s2 > 0
            end = self.end * other.end
            ropen = self.ropen or other.ropen
        elif s1 > 0 and t1 > 0:
            lopen = self.lopen or other.lopen
            start = self.start * other.start
            end = self.end * other.end
            ropen = self.ropen or other.ropen
        else:
            assert False, (s1, s2, t1, t2)
        if ZERO not in self and ZERO not in other and start < ZERO < end:
            exc = {ZERO}
        else:
            exc = set()
        result = _Range(lopen=lopen, start=start, end=end, ropen=ropen, exc=exc)
        if negate:
            result = -result
        return result

    def __init__(self, lopen: cython.bint = True, start: EndPoint = -oo, end: EndPoint = oo,
                 ropen: cython.bint = True, exc: set = set()) -> None:
        assert lopen or start is not -oo, self
        assert ropen or end is not oo, self
        assert all(start < x < end for x in exc), self
        if start > end:
            raise ValueError("_Range cannot be empty")
        if (lopen or ropen) and start == end:
            raise ValueError("_Range cannot be empty")
        self.lopen = lopen
        self.start = start
        self.end = end
        self.ropen = ropen
        self.exc = exc

    def __neg__(self) -> _Range:
        return _Range(self.ropen, -self.end, -self.start, self.lopen, {-ep for ep in self.exc})

    def __pow__(self, n: int) -> _Range:
        r"""Exponentiation. Computes {x ** n for x in self}. Note that this is
        not based on Minkowski multiplication :meth:`__mul__`.

        >>> def EP(*args):
        ...     from gmpy2 import mpq
        ...     return EndPoint_Finite(mpq(*args))

        >>> print(_Range(True, ZERO, oo, True, {EP(1), EP(2)}) ** 0)
        [1, 1]

        >>> print(_Range(False, EP(-2), EP(1), True, {EP(-1)}) ** 2)
        [0, 4] \ {1}

        >>> print(_Range(True, EP(-1), EP(2), False, {EP(1)}) ** 2)
        [0, 4] \ {1}

        >>> print(_Range(False, EP(-3), EP(2), False, {EP(1)}) ** 2)
        [0, 9]

        >>> print(_Range(False, EP(-3), EP(2), False, {EP(-1), EP(1)}) ** 2)
        [0, 9] \ {1}

        >>> print(_Range(False, EP(1), EP(2), True) ** 3)
        [1, 8)

        >>> r = _Range(True, EP(-3), EP(-1), False, {EP(-2)})
        >>> print(r ** 1)
        (-3, -1] \ {-2}

        >>> print(r ** 2)
        [1, 9) \ {4}

        >>> r = _Range(True, EP(-3), EP(2), False, {EP(-2)})
        >>> print(r ** 2)
        [0, 9)
        >>> print(r ** 3)
        (-27, 8] \ {-8}

        >>> print(_Range(False, EP(-1), EP(2), False, {EP(0), EP(1)}) ** 2)
        (0, 4]

        >>> print(_Range(True, EP(-1), EP(2), False, {EP(0), EP(1)}) ** 2)
        (0, 4] \ {1}
        """
        if n == 0:
            return self.from_constant(EndPoint_Finite(mpq(1)))
        if n % 2 == 0:
            self = self.abs()
        start = self.start ** n
        end = self.end ** n
        exc = {p ** n for p in self.exc}
        return _Range(self.lopen, start, end, self.ropen, exc)

    def __repr__(self) -> cython.unicode:
        return f'_Range(lopen={self.lopen}, start={self.start!r}, end={self.end!r}, ' \
               f'ropen={self.ropen}, exc={self.exc})'

    def __str__(self) -> cython.unicode:
        left = '(' if self.lopen else '['
        start = str(self.start)
        end = str(self.end)
        right = ')' if self.ropen else ']'
        exc_entries = {str(q) for q in self.exc}
        if exc_entries:
            exc = f' \\ {{{", ".join(exc_entries)}}}'
        else:
            exc = ''
        return f'{left}{start}, {end}{right}{exc}'

    def abs(self) -> _Range:
        if self.start >= ZERO:
            return self
        if self.end <= ZERO:
            return _Range(self.ropen, -self.end, -self.start, self.lopen, {-p for p in self.exc})
        non_negative = _Range(
            ZERO in self.exc, ZERO, self.end, self.ropen, {p for p in self.exc if p > ZERO})
        abs_of_negative = _Range(
            True, ZERO, -self.start, self.lopen, {-p for p in self.exc if p < ZERO})
        return non_negative.union(abs_of_negative)

    @classmethod
    def from_constant(cls, ep: EndPoint) -> _Range:
        assert ep.finite
        return _Range(lopen=False, start=ep, end=ep, ropen=False, exc=set())

    def intersection(self, other: _Range) -> _Range:
        if self.start < other.start:
            lopen = other.lopen
            start = other.start
        elif other.start < self.start:
            lopen = self.lopen
            start = self.start
        else:
            assert self.start == other.start
            start = self.start
            lopen = self.lopen or other.lopen
        if self.end < other.end:
            end = self.end
            ropen = self.ropen
        elif other.end < self.end:
            end = other.end
            ropen = other.ropen
        else:
            assert self.end == other.end
            end = self.end
            ropen = self.ropen or other.ropen
        # Fix the case that ivl is closed on either side and the corresponding
        # endpoint is in self.exc or other.exc
        if start in self.exc or start in other.exc:
            lopen = True
        if end in self.exc or end in other.exc:
            ropen = True
        exc = set()
        for x in self.exc:
            if start < x < end:
                exc.add(x)
        for x in other.exc:
            if start < x < end:
                exc.add(x)
        return _Range(lopen, start, end, ropen, exc)

    def is_disjoint(self, other: _Range) -> cython.bint:
        if self.end < other.start:
            return True
        if self.end == other.start and (self.lopen or other.lopen):
            return True
        if self.start > other.end:
            return True
        if self.start == other.end and (self.ropen or other.ropen):
            return True
        if self.is_point() and self.start in other.exc:
            return True
        if other.is_point() and other.start in self.exc:
            return True
        return False

    @cython.ccall
    def is_point(self) -> cython.bint:
        # It is assumed and has been asserted that the interval is not empty.
        return self.start == self.end

    def is_subset(self, other: _Range) -> cython.bint:
        if self.start < other.start:
            return False
        if self.start == other.start and not self.lopen and other.lopen:
            return False
        if self.end > other.end:
            return False
        if self.end == other.end and not self.ropen and other.ropen:
            return False
        for q in other.exc:
            if q in self:
                return False
        return True

    @cython.cfunc
    def is_zero(self) -> cython.bint:
        return self.is_point() and self.start == ZERO

    def minkowski_pow(self, n: int) -> _Range:
        """Compute a superset of the n-fold Minkowski product.

        >>> def EP(*args):
        ...     from gmpy2 import mpq
        ...     return EndPoint_Finite(mpq(*args))

        >>> r = _Range(False, EP(1), EP(2), True)
        >>> print(r.minkowski_pow(0))
        [1, 1]
        >>> print(r.minkowski_pow(1))
        [1, 2)
        >>> print(r.minkowski_pow(6))
        [1, 64)

        >>> r = _Range(True, -oo, oo, True, set())
        >>> print(r.minkowski_pow(6))
        (-oo, oo)

        >>> r = _Range(True, -oo, EP(0), True, set())
        >>> print(r.minkowski_pow(2))
        (0, oo)
        >>> print(r.minkowski_pow(3))
        (-oo, 0)

        .. seealso:: :meth:`__mul__ <.simplifiy._Range.__mul__>` -- Minkowski product
        """
        if n == 0:
            return self.from_constant(EndPoint_Finite(mpq(1)))
        result = self.minkowski_pow(n // 2)
        result = result * result
        if n % 2 == 1:
            result = result * self
        return result

    # @staticmethod
    # def from_term(f: Term, knowl: _Knowledge) -> _Range:
    #     """
    #     >>> from logic1.theories.RCF import VV
    #     >>> x, y = VV.get('x', 'y')
    #     >>> print(_Range.from_term(Term(0), _Knowledge()))
    #     [0, 0]
    #     >>> f = x**2 + y**2
    #     >>> print(_Range.from_term(f, _Knowledge()))
    #     [0, oo)
    #     >>> g = -x**2 - y**2 - 1
    #     >>> print(_Range.from_term(g, _Knowledge()))
    #     (-oo, -1]
    #     >>> h = (x - y) ** 2
    #     >>> print(_Range.from_term(h, _Knowledge()))
    #     (-oo, oo)
    #     >>> print(_Range.from_term(h, _Knowledge(
    #     ...    {x: _Range(True, mpq(0), oo, True), y: _Range(True, -oo, mpq(0), True)})))
    #     (0, oo)
    #     >>> print(_Range.from_term(h, _Knowledge(
    #     ...    {x: _Range(True, -oo, mpq(0), False), y: _Range(False, mpq(0), oo, True)})))
    #     [0, oo)
    #     """
    #     R = _Range(True, -oo, oo, True, set())
    #     poly_result = _Range.from_constant(mpq(0))
    #     gens = f.poly.parent().gens()
    #     for exponent, coefficient in f.poly.dict().items():
    #         term_result = _Range.from_constant(mpq(coefficient))
    #         for g, e in zip(gens, exponent):
    #             ge_result = knowl.get(Term(g)) ** e
    #             term_result = term_result * ge_result
    #         poly_result = poly_result + term_result
    #         if poly_result == R:
    #             return R
    #     return poly_result

    @cython.ccall
    def transform(self, scale_mpq: mpq, shift_mpq: mpq) -> _Range:
        scale = EndPoint_Finite(scale_mpq)
        shift = EndPoint_Finite(shift_mpq)
        if scale > ZERO:
            lopen = self.lopen
            start = scale * self.start + shift
            end = scale * self.end + shift
            ropen = self.ropen
        elif scale < ZERO:
            lopen = self.ropen
            start = scale * self.end + shift
            end = scale * self.start + shift
            ropen = self.lopen
        else:
            raise ValueError('scaling by 0 is not supported')
        exc = {scale * point + shift for point in self.exc}
        return _Range(lopen, start, end, ropen, exc)

    @cython.ccall
    def union(self, other: _Range) -> _Range:
        if (other.start, other.lopen) < (self.start, self.lopen):
            self, other = other, self
        final_exc = set()
        if (other.end, not other.ropen) < (self.end, not self.ropen):
            end, ropen = self.end, self.ropen
        else:
            end, ropen = other.end, other.ropen
            if self.end < other.start:
                raise ValueError("union is not a _Range")
            if self.end == other.start and self.ropen and other.lopen:
                assert self.end.finite
                final_exc.add(self.end)
        for p in self.exc:
            if p not in other:
                final_exc.add(p)
        for p in other.exc:
            if p not in self:
                final_exc.add(p)
        return _Range(self.lopen, self.start, end, ropen, final_exc)
