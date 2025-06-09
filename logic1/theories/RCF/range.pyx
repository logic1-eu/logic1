# cython: profile=False
# cython: linetrace=False
# distutils: define_macros=CYTHON_TRACE_NOGIL=0
# type: ignore

from __future__ import annotations
from typing import Final, Optional

import cython

from cython.cimports.gmpy2 import GMPy_MPQ_New, import_gmpy2, MPQ, mpq, mpq_t

cdef extern from "gmp.h":
    void mpq_add(mpq_t sum, const mpq_t addend1, const mpq_t addend2)
    int mpq_cmp(const mpq_t op1, const mpq_t op2)
    int mpq_equal(const mpq_t op1, const mpq_t op2)
    void mpq_init(mpq_t x)
    void mpq_mul(mpq_t product, const mpq_t multiplier, const mpq_t multiplicand)
    void mpq_set(mpq_t rop, const mpq_t op)
    int mpq_sgn(const mpq_t op)

import_gmpy2()

mpq_ZERO: Final[mpq] = mpq(0)


@cython.final
@cython.cclass
class EndPoint:
    # This class assumed to remain immutable.

    finite_value: Optional[mpq] = cython.declare(Optional[mpq], visibility='readonly')
    infinite_sign: cython.int = cython.declare(cython.int, visibility='readonly')

    def __add__(self, other: EndPoint) -> EndPoint:
        if self.finite_value is not None and other.finite_value is not None:
            f1 = MPQ(self.finite_value)
            f2 = MPQ(other.finite_value)
            sum_ = GMPy_MPQ_New(NULL)
            mpq_add(MPQ(sum_), f1, f2)
            return EndPoint.__new__(EndPoint, sum_)
        elif self.finite_value is not None and other.finite_value is None:
            return other
        elif self.finite_value is None and other.finite_value is not None:
            return self
        else:
            if self.infinite_sign != other.infinite_sign:
                raise ValueError(f'addition between {self!s} and {other!s} is not supported')
            return self

    def __cinit__(self, finite_value: Optional[mpq], infinite_sign: cython.int = 1) -> None:
        assert finite_value is None or infinite_sign == 1
        assert infinite_sign == 1 or infinite_sign == -1
        self.finite_value = finite_value
        self.infinite_sign = infinite_sign

    def __eq__(self, other: EndPoint) -> cython.bint:
        if (self.finite_value is not None) != (other.finite_value is not None):
            return False
        if self.finite_value is not None:
            return mpq_equal(MPQ(self.finite_value), MPQ(other.finite_value))
        else:
            return self.infinite_sign == other.infinite_sign

    def __ge__(self, other: EndPoint) -> cython.bint:
        return not (self < other)

    def __gt__(self, other: EndPoint) -> cython.bint:
        return other < self

    def __hash__(self) -> int:
        return hash((self.finite_value, self.infinite_sign))

    def __le__(self, other: EndPoint) -> cython.bint:
        return not (other < self)

    def __lt__(self, other: EndPoint) -> cython.bint:
        if self.finite_value is not None and other.finite_value is not None:
            return mpq_cmp(MPQ(self.finite_value), MPQ(other.finite_value)) < 0
        elif self.finite_value is not None and other.finite_value is None:
            return other.infinite_sign == 1
        elif self.finite_value is None and other.finite_value is not None:
            return self.infinite_sign == -1
        else:
            return self.infinite_sign < other.infinite_sign

    def __mul__(self, other: EndPoint) -> EndPoint:
        # Treat multiplication with zero as a special case, defining oo * 0 =
        # -oo * 0 = 0.
        if self.finite_value is not None and mpq_equal(MPQ(self.finite_value), MPQ(mpq_ZERO)):
            return self
        if other.finite_value is not None and mpq_equal(MPQ(other.finite_value), MPQ(mpq_ZERO)):
            return other
        # Complete case distinction on the regular cases:
        if self.finite_value is not None and other.finite_value is not None:
            f1 = MPQ(self.finite_value)
            f2 = MPQ(other.finite_value)
            product = GMPy_MPQ_New(NULL)
            mpq_mul(MPQ(product), f1, f2)
            return EndPoint.__new__(EndPoint, product)
        elif self.finite_value is not None and other.finite_value is None:
            finite_sign: cython.int = mpq_sgn(MPQ(self.finite_value))
            return EndPoint.__new__(EndPoint, None, finite_sign * other.infinite_sign)
        elif self.finite_value is None and other.finite_value is not None:
            finite_sign: cython.int = mpq_sgn(MPQ(other.finite_value))
            return EndPoint.__new__(EndPoint, None, finite_sign * self.infinite_sign)
        else:
            return EndPoint.__new__(EndPoint, None, self.infinite_sign * other.infinite_sign)

    def __neg__(self) -> EndPoint:
        if self.finite_value is not None:
            return EndPoint.__new__(EndPoint, -self.finite_value)
        else:
            return EndPoint.__new__(EndPoint, None, -self.infinite_sign)

    def __pow__(self, n: int) -> EndPoint:
        if self.finite_value is not None:
            return EndPoint.__new__(EndPoint, self.finite_value ** n)
        else:
            return EndPoint.__new__(EndPoint, None, self.infinite_sign ** n)

    def __repr__(self) -> str:
        return f'EndPoint({self.finite_value=!r}, {self.infinite_sign=!r})'

    def __str__(self) -> str:
        if self.finite_value is not None:
            return str(self.finite_value)
        elif self.infinite_sign == 1:
            return 'oo'
        else:
            assert self.infinite_sign == -1
            return '-oo'

    @cython.ccall
    def is_finite(self) -> cython.bint:
        return self.finite_value is not None

    @cython.cfunc
    def sgn(self) -> cython.int:
        if self.finite_value is not None:
            return mpq_sgn(MPQ(self.finite_value))
        else:
            return self.infinite_sign


oo: Final[EndPoint] = EndPoint(None, 1)
ZERO: Final[EndPoint] = EndPoint(mpq_ZERO)


@cython.final
@cython.cclass
class _Range:
    r"""Non-empty range IVL \ EXC, where IVL is an interval with boundaries in
    Q extended by {-oo, oo}, and EXC is a is_finite() subset of the interior of
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
        ...     return EndPoint(mpq(*args))

        >>> print(_Range(True, EP(1), EP(3), False, {EP(2)})
        ...     + _Range(True, EP(4), EP(6), False, {EP(5)}))
        (5, 9]

        >>> print(_Range.from_constant(EP(1)) + _Range(True, EP(4), EP(6), False, {EP(5)}))
        (5, 7] \ {6}
        """
        ret = self.copy()
        ret += other
        return ret

    def __iadd__(self, other: _Range) -> _Range:
        if self.is_point():
            assert self.start.finite_value is not None
            exc = {point + self.start for point in other.exc}
        elif other.is_point():
            assert other.start.finite_value is not None
            exc = {point + other.start for point in self.exc}
        else:
            exc = set()
        self.lopen = self.lopen or other.lopen
        self.start += other.start
        self.ropen = self.ropen or other.ropen
        self.end += other.end
        self.exc = exc
        return self

    def __contains__(self, ep: EndPoint) -> cython.bint:
        assert ep.finite_value is not None
        if ep < self.start or (self.lopen and ep == self.start):
            return False
        if self.end < ep or (self.ropen and ep == self.end):
            return False
        if ep in self.exc:
            return False
        return True

    def __eq__(self, other: _Range) -> cython.bint:
        if self.lopen != other.lopen:
            return False
        if self.ropen != other.ropen:
            return False
        if self.start != other.start:
            return False
        if self.end != other.end:
            return False
        if self.exc != other.exc:
            return False
        return True

    def __mul__(self, other: _Range) -> _Range:
        ret = self.copy()
        ret *= other
        return ret

    def __imul__(self, other: _Range) -> _Range:
        if self.is_zero():
            return self
        if other.is_zero():
            return self.iset(other)
        elif self.is_point():
            h = self.start
            self.iset(other)
            return self.iscale(h)
        elif other.is_point():
            return self.iscale(other.start)
        negate = False
        if self.end.sgn() <= 0:
            self.ineg()
            negate = not negate
        if other.end.sgn() <= 0:
            other = -other
            negate = not negate
        if self.start.sgn() > other.start.sgn():
            tmp = self.copy()
            self.iset(other)
            other = tmp
        self._imul_core(other)
        if negate:
            self.ineg()
        return self
    
    @cython.cfunc
    def _imul_core(self, other: _Range) -> cython.void:
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
        self.lopen = lopen
        self.start = start
        self.end = end
        self.ropen = ropen
        self.exc = exc

    def __init__(self, lopen: cython.bint = True, start: EndPoint = -oo, end: EndPoint = oo,
                 ropen: cython.bint = True, exc: set = set()) -> None:
        assert lopen or start != -oo, self
        assert ropen or end != oo, self
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
        return self.copy().ineg()

    def __pow__(self, n: int) -> _Range:
        r"""Exponentiation. Computes {x ** n for x in self}. Note that this is
        not based on Minkowski multiplication :meth:`__mul__`.

        >>> def EP(*args):
        ...     from gmpy2 import mpq
        ...     return EndPoint(mpq(*args))

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
            return self.from_constant(EndPoint(mpq(1)))
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
    
    def copy(self) -> _Range:
        return _Range(self.lopen, self.start, self.end, self.ropen, self.exc)

    @classmethod
    def from_constant(cls, ep: EndPoint) -> _Range:
        assert ep.finite_value is not None
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

    @cython.ccall
    def is_zero(self) -> cython.bint:
        return self.is_point() and self.start == ZERO
    
    @cython.ccall
    def ineg(self) -> _Range:
        self.lopen, self.ropen = self.ropen, self.lopen
        self.start, self.end = -self.end, -self.start
        self.exc = {-ep for ep in self.exc}
        return self
    
    @cython.cfunc    
    def iset(self, other: _Range) -> _Range:
        self.lopen = other.lopen
        self.start = other.start
        self.end = other.end
        self.ropen = other.ropen
        self.exc = other.exc
        return self

    def minkowski_pow(self, n: int) -> _Range:
        """Compute a superset of the n-fold Minkowski product.

        >>> def EP(*args):
        ...     from gmpy2 import mpq
        ...     return EndPoint(mpq(*args))

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
            return self.from_constant(EndPoint(mpq(1)))
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

    @cython.cfunc
    def scale(self, ep: EndPoint) -> _Range:
        assert ep != ZERO
        exc = {ep * point for point in self.exc}
        if ep > ZERO:
            return _Range(self.lopen, ep * self.start, ep * self.end, self.ropen, exc)
        else:
            return _Range(self.ropen, ep * self.end, ep * self.start, self.lopen, exc)

    @cython.cfunc
    def iscale(self, ep: EndPoint) -> _Range:
        assert ep != ZERO
        if ep > ZERO:
            self.start *= ep
            self.end *= ep
        else:
            self.lopen, self.ropen = self.ropen, self.lopen
            self.start, self.end = self.end * ep, self.start * ep
        self.exc = {ep * point for point in self.exc}
        return self

    @cython.ccall
    def transform(self, scale_mpq: mpq, shift_mpq: mpq) -> _Range:
        scale = EndPoint(scale_mpq)
        shift = EndPoint(shift_mpq)
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
                assert self.end.finite_value is not None
                final_exc.add(self.end)
        for p in self.exc:
            if p not in other:
                final_exc.add(p)
        for p in other.exc:
            if p not in self:
                final_exc.add(p)
        return _Range(self.lopen, self.start, end, ropen, final_exc)
