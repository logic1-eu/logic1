# type: ignore
from fractions import Fraction
from cython.cimports.gmpy2 import import_gmpy2, MPQ, mpq

import cython

import_gmpy2()

siInit("/Users/sturm/miniforge3/envs/logic1_dev/lib/libSingular.dylib")


@cython.cclass
class Ring:

    _ring = cython.declare(cython.pointer[ring])

    def __init__(self, generator_names: list[str]):
        n: cython.int = len(generator_names)

        names = cython.cast(cython.pp_char, omAlloc0(n * cython.sizeof(cython.p_char)))
        for i in range(n):
            names[i] = omStrDup(generator_names[i].encode())

        # There is an issue with enums in Pure Python mode:
        # https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#structs-unions-enums
        cdef rRingOrder_t *order = <rRingOrder_t *>omAlloc0(2 * cython.sizeof(rRingOrder_t))
        order[0] = ringorder_Dp
        order[1] = ringorder_no

        block0 = cython.cast(cython.p_int, omAlloc0(2 * cython.sizeof(cython.int)))
        block0[0] = 1
        block0[1] = 0
    
        block1 = cython.cast(cython.p_int, omAlloc0(2 * cython.sizeof(cython.int)))
        block1[0] = n
        block1[1] = 0

        self._ring = rDefault(0, n, names, 1, order, block0, block1, cython.NULL)

        # check "ShortOut"

    def gen(self, n: int) -> Term:
        rChangeCurrRing(self._ring)
        _p = p_ISet(1, self._ring)
        p_SetExp(_p, n+1, 1, self._ring)
        p_Setm(_p, self._ring)
        return term(_p, self._ring)

    def print(self):        
        rPrint(self._ring)
        print()


@cython.cclass
class Term:

    _poly = cython.declare(cython.pointer[poly])
    _ring = cython.declare(cython.pointer[ring])

    def __add__(self, other: Term) -> Term:
        assert self._ring == other._ring
        ring = self._ring
        p1 = p_Copy(self._poly, ring)
        p2 = p_Copy(other._poly, ring)
        sum = p_Add_q(p1, p2, ring)
        return term(sum, ring)
    
    def __init__(self, arg: Optional[Fraction | int | mpq] = None,
                 r: Optional[Ring] = None) -> None:
        if arg is None:
            self._poly = cython.NULL
            self._ring = cython.NULL
        elif isinstance(arg, mpq):
            self._init_mpq(arg, r)
        else:
            raise ValueError(f'expected Fraction, int, or mpq; {arg} is {type(arg)}')
    
    def _init_mpq(self, q: mpq, r: Ring) -> None:
        _q = MPQ(q)
        n = nlInit2gmp(mpq_numref(_q), mpq_denref(_q), r._ring.cf)
        self._poly = p_NSet(n, r._ring)
        self._ring = r._ring

    def __mul__(self, other: Term) -> Term:
        assert self._ring == other._ring
        ring = self._ring
        e1: cython.ulong = p_GetMaxExp(self._poly, ring)
        e2: cython.ulong = p_GetMaxExp(other._poly, ring)
        e: cython.ulong = e1 + e2
        if unlikely(e > ring.bitmask):
            raise OverflowError(f'exponent overflow {e}')
        prod = pp_Mult_qq(self._poly, other._poly, ring)
        return term(prod, ring)
    
    def __pow__(self, n: int) -> Term:
        if unlikely(n < 0):
            raise ValueError(f'negative exponent {n}')
        elif n == 0:
            return term(p_ISet(1, self._ring), self._ring)
        else:
            ret = self ** (n // 2)
            ret = ret * ret
            if n % 2 == 1:
                ret = ret * self
            return ret

    def print(self):
        p_Write(self._poly, self._ring, self._ring)

@cython.cfunc
def term(poly: cython.pointer[poly], ring: cython.pointer[ring]) -> Term:
    t = Term()
    t._poly = poly
    t._ring = ring
    return t


def main():
    R = Ring(['x', 'y'])
    R.print()
    x = R.gen(0)
    y = R.gen(1)
    f = (x + y + Term(mpq(-1, 2), R)) ** 2
    x.print()
    y.print()
    f.print()
    g = f + Term(mpq(1, 2), R)
    g.print()
