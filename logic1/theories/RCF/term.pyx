# type: ignore
from fractions import Fraction
from typing import Final

from gmpy2 import mpq
from cysignals.signals cimport sig_on, sig_off
import cython
from cython.cimports.gmpy2 import GMPy_MPQ_New, GMPy_MPZ_New, import_gmpy2, MPQ, mpq, MPZ

from logic1 import firstorder

import_gmpy2()

siInit("/Users/sturm/miniforge3/envs/logic1_dev/lib/libSingular.dylib")


@cython.cclass
class Ring:

    _singular_ring = cython.declare(cython.pointer[ring])

    def __init__(self, generator_names: list[str]):
        n: cython.int = len(generator_names)

        if n == 0:
            raise ValueError('Ring requires at least one generator')

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

        ring = rDefault(0, n, names, 1, order, block0, block1, cython.NULL)
        if ring is cython.NULL:
            raise ValueError("Failed to allocate Singular ring")
        ring.ShortOut = 0  # disable Singular's short printing
        self._singular_ring = ring

        global current_ring
        current_ring = self

    def __repr__(self) -> str:
        names = ', '.join(repr(name) for name in self.get_names())
        return f'Ring([{names}])'

    def __str__(self) -> str:
        """The best Singular has to offer
        """
        return rString(self._singular_ring).decode()

    def get_names(self) -> Iterator(str):
        SR = self._singular_ring
        for name in SR.names[:SR.N]:
            yield name.decode()
    
    def get_var_by_index(self, index: int) -> Variable:
        assert index in range(self._singular_ring.N), f'Invalid index {index}'
        _p = p_ISet(1, self._singular_ring)
        p_SetExp(_p, index + 1, 1, self._singular_ring)
        p_Setm(_p, self._singular_ring)
        return variable(_p, self)

    def get_var_by_name(self, name: str) -> Variable:
        SR = self._singular_ring
        for index in range(SR.N):
            if SR.names[index].decode() == name:
                return self.get_var_by_index(index)
        assert False, f'Invalid name {name}'

    def get_vars(self) -> Iterator[Variable]:
        for index in range(self._singular_ring.N):
            yield self.get_var_by_index(index)

    @cython.cfunc
    def mpq_to_number(self, q: mpq) -> cython.pointer[number]:
        """Create a singular number from an mpq.
        """
        return nlInit2gmp(mpq_numref(MPQ(q)), mpq_denref(MPQ(q)), self._singular_ring.cf)
    
    @cython.cfunc
    def number_to_mpq(self, n: cython.pointer[number]) -> mpq:
        """Create an mpq from a singular number.
        """
        # Immediate integers handles carry the tag 'SR_INT', i.e. the last bit is 1.
        # This distinguishes immediate integers from other handles which point to
        # structures aligned on 4 byte boundaries and therefore have last bit zero.
        # (The second bit is reserved as tag to allow extensions of this scheme.)
        # Using immediates as pointers and dereferencing them gives address errors.

        ret = GMPy_MPQ_New(cython.NULL)
        tmp = GMPy_MPZ_New(cython.NULL)

        n_num = nlGetNumerator(n, self._singular_ring.cf)
        if SR_HDL(n_num) & SR_INT:
            mpz_set_si(MPZ(tmp), SR_TO_INT(n_num))
        else:
            mpz_set(MPZ(tmp), n_num.z)
        mpq_set_num(MPQ(ret), MPZ(tmp))
        nlDelete(cython.address(n_num), self._singular_ring.cf)

        n_den = nlGetDenom(n, self._singular_ring.cf)
        if SR_HDL(n_den) & SR_INT:
            mpz_set_si(MPZ(tmp), SR_TO_INT(n_den))
        else:
            mpz_set(MPZ(tmp), n_den.z)
        mpq_set_den(MPQ(ret), MPZ(tmp))
        nlDelete(cython.address(n_den), self._singular_ring.cf)

        return ret
    
    def rPrint(self):
        rPrint(self._singular_ring)
        print()
    

current_ring = cython.declare(Ring)


class VariableSet(firstorder.atomic.VariableSet['Variable']):
    """The infinite set of all variables belonging to the theory of Real Closed
    Fields. Variables are uniquely identified by their name, which is a
    :external:class:`.str`. This class is a singleton, whose single instance is
    assigned to :data:`.VV`.

    .. seealso::
        Final methods inherited from parent class:

        * :meth:`.firstorder.atomic.VariableSet.get`
            -- obtain several variables simultaneously
        * :meth:`.firstorder.atomic.VariableSet.imp`
            -- import variables into global namespace
    """

    _ring: Optional[Ring]
    _stack: list[Optional[Ring]]

    # required by the abstract parent class
    @property
    def stack(self) -> list[Optional[Ring]]:
        return self._stack
    
    def __getitem__(self, name: str) -> Variable:
        """Implements abstract method
        :meth:`.firstorder.atomic.VariableSet.__getitem__`.
        """
        if not isinstance(name, str):
            raise ValueError(f'expecting string as index; {name} is {type(name)}')
        self.add_vars((name,))
        return self._ring.get_var_by_name(name)
            
    def __init__(self) -> None:
        self._ring = None

    def __repr__(self) -> str:
        vars_ = self.polynomial_ring.get_vars()
        s = ', '.join(str(g) for g in (*vars_, '...'))
        return f'{{{s}}}'

    def add_vars(self, new_names: Iterable[str]) -> None:

        def sort_key(s: str) -> tuple[str, int]:
            base = s.rstrip('0123456789')
            index = s[len(base):]
            n = int(index) if index else -1
            return base, n

        if self._ring is None:
            names = []
        else:
            names = list(self._ring.get_names())
        have_appended = False
        for name in new_names:
            if name not in names:
                names.append(name)
                have_appended = True
        if have_appended:
            names.sort(key=sort_key)
            self._ring = Ring(names)

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        names = set(self._ring.get_names())
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in names:
            i += 1
            v = f'G{i:04d}{suffix}'
        self.add_vars((v,))
        return self._ring.get_var_by_name(v)

    def pop(self) -> None:
        from . import cache_clear
        ...
        cache_clear()

    def push(self) -> None:
        from . import cache_clear
        ...
        cache_clear()


VV: Final = VariableSet()
"""
The unique instance of :class:`.VariableSet`.
"""


@cython.cclass
class Term:

    _parent = cython.declare(Ring, visibility='readonly')
    _mpq = cython.declare(mpq)
    _poly = cython.declare(cython.pointer[poly])

    def __add__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self + Term(other)
        _other = cython.cast(Term, other)
        if current_ring is None:
            assert self._parent is None and _other._parent is None
            return Term(self._mpq + _other._mpq)
        self._coerce(current_ring)
        _other._coerce(current_ring)
        SR = current_ring._singular_ring
        p1 = p_Copy(self._poly, SR)
        p2 = p_Copy(_other._poly, SR)
        sum = p_Add_q(p1, p2, SR)
        return term(sum, current_ring)
    
    def __eq__(self, other: Term) -> cython.bint:
        if current_ring is None:
            assert self._parent is None and other._parent is None
            return self._mpq == other._mpq
        self._coerce(current_ring)
        other._coerce(current_ring)
        SR = current_ring._singular_ring
        ret: cython.bint = p_EqualPolys(self._poly, other._poly, SR)
        return ret
    
    assert hash(mpq(0)) == 0  # so that mpq(0) hashes equally to cython.NULL in __hash__

    def __hash__(self) -> int:
        if self._parent is None:
            return hash(self._mpq)

        SR: Final = self._parent._singular_ring
        h_names: Final = [hash(name) for name in SR.names[:SR.N]]
        ret: cython.long = 0
        p = self._poly
        while p:
            ret_mon: cython.long = hash(self._parent.number_to_mpq(p_GetCoeff(p, SR)))
            for v in range(1, SR.N + 1):
                n = p_GetExp(p, v, SR)
                if n != 0:
                    ret_mon = (1000003 * ret_mon) ^ h_names[v - 1]
                    ret_mon = (1000003 * ret_mon) ^ n
            ret += ret_mon
            p = pNext(p)
        return ret

    def __init__(self, arg: Fraction | int | mpq) -> None:
        """
        >>> Term(Fraction(1, 2))
        1/2
        >>> Term(42)
        42
        >>> Term(mpq(1, 2))
        1/2
        """
        if isinstance(arg, Fraction):
            q = mpq(arg.numerator, arg.denominator)
        elif isinstance(arg, int):
            q = mpq(arg)
        elif isinstance(arg, mpq):
            q = arg
        else:
            raise ValueError(f'expected Fraction, int, or mpq; {arg} is {type(arg)}')
        self._mpq = q
        self._parent = None
    
    def __mul__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self * Term(other)
        _other = cython.cast(Term, other)
        if current_ring is None:
            assert self._parent is None and _other._parent is None
            return Term(self._mpq * other._mpq)
        self._coerce(current_ring)
        other._coerce(current_ring)
        SR = current_ring._singular_ring
        e1: cython.ulong = p_GetMaxExp(self._poly, SR)
        e2: cython.ulong = p_GetMaxExp(_other._poly, SR)
        e: cython.ulong = e1 + e2
        if unlikely(e > SR.bitmask):
            raise OverflowError(f'exponent overflow {e}')
        prod = pp_Mult_qq(self._poly, _other._poly, SR)
        return term(prod, current_ring)
    
    def __neg__(self):
        if self._parent is None:
            return Term(-self._mpq)
        SR = self._parent._singular_ring
        p = p_Copy(self._poly, SR)
        negative = p_Neg(p, SR)
        return term(negative, self._parent)
        
    def __pow__(self, n: int) -> Term:
        if unlikely(n < 0):
            raise ValueError(f'negative exponent {n}')
        elif n == 0:
            return Term(1)
        else:
            ret = self ** (n // 2)
            ret = ret * ret
            if n % 2 == 1:
                ret = ret * self
            return ret

    def __radd__(self, other: object) -> Term:
        assert not isinstance(other, Term)
        return Term(other) + self

    def __repr__(self) -> str:
        if self._parent is None:
            return str(self._mpq)
        import re
        plusminus_pattern = re.compile(r"([^\(^])([\+\-])")
        parenthvar_pattern = re.compile(r"\(([a-zA-Z][a-zA-Z0-9]*)\)")
        p = self._poly
        SR = self._parent._singular_ring
        s = p_String(p, SR, SR).decode()
        s = plusminus_pattern.sub("\\1 \\2 ", s)
        s = parenthvar_pattern.sub("\\1", s)
        return s
    
    def __rmul__(self, other: object) -> Term:
        assert not isinstance(other, Term)
        return Term(other) * self

    def __rsub__(self, other: object) -> Term:
        assert not isinstance(other, Term)
        return Term(other) - self
    
    def __str__(self):
        MUL: Final = '*'
        POW: Final = '^'

        def _format_first(c: mpq, d) -> str:
            m = _format_mon(d)
            if c == mpq(1):
                return m if m else f'{c}'
            elif c == mpq(-1):
                return f'-{m}' if m else f'{c}'
            elif c != mpq(0):
                return f'{c}{MUL}{m}' if m else f'{c}'
            else:
                assert False, f'zero summand in {self!r}'
        
        def _format_next(c: mpq, d) -> str:
            m = _format_mon(d)
            if c == mpq(1):
                return f' + {m}' if m else f' + {c}'
            elif c == mpq(-1):
                return f' - {m}' if m else f' - {-c}'
            elif c > mpq(0):
                return f' + {c}{MUL}{m}' if m else f' + {c}'
            elif c < mpq(0):
                return f' - {-c}{MUL}{m}' if m else f' - {-c}'
            else:
                assert False, f'zero summand in {self!r}'

        def _format_var(v: Variable) -> str:
            SR = self._parent._singular_ring
            return p_String(v._poly, SR, SR).decode()

        def _format_mon(d: dict[Variable, int]) -> str:
            ret = ''
            if self._parent is not None:
                for v in self._parent.get_vars():
                    e = d.get(v, 0)
                    if e == 0:
                        continue
                    elif e == 1:
                        ret += f'{MUL}{_format_var(v)}'
                    else:
                        ret += f'{MUL}{_format_var(v)}{POW}{e}'
            return ret.lstrip(MUL)
        
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

    def __sub__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self - Term(other)
        _other = cython.cast(Term, other)
        if current_ring is None:
            assert self._parent is None and _other._parent is None
            return Term(self._mpq - _other._mpq)
        self._coerce(current_ring)
        _other._coerce(current_ring)
        SR = current_ring._singular_ring
        p1 = p_Copy(self._poly, SR)
        p2 = p_Copy(_other._poly, SR)
        difference = p_Sub(p1, p2, SR)
        return term(difference, current_ring)

    def as_variable(self) -> Variable:
        if not self.is_variable():
            raise ValueError(f'{self} is not a variable')
        return variable(self._poly, self._parent)

    def _coerce(self, R: Ring) -> None:
        """Mutable on self.
        """
        if self._parent is R:
            return

        if self._parent is None:
            n = R.mpq_to_number(self._mpq)
            self._poly = p_NSet(n, R._singular_ring)
            self._parent = R
            return
        
        # discuss
        # The following two lines once caused infinite recursions
        # __str()__ -> __eq__() -> _coerce():
        #
        # self_names = [str(g) for g in self._parent.get_vars()]
        # names = [str(g) for g in R.get_vars()]

        ind_map: list = []
        names = list(R.get_names())
        for name in self._parent.get_names():
            if name in names:
                ind_map.append(names.index(name) + 1)
            else:                                            
                raise ValueError(f'cannot coerce from {self._parent!r} to {R!r}')

        self_SR: cython.pointer[ring] = self._parent._singular_ring
        SR: cython.pointer[ring] = R._singular_ring
        assert SR != self_SR  # why? discuss
        assert self_SR.N <= SR.N

        ret = p_ISet(0, SR)
        p = self._poly
        while p:
            c = p_GetCoeff(p, self_SR)
            if not self_SR.cf.cfIsZero(c, self_SR.cf):
                mon = p_Init(SR)
                p_SetCoeff(mon, c, SR)
                for j in range(1, self_SR.N + 1):
                    e: cython.int = p_GetExp(p, j, self_SR)
                    if e:
                        p_SetExp(mon, ind_map[j-1], e, SR)
                p_Setm(mon, SR)
                ret = p_Add_q(ret, mon, SR)
            p = pNext(p)
        self._poly = ret
        self._parent = R

    def copy(self) -> Term:
        ret = Term(0)
        ret._parent = self._parent
        ret._mpq = self._mpq
        ret._poly = self._poly
        return ret
    
    def _dump(self):
        """Dump type and attributes of self, for debugging.
        """
        print(f'class {self.__class__.__name__}')
        if self._parent is None:
            print(f'    _parent = None')
        else:
            print(f'    _parent: Ring = {self._parent!r}')
        print(f'    _mpq: mpq = {self._mpq!r}')
        SR = self._parent._singular_ring
        a = cython.cast(cython.ulong, self._poly)
        poly = p_String(self._poly, SR, SR).decode()
        print(f'    _poly: cython.pointer[ring] = {a} ({poly})')

    def is_monomial(self) -> cython.bint:
        """Return :obj:`True` if this term is a monomial.
        """
        if self._parent is None:
            return self._mpq == 1
        SR = self._parent._singular_ring
        p = self._poly
        if p == cython.NULL:
            return False
        if pNext(p) != cython.NULL:
            return False
        if not SR.cf.cfIsOne(p_GetCoeff(p, SR), SR.cf):
            return False
        return True

    def is_variable(self) -> cython.bint:
        """Return :obj:`True` if this term is a variable.
        """
        if self._parent is None:
            return False
        if not self.is_monomial():
            return False
        if p_Deg(self._poly, self._parent._singular_ring) != 1:
            return False
        return True

    def lc(self) -> mpq:
        if self._parent is None:
            return self._mpq
        if self._poly == cython.NULL:
            return mpq(0)
        c = p_GetCoeff(self._poly, self._parent._singular_ring)
        ret = self._parent.number_to_mpq(c)
        return ret

    def summands(self) -> Iterator[tuple[dict[Variable, int], mpq]]:
        """Iterate over the summands of self yielding pairs of monomials and
        coefficients.
        """
        if self._parent is None:
            if self._mpq != 0:
                yield ({}, self._mpq)
        else:
            SR: Final[cython.pointer[ring]] = self._parent._singular_ring
            p: cython.pointer[poly] = self._poly
            while p:
                d: cython.dict = dict()
                for v in range(1, SR.N + 1):
                    n = p_GetExp(p, v, SR)
                    if n != 0:
                        d[self._parent.get_var_by_index(v - 1)] = n
                c = self._parent.number_to_mpq(p_GetCoeff(p, SR))
                yield d, c
                p = pNext(p)


@cython.cfunc
def term(poly: cython.pointer[poly], R: Ring) -> Term:
    t = Term(0)
    t._parent = R
    t._poly = poly
    return t


@cython.cclass
class Variable(Term):
    pass


@cython.cfunc
def variable(poly: cython.pointer[poly], R: Ring) -> Variable:
    # print(p_String(poly, R._singular_ring, R._singular_ring).decode())
    # print(R)
    t = Variable(0)
    t._parent = R
    t._poly = poly
    return t


def main():
    R = Ring(['x', 'y'])
    R.rPrint()
    x = R.get_var_by_index(0)
    y = R.get_var_by_index(1)
