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


p_number = cython.typedef(cython.pointer[number])
p_poly = cython.typedef(cython.pointer[poly])
pp_number = cython.typedef(cython.pointer[p_number])


@cython.cclass
class Ring:

    _singular_ring = cython.declare(cython.pointer[ring])

    def __init__(self, generators: Iterable[str]):
        generator_names = list(generators)
        generator_names.sort(key=Ring.sort_key)

        n: cython.int = len(generator_names)

        if n == 0:
            raise ValueError('Ring requires at least one generator')

        assert all(generator_names[i] != generator_names[i - 1] for i in range(1, n))
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

    def __or__(self, other: Ring) -> Ring:
        if self is other:
            return self
        if self._singular_ring.N < other._singular_ring.N:
            self, other = other, self
        names = set(self.get_names())
        is_subset = True
        for name in other.get_names():
            if name not in names:
                names.add(name)
                is_subset = False
        if is_subset:
            return self
        return Ring(names)

    def __repr__(self) -> str:
        names = ', '.join(repr(name) for name in self.get_names())
        return f'Ring([{names}])'

    def __str__(self) -> str:
        """The best Singular has to offer
        """
        return rString(self._singular_ring).decode()

    def get_names(self) -> Iterator[str]:
        SR = self._singular_ring
        for name in SR.names[:SR.N]:
            yield name.decode()
    
    def get_var_by_index(self, index: int) -> Variable:
        assert 0 <= index < self._singular_ring.N, f'Invalid index {index}'
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
    def number_to_mpq(self, nn: pp_number) -> mpq:
        """Create an mpq from a singular number.
        """
        # Immediate integers handles carry the tag 'SR_INT', i.e. the last bit is 1.
        # This distinguishes immediate integers from other handles which point to
        # structures aligned on 4 byte boundaries and therefore have last bit zero.
        # (The second bit is reserved as tag to allow extensions of this scheme.)
        # Using immediates as pointers and dereferencing them gives address errors.

        # n = n_Copy(n, self._singular_ring.cf)
        n = nn[0]

        ret = GMPy_MPQ_New(cython.NULL)
        tmp = GMPy_MPZ_New(cython.NULL)

        # nlGetNumerator is a C++ function. n is passed by reference and
        # modified.
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

        nn[0] = n  # C-analogue of call-by-reference in C++

        return ret
    
    def rPrint(self):
        rPrint(self._singular_ring)
        print()

    @staticmethod
    def sort_key(s: str) -> tuple[str, int]:
        base = s.rstrip('0123456789')
        index = s[len(base):]
        n = int(index) if index else -1
        return base, n
    

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

    _stack: list[set[str]]

    # required by the abstract parent class
    @property
    def stack(self) -> list[set[str]]:
        return self._stack
    
    @property
    def _used(self) -> set[str]:
        return self._stack[-1]
    
    @_used.setter
    def _used(self, names: set[str]) -> None:
        self._stack[-1] = names
    
    def __getitem__(self, name: str) -> Variable:
        """Implements abstract method
        :meth:`.firstorder.atomic.VariableSet.__getitem__`.
        """
        if not isinstance(name, str):
            raise ValueError(f'expecting string as index; {name} is {type(name)}')
        R = Ring([name])
        self._used.add(name)
        return R.get_var_by_index(0)
            
    def __init__(self) -> None:
        self._stack = [set()]

    def __repr__(self) -> str:
        names = sorted(self._used, key=Ring.sort_key)
        s = ', '.join(name for name in (*names, '...'))
        return f'{{{s}}}'

    def _drop(self) -> None:
        if len(self._stack) <= 1:
            raise ValueError('ignoring _drop at bottom of stack')
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
            raise ValueError('ignoring merge at bottom of stack')
        self.stack[-2].update(self._stack[-1])
        self.stack.pop()

    def pop(self) -> None:
        raise NotImplementedError()

    def push(self) -> None:
        raise NotImplementedError()

    def stash(self) -> None:
        self._stack.append(set())


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
        parent = self._coerce_to_common_parent(_other)
        if parent is None:
            return Term(self._mpq + _other._mpq)
        SR = cython.cast(Ring, parent)._singular_ring
        p1 = p_Copy(self._poly, SR)
        p2 = p_Copy(_other._poly, SR)
        sum = p_Add_q(p1, p2, SR)
        return term(sum, parent)
    
    def __eq__(self, other: Term) -> cython.bint:
        parent = self._coerce_to_common_parent(other)
        if parent is None:
            return self._mpq == other._mpq
        SR = cython.cast(Ring, parent)._singular_ring
        return p_EqualPolys(self._poly, other._poly, SR) == 1
    
    assert hash(mpq(0)) == 0  # ensure that mpq(0) hashes equally to cython.NULL in __hash__

    def __hash__(self) -> int:
        if self._parent is None:
            return hash(self._mpq)

        SR: Final = self._parent._singular_ring
        h_names: Final = [hash(name) for name in SR.names[:SR.N]]
        ret: cython.long = 0
        monomial: Term
        for coefficient, monomial in self:
            ret_mon: cython.long = hash(coefficient)
            for v in range(1, SR.N + 1):
                n = p_GetExp(monomial._poly, v, SR)
                if n != 0:
                    ret_mon = (1000003 * ret_mon) ^ h_names[v - 1]
                    ret_mon = (1000003 * ret_mon) ^ n
            ret += ret_mon
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
    
    def __iter__(self) -> Iterator[tuple[mpq, Term]]:
        """Iterate over the polynomial representation of the term, yielding
        pairs of coefficients and power products.

        >>> from gmpy2 import mpq
        >>> x, y = VV.get('x', 'y')
        >>> t = (x - y + 2) ** 2
        >>> [(abs(coef), power_product) for coef, power_product in t]
        [(mpq(1,1), x^2), (mpq(2,1), x*y), (mpq(1,1), y^2), (mpq(4,1), x),
         (mpq(4,1), y), (mpq(4,1), 1)]

        .. seealso::
            `The Sage implementation on GitHub
            <https://github.com/sagemath/sage/blob/develop/src/sage/rings/polynomial/multi_polynomial_libsingular.pyx>`
        """
        R = self._parent
        if R is None:
            if self._mpq != 0:
                yield self._mpq, Term(1)
        else:
            SR = R._singular_ring
            p = p_Copy(self._poly, SR)
            while p:
                next = pNext(p)
                p.next = cython.NULL
                t = term(p, R)
                coefficient = t.lc()
                p_SetCoeff(t._poly, n_Init(1, SR.cf), SR)
                p_Setm(t._poly, SR)  # necessary according to comment in decl
                yield coefficient, t
                p = next
            
    def __mul__(self, other: object) -> Term:
        if not isinstance(other, Term):
            return self * Term(other)
        _other = cython.cast(Term, other)
        parent = self._coerce_to_common_parent(_other)
        if parent is None:
            return Term(self._mpq * _other._mpq)
        SR = cython.cast(Ring, parent)._singular_ring
        e1: cython.ulong = p_GetMaxExp(self._poly, SR)
        e2: cython.ulong = p_GetMaxExp(_other._poly, SR)
        e: cython.ulong = e1 + e2
        if unlikely(e > SR.bitmask):
            raise OverflowError(f'exponent overflow {e}')
        prod = pp_Mult_qq(self._poly, _other._poly, SR)
        return term(prod, parent)
    
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
        return self + (-other)
    
    def __truediv__(self, other: object) -> Term:
        """True division. `other` must be a non-zero constant.
        """
        # compare sage.src.sage.libs.singular.polynomial.singular_polynomial_div_coeff
        if not isinstance(other, Term):
            return self / Term(other)
        _other = cython.cast(Term, other)
        if _other.is_zero():
            raise ZeroDivisionError()
        if not _other.is_constant():
            raise ValueError(f'non-constant divisor {_other}')
        return (1 / other.as_constant()) * self

    def __xor__(self, other: object) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_constant(self) -> mpq:
        if not self.is_constant():
            raise ValueError(f'{self} is not constant')
        return self.constant_coefficient()

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
        
        ind_map: list = []
        names = list(R.get_names())
        for name in self._parent.get_names():
            if name in names:
                ind_map.append(names.index(name) + 1)
            else:                                            
                raise ValueError(f'cannot coerce from {self._parent!r} to {R!r}')

        self_SR: cython.pointer[ring] = self._parent._singular_ring
        SR: cython.pointer[ring] = R._singular_ring
        assert SR != self_SR
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

    def _coerce_to_common_parent(self, other: Term) -> Optional[Ring]:
        if self._parent is None and other._parent is None:
            return None
        elif self._parent is None:
            self._coerce(other._parent)
            return other._parent
        elif other._parent is None:
            other._coerce(self._parent)
            return self._parent
        else:
            R = self._parent | other._parent
            self._coerce(R)
            other._coerce(R)
            return R

    def constant_coefficient(self) -> mpq:
        """Return the constant coefficient of this Term.
        """
        # Compare sage.rings.polynomial.multi_polynomial_libsingular.constant_coefficient
        if self._parent is None:
            return self._mpq
        R = self._parent
        SR = R._singular_ring
        p = self._poly
        if p == cython.NULL:
            return mpq(0)
        while p.next:
            p = pNext(p)
        if p_LmIsConstant(p, SR):
            return term(p, R).lc()
        else:
            return mpq(0)
            
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

    def is_constant(self) -> cython.bint:
        """Return :obj:`True` if this term is constant.
        """
        if self._parent is None:
            return True
        else:
            return p_IsConstant(self._poly, self._parent._singular_ring) == 1

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

    def is_zero(self) -> cython.bint:
        """Return :obj:`True` if this term is a zero.
        """
        if self._parent is None:
            return self._mpq == 0
        else:
            return self._poly == cython.NULL

    def lc(self) -> mpq:
        R = self._parent
        if R is None:
            return self._mpq
        if self._poly == cython.NULL:
            return mpq(0)
        # We use direct access to p.coef, in order to avoid
        # return-by-reference of p_GetCoeff().
        return R.number_to_mpq(cython.address(self._poly.coef))

    def summands(self) -> Iterator[tuple[dict[Variable, int], mpq]]:
        """Iterate over the summands of self yielding pairs of monomials and
        coefficients.
        """
        power_product: Term
        for coefficient, power_product in self:
            if power_product._parent is None:
                assert power_product._mpq == 1
                yield {}, coefficient
            else:
                SR = power_product._parent._singular_ring
                p = power_product._poly
                d: cython.dict = dict()
                for i in range(1, SR.N + 1):
                    n = p_GetExp(p, i, SR)
                    if n != 0:
                        d[self._parent.get_var_by_index(i - 1)] = n
                yield d, coefficient


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
