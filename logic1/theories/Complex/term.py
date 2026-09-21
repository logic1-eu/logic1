"""Terms and variables in the theory :mod:`Complex <logic1.theories.Complex>`.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import functools
import re
from typing import Callable, ClassVar, Final, Generic, Never, Optional, Self

from gmpy2 import mpq

from logic1 import firstorder

from logic1.theories.Complex.types import Number, RationalNumber, τ
from logic1.theories.Complex import ast
from logic1.theories.Complex.format import StrFormatter, TermReprFormatter
from logic1.theories.Complex.normalize import cartesian_normal_form, conjugate_normal_form



@dataclass
class VariableSet(firstorder.term.VariableSet['Variable']):
    """Infinite set of variables used in the theory of complex numbers.
    Implements the abstract class :class:`.firstorder.term.VariableSet`.
    :class:`Variables <.Variable>` are obtained from the global instance
    :data:`VV`.

    >>> VV['z']
    z
    >>> VV.get('a', 'b')
    (a, b)
    """

    _names: set[str] = field(default_factory=set)
    """The set of currently used variable names.
    """

    _IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")
    """Regular expression for valid variable names.
    """

    @property
    def stack(self) -> list[set[str]]:
        """Return the current stack of variable names. Implements the abstract
        property :attr:`.firstorder.term.VariableSet.stack`.

        >>> VV.reset()
        >>> z = VV['z']
        >>> VV.stack
        [{'z'}]
        """
        return [self._names]

    def __getitem__(self, index: str) -> Variable:
        """Return the variable with the given name. Raise a :class:`ValueError`
        if the name is not a valid Python identifier. Implements the abstract
        method :meth:`.firstorder.term.VariableSet.__getitem__`.

        >>> VV['z']
        z
        >>> VV['1z']
        Traceback (most recent call last):
        ...
        ValueError: variable name '1z' is not a valid identifier
        """
        if not isinstance(index, str):
            raise ValueError(f'expecting string as index; {index} is {type(index)}')
        if not self._IDENTIFIER_RE.fullmatch(index):
            raise ValueError(f'variable name \'{index}\' is not a valid identifier')
        self._names.add(index)
        return Variable._from_ast(ast.Var(index))

    def __repr__(self) -> str:
        """Return a string representation of this variable set.

        >>> VV.reset()
        >>> VV.get('x', 'y', 'z')
        (x, y, z)
        >>> VV
        {x, y, z, ...}
        """
        s = ', '.join(str(g) for g in (*sorted(self._names), '...'))
        return f'{{{s}}}'

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :code:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.

        >>> VV.fresh()
        G0001
        """
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in self._names:
            i += 1
            v = f'G{i:04d}{suffix}'
        return self[v]

    def pop(self) -> None:
        """Raise a :class:`NotImplementedError`. Implements the abstract method
        :meth:`.firstorder.term.VariableSet.pop`.
        """
        raise NotImplementedError()

    def push(self) -> None:
        """Raise a :class:`NotImplementedError`. Implements the abstract method
        :meth:`.firstorder.term.VariableSet.push`.
        """
        raise NotImplementedError()

    def reset(self) -> None:
        """Clear all used variable names.

        >>> VV.reset()
        >>> z = VV['z']
        >>> VV
        {z, ...}
        >>> VV.reset()
        >>> VV
        {...}
        """
        self._names = set()


VV: Final[VariableSet] = VariableSet()
"""The global :class:`VariableSet` used in the theory of complex numbers.

>>> VV['z']
z
>>> VV.get('a', 'b')
(a, b)
"""


@dataclass
@functools.total_ordering
class SortKey(Generic[τ]):
    """A sort key for terms.

    >>> z = VV['z']
    >>> SortKey(z) < SortKey(z + 1)
    True

    .. seealso:: :meth:`.Term.sort_key`
    """

    term: τ
    """The :class:`.Term` for which this is a sort key.
    """

    def __eq__(self, other: object) -> bool:
        """Return :obj:`True` if the underlying terms are equivalent.

        >>> z = VV['z']
        >>> SortKey(z) == SortKey(z)
        True
        >>> SortKey(z) == SortKey(z + 1)
        False
        """
        if not isinstance(other, SortKey):
            return False
        return self.term.normal_ast.sort_key() == other.term.normal_ast.sort_key()

    def __hash__(self) -> int:
        """Return the hash value of the underlying term.

        >>> z = VV['z']
        >>> hash(SortKey(z)) == hash(z)
        True
        """
        return hash(self.term)

    def __le__(self, other: SortKey) -> bool:
        """Comparison of terms based on :class:`.ast.SortKey`. The remaining
        comparison operators are derived from this using
        :func:`functools.total_ordering`.

        >>> z = VV['z']
        >>> SortKey(z) <= SortKey(z)
        True
        >>> SortKey(z) <= SortKey(z + 1)
        True
        """
        return self.term.normal_ast.sort_key() <= other.term.normal_ast.sort_key()


class Term(firstorder.Term['Term', 'Variable', Number, SortKey]):
    """Term in the theory of complex numbers. Implements the abstract class
    :class:`.firstorder.term.Term`. It is represented internally as
    :class:`AST <.ast.AST>` in normal form. The default normal form is
    :func:`.conjugate_normal_form`, but it can be changed globally using
    :meth:`.set_normal_form`.

    >>> z = VV['z']
    >>> (z + I) ** 2
    z**2 + 2 * I * z - 1
    >>> Re(z)
    mpq(1,2) * z + mpq(1,2) * ~z

    .. seealso::
        :class:`Variable`, :data:`VV`, :func:`Re`, :func:`Im`, :func:`Conj`,
        :class:`AtomicFormula <.Complex.atomic.AtomicFormula>`
    """

    _normalizer: ClassVar[Callable[[ast.AST], ast.AST]] = conjugate_normal_form
    """The global normal form for terms. The default normal form is
    :func:`.conjugate_normal_form`, but it can be changed globally using
    :meth:`.set_normal_form`.
    """

    _ast: ast.AST
    """The AST representation of this term in normal form
    :attr:`_current_normal_form`.
    """

    _current_normal_form: Callable[[ast.AST], ast.AST]
    """The current normal form used for :attr:`_ast`.
    """

    _hash: Optional[int] = None
    """The cached hash value of this term with respect to conjugate normal form.
    """

    @property
    def normal_ast(self) -> ast.AST:
        """The AST representation of this term in the global normal form.
        """
        if self._current_normal_form != Term._normalizer:
            self._ast = Term._normalizer(self._ast)
            self._current_normal_form = Term._normalizer
        return self._ast

    def __init__(self, number: Number) -> None:
        """Initialize a term from a number.

        >>> Term(2)
        Term(2)
        >>> Term(1.5)
        Term(mpq(3,2))
        >>> Term(1 + 2j)
        1 + 2 * I
        """
        self._ast = Term._normalizer(ast.AST.from_number(number))
        self._current_normal_form = Term._normalizer

    def __add__(self, other: Number | Term) -> Term:
        """Add another term or a number to this term.

        >>> z = VV['z']
        >>> z + 2
        z + 2
        """
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast + other.normal_ast)
        return self + Term(other)

    def __eq__(self, other: Number | Term) -> Eq:  # type: ignore[override]
        """Construct an equality between this term and another term or a number.

        >>> z = VV['z']
        >>> z == 2
        z == 2
        """
        if isinstance(other, Term):
            return Eq(self, other)
        return self == Term(other)

    def __ge__(self, other: Number | Term) -> Ge:
        """Construct a non-strict inequality between this term and another term
        or a number. Raise a :class:`ValueError` if either side of the
        inequality is not real.

        >>> z = VV['z']
        >>> z * ~z >= 0
        z * ~z >= 0
        >>> z >= 0
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z >= 0 because it is not real
        """
        if isinstance(other, Term):
            return Ge(self, other)
        return self >= Term(other)

    def __gt__(self, other: Number | Term) -> Gt:
        """Construct a strict inequality between this term and another term
        or a number. Raise a :class:`ValueError` if either side of the
        inequality is not real.

        >>> z = VV['z']
        >>> z * ~z > 0
        z * ~z > 0
        >>> z > 0
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z > 0 because it is not real
        """
        if isinstance(other, Term):
            return Gt(self, other)
        return self > Term(other)

    def __hash__(self) -> int:
        """Return the hash value of this term.
        """
        if self._hash is None:
            if self._current_normal_form == conjugate_normal_form:
                self._hash = hash(self._ast)
            else:
                self._hash = hash(conjugate_normal_form(self._ast))
        return self._hash

    def __invert__(self) -> Term:
        """Return the complex conjugate of this term.

        >>> ~I
        -I
        """
        return self.conjugate()

    def __le__(self, other: Number | Term) -> Le:
        """Construct a non-strict inequality between this term and another term
        or a number. Raise a :class:`ValueError` if either side of the
        inequality is not real.

        >>> z = VV['z']
        >>> z * ~z <= 0
        z * ~z <= 0
        >>> z <= 0
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z <= 0 because it is not real
        """
        if isinstance(other, Term):
            return Le(self, other)
        return self <= Term(other)

    def __lt__(self, other: Number | Term) -> Lt:
        """Construct a strict inequality between this term and another term
        or a number. Raise a :class:`ValueError` if either side of the
        inequality is not real.

        >>> z = VV['z']
        >>> z * ~z < 0
        z * ~z < 0
        >>> z < 0
        Traceback (most recent call last):
        ...
        ValueError: Cannot create atomic formula z < 0 because it is not real
        """
        if isinstance(other, Term):
            return Lt(self, other)
        return self < Term(other)

    def __mul__(self, other: Number | Term) -> Term:
        """Multiply this term by another term or a number.

        >>> z = VV['z']
        >>> z * 2
        2 * z
        """
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast * other.normal_ast)
        return self * Term(other)

    def __ne__(self, other: Number | Term) -> Ne:  # type: ignore[override]
        """Construct an inequality between this term and another term or a
        number.

        >>> z = VV['z']
        >>> z != 2
        z != 2
        """
        if isinstance(other, Term):
            return Ne(self, other)
        return self != Term(other)

    def __neg__(self) -> Term:
        """Return the negation of this term.

        >>> z = VV['z']
        >>> -z
        -z
        """
        return Term._from_ast(-self.normal_ast)

    def __pow__(self, other: int) -> Term:
        """Raise this term to a non-negative integer power. Raise
        a :class:`TypeError` if the exponent is negative.

        >>> I ** 2
        -Term(1)
        """
        return Term._from_ast(self.normal_ast ** other)

    def __radd__(self, other: Number | Term) -> Term:
        """Add this term to a number. All other cases are handled by
        :meth:`__add__`.

        >>> z = VV['z']
        >>> 2 + z
        z + 2
        """
        assert not isinstance(other, Term)
        return Term(other) + self

    def __repr__(self) -> str:
        """Return a string representation of this term that is valid Python
        code. Allows for the reconstruction of the original expression up to
        definition of variables and conversion of number types.

        >>> z = VV['z']
        >>> repr(z ** 2 + I)
        'z**2 + I'
        """
        return self.normal_ast.accept(TermReprFormatter())

    def __rmul__(self, other: Number | Term) -> Term:
        """Multiply a number by this term. All other cases are handled by
        :meth:`__mul__`.

        >>> z = VV['z']
        >>> 2 * z
        2 * z
        """
        assert not isinstance(other, Term)
        return Term(other) * self

    def __rsub__(self, other: Number | Term) -> Term:
        """Subtract this term from a number. All other cases are handled by
        :meth:`__sub__`.

        >>> z = VV['z']
        >>> 2 - z
        -z + 2
        """
        assert not isinstance(other, Term)
        return Term(other) - self

    def __rtruediv__(self, other: Number | Term) -> Term:
        """Divide a number by this term. Raise a :class:`ValueError` if
        this term is not constant. All other cases are handled by
        :meth:`__truediv__`.

        >>> 1 / I
        -I
        >>> z = VV['z']
        >>> 1 / z
        Traceback (most recent call last):
        ...
        ValueError: Cannot divide by a non-constant term
        """
        assert not isinstance(other, Term)
        return Term(other) / self

    def __str__(self) -> str:
        """Return a human-readable string representation of this term.

        >>> z = VV['z']
        >>> str(z ** 2 + I)
        'z^2 + i'
        """
        return self.normal_ast.accept(StrFormatter())

    def __sub__(self, other: Number | Term) -> Term:
        """Subtract another term or a number from this term.

        >>> z = VV['z']
        >>> z - 2
        z - 2
        """
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast - other.normal_ast)
        return self - Term(other)

    def __truediv__(self, other: Number | Term) -> Term:
        """Divide this term by another term or a number. Raise a
        :class:`ValueError` if the other term is not constant.

        >>> z = VV['z']
        >>> z / 2
        mpq(1,2) * z
        >>> z / z
        Traceback (most recent call last):
        ...
        ValueError: Cannot divide by a non-constant term
        """
        if isinstance(other, Term):
            try:
                return Term._from_ast(self.normal_ast / other.normal_ast)
            except ValueError:
                raise ValueError("Cannot divide by a non-constant term")
        return self / Term(other)

    def __xor__(self, other: Never) -> Term:
        """Raise a :class:`NotImplementedError`. The operator
        :meth:`** <.Complex.term.Term.__pow__>` is used for
        exponentiation instead.
        """
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_latex(self) -> str:
        """Return a LaTeX representation as a string. Implements the abstract
        method :meth:`.firstorder.term.Term.as_latex`.

        >>> z = VV['z']
        >>> (z + 2 * I).as_latex()
        'z + 2 i'
        """
        return self.normal_ast.as_latex()

    def as_variable(self) -> Variable:
        """Return this term as a variable. Raises a :class:`ValueError`
        if this term is not a variable.

        >>> z = VV['z']
        >>> z.as_variable()
        z
        >>> (z + 1).as_variable()
        Traceback (most recent call last):
        ...
        ValueError: Term z + 1 is not a variable
        """
        maybe_var = conjugate_normal_form(self.normal_ast)
        if isinstance(maybe_var, ast.Var):
            return VV[maybe_var.name]
        raise ValueError(f'Term {self} is not a variable')

    def conjugate(self) -> Term:
        """Return the complex conjugate of this term.'

        >>> z = VV['z']
        >>> (z + 2).conjugate()
        ~z + 2
        >>> (2 * I).conjugate()
        -2 * I

        .. seealso:: :func:`.Conj`, :meth:`~ <.Complex.term.Term.__invert__>`
        """
        return Term._from_ast(ast.Conj(self.normal_ast))

    def eval(self) -> tuple[mpq, mpq]:
        """Evaluate this term to a pair of its real and imaginary parts.
        Raise a :class:`ValueError` if this term is not constant.

        >>> (1 + 2 * I).eval()
        (mpq(1,1), mpq(2,1))
        >>> z = VV['z']
        >>> (z + 2).eval()
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate variable z
        """
        return self.normal_ast.eval()

    @classmethod
    def _from_ast(cls, ast: ast.AST) -> Self:
        """Construct a term from an AST. Note that AST variables are *not*
        registered in the global variable set :data:`VV`.

        >>> Term._from_ast(ast.Var('z') + ast.Rat(mpq(1, 2)))
        z + mpq(1,2)
        """
        term = cls.__new__(cls)
        term._ast = cls._normalizer(ast)
        term._current_normal_form = cls._normalizer
        return term

    @staticmethod
    def from_real_imag(real: RationalNumber, imag: RationalNumber) -> Term:
        """Convert a pair of real and imaginary parts to a term.

        >>> Term.from_real_imag(1, 2)
        1 + 2 * I
        """
        return Term(real) + Term(imag) * I

    def imaginary_part(self) -> Term:
        """Return the imaginary part of this term.

        >>> (2 * I).imaginary_part()
        Term(2)
        >>> z = VV['z']
        >>> (z + 2).imaginary_part()
        -mpq(1,2) * I * z + mpq(1,2) * I * ~z

        .. seealso:: :func:`.Im`
        """
        return Term._from_ast(ast.Im(self.normal_ast))

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.

        >>> z = VV['z']
        >>> (z + 2).is_constant()
        False
        >>> (2 * I).is_constant()
        True
        """
        return self.normal_ast.is_constant()

    def is_imaginary(self) -> bool:
        """Return :obj:`True` if this term is imaginary, i.e., its real
        part is zero.

        >>> x = VV['x']
        >>> (x + 2).is_imaginary()
        False
        >>> (2 * I).is_imaginary()
        True
        """
        return self.real_part().is_zero()

    def is_rational(self) -> bool:
        """Return :obj:`True` if this term is a rational number.

        >>> Term(1).is_rational()
        True
        >>> z = VV['z']
        >>> z.is_rational()
        False
        """
        return self.is_constant() and self.is_real()

    def is_real(self) -> bool:
        """Return :obj:`True` if this term is real, i.e., its imaginary
        part is zero.

        >>> z = VV['z']
        >>> (z + 2).is_real()
        False
        >>> (z + z.conjugate()).is_real()
        True
        """
        return self.imaginary_part().is_zero()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.

        >>> z = VV['z']
        >>> (z + 2).is_variable()
        False
        >>> z.is_variable()
        True
        >>> I.is_variable()
        False
        """
        return isinstance(conjugate_normal_form(self.normal_ast), ast.Var)

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term is zero.

        >>> z = VV['z']
        >>> (z + 2).is_zero()
        False
        >>> (z - z).is_zero()
        True
        """
        return self.normal_ast.is_zero()

    def lc(self) -> Term:
        """Return the leading coefficient of this term.

        >>> z = VV['z']
        >>> (3 * z - 2).lc()
        Term(3)
        >>> (-z * ~z).lc()
        -Term(1)
        """
        if self.is_constant():
            return self
        if isinstance(self.normal_ast, ast.Add):
            return Term._from_ast(self.normal_ast.args[0]).lc()
        if isinstance(self.normal_ast, ast.Neg):
            return -Term._from_ast(self.normal_ast.arg).lc()
        if isinstance(self.normal_ast, ast.Mul):
            result = Term(1)
            for arg in self.normal_ast.args:
                result = result * Term._from_ast(arg).lc()
            return result
        return Term(1)

    def real_part(self) -> Term:
        """Return the real part of this term.

        >>> (2 * I).real_part()
        Term(0)
        >>> z = VV['z']
        >>> z.real_part()
        mpq(1,2) * z + mpq(1,2) * ~z

        .. seealso:: :func:`.Re`
        """
        return Term._from_ast(ast.Re(self.normal_ast))

    def _repr_latex_(self) -> str:
        """Return a LaTeX representation for Jupyter notebooks.

        >>> z = VV['z']
        >>> (z + 2 * I)._repr_latex_()
        '$\\\\displaystyle z + 2 i$'
        """
        return self.normal_ast._repr_latex_()

    @classmethod
    def set_normal_form(cls, normalizer: Callable[[ast.AST], ast.AST]) -> Callable[[ast.AST], ast.AST]:
        """Set the global normal form for terms and return the previous
        used normal form. The default normal form
        is :func:`.conjugate_normal_form`.

        >>> z = VV['z']
        >>> z
        z
        >>> old = Term.set_normal_form(cartesian_normal_form)
        >>> z
        Re(z) + I * Im(z)
        >>> old == conjugate_normal_form
        True
        >>> _ = Term.set_normal_form(old)
        >>> z
        z

        .. seealso::
            :func:`.conjugate_normal_form`,
            :func:`.cartesian_normal_form`
        """
        ret = cls._normalizer
        cls._normalizer = normalizer
        return ret

    def sort_key(self) -> SortKey[Self]:
        """Return a sort key suitable for ordering terms. Implements the
        abstract method :meth:`.firstorder.term.Term.sort_key`.

        >>> z = VV['z']
        >>> z.sort_key() < (z + 1).sort_key()
        True
        """
        return SortKey(self)

    def subs(self, sigma: Mapping[Variable, Number | Term]) -> Term:
        """Return a term obtained by substituting the variables in this term
        according to the given mapping.

        >>> a, b = VV.get('a', 'b')
        >>> (a ** 2).subs({a: I})
        -Term(1)
        >>> (a + b).subs({a: 1, b: a})
        a + 1
        """
        ast_sigma: dict[ast.Var, ast.AST] = {}
        for var, value in sigma.items():
            ast_var = conjugate_normal_form(var.normal_ast)
            assert isinstance(ast_var, ast.Var)
            if isinstance(value, Term):
                ast_sigma[ast_var] = value.normal_ast
            else:
                ast_sigma[ast_var] = ast.AST.from_number(value)
        return Term._from_ast(self.normal_ast.subs(ast_sigma))

    def _summands(self) -> Iterator[tuple[Mapping[Term, int], Term]]:
        """Return an iterator that yields each summand of this term
        as a pair of a mapping from terms to their exponents, and a coefficient
        in decreasing order of the leading term.

        >>> z = VV['z']
        >>> list((z**2 + 2 * z + 1)._summands())
        [({z: 2}, Term(1)), ({z: 1}, Term(2)), ({}, Term(1))]
        """
        constant = Term(0)
        products = self.normal_ast.args if isinstance(self.normal_ast, ast.Add) else [self.normal_ast]
        for product in products:
            if product.is_constant():
                constant = constant + Term._from_ast(product)
                continue
            coeff = Term(1)
            if isinstance(product, ast.Neg):
                coeff = -coeff
                product = product.arg
            factors = product.args if isinstance(product, ast.Mul) else [product]
            mapping = {}
            for factor in factors:
                if factor.is_constant():
                    coeff = coeff * Term._from_ast(factor)
                    continue
                if isinstance(factor, ast.Neg):
                    coeff = -coeff
                    factor = factor.arg
                if isinstance(factor, ast.Pow):
                    mapping[Term._from_ast(factor.base)] = factor.exponent
                else:
                    mapping[Term._from_ast(factor)] = 1
            yield (mapping, coeff)
        yield ({}, constant)

    def vars(self) -> Iterator[Variable]:
        """Return an iterator that yields each variable of this term once.
        Implements the abstract method :meth:`.firstorder.term.Term.vars`.

        >>> a, b, c = VV.get('a', 'b', 'c')
        >>> vars = (a + b * c).vars()
        >>> list(sorted(vars, key=Term.sort_key))
        [a, b, c]
        """
        result = set()
        stack = [self.normal_ast]
        while stack:
            node = stack.pop()
            if isinstance(node, ast.Var):
                result.add(VV[node.name])
            else:
                stack.extend(arg for arg in node.args if isinstance(arg, ast.AST))
        yield from result


class Variable(Term, firstorder.Variable['Variable', Number, SortKey['Variable']]):
    """Variable in the theory of complex numbers. Implements the abstract class
    :class:`.firstorder.term.Variable`. Variables are obtained from the global
    variable set :data:`VV`.

    >>> VV['z']
    z
    >>> VV.get('a', 'b')
    (a, b)
    """

    @property
    def name(self) -> str:
        """The name of this variable.

        >>> z = VV['z']
        >>> z.name
        'z'
        """
        _ast = conjugate_normal_form(self.normal_ast)
        assert isinstance(_ast, ast.Var)
        return _ast.name

    def __init__(self) -> None:
        """This constructor is not meant to be called directly. Use :data:`VV`
        to create variables.
        """
        raise NotImplementedError("Use VV[...] to create variables")

    def fresh(self) -> Variable:
        """Return a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.term.Variable.fresh`.

        >>> z = VV['z']
        >>> z.fresh()
        G0001_z
        """
        return VV.fresh(suffix=f'_{str(self)}')


I: Final[Term] = Term(1j)
"""The imaginary unit.

>>> I**2
-Term(1)
"""


def Re(term: Number | Term) -> Term:
    """Return the real part of a term.

    >>> Re(2 * I)
    Term(0)
    >>> z = VV['z']
    >>> Re(z)
    mpq(1,2) * z + mpq(1,2) * ~z
    """
    if not isinstance(term, Term):
        term = Term(term)
    return term.real_part()


def Im(term: Number | Term) -> Term:
    """Return the imaginary part of a term.

    >>> Im(2 * I)
    Term(2)
    >>> z = VV['z']
    >>> Im(z)
    -mpq(1,2) * I * z + mpq(1,2) * I * ~z
    """
    if not isinstance(term, Term):
        term = Term(term)
    return term.imaginary_part()


def Conj(term: Number | Term) -> Term:
    """Return the complex conjugate of a term.

    >>> z = VV['z']
    >>> Conj(z + 2)
    ~z + 2
    >>> Conj(2 * I)
    -2 * I
    """
    if not isinstance(term, Term):
        term = Term(term)
    return term.conjugate()


from logic1.theories.Complex.atomic import Eq, Ge, Gt, Le, Lt, Ne
