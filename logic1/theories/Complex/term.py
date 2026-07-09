from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import functools
from typing import Callable, ClassVar, Final, Generic, Never, Self

from gmpy2 import mpq

from logic1 import firstorder

from logic1.theories.Complex.types import Number, RationalNumber, τ
from logic1.theories.Complex import ast
from logic1.theories.Complex.format import ReprFormatter, StrFormatter
from logic1.theories.Complex.normalize import cartesian_normal_form, conjugate_normal_form


@dataclass
class VariableSet(firstorder.term.VariableSet['Variable']):

    _names: set[str] = field(default_factory=set)

    @property
    def stack(self) -> list[set[str]]:
        return [self._names]

    def __getitem__(self, index: str) -> Variable:
        """Implements the abstract method
        :meth:`.firstorder.term.VariableSet.__getitem__`.
        """
        if not isinstance(index, str):
            raise ValueError(f'expecting string as index; {index} is {type(index)}')
        self._names.add(index)
        return Variable._from_ast(ast.Var(index))

    def __repr__(self) -> str:
        s = ', '.join(str(g) for g in (*self._names, '...'))
        return f'{{{s}}}'

    def fresh(self, suffix: str = '') -> Variable:
        """Return a fresh variable, by default from the sequence G0001, G0002,
        ..., G9999, G10000, ... This naming convention is inspired by Lisp's
        gensym(). If the optional argument :data:`suffix` is specified, the
        sequence G0001<suffix>, G0002<suffix>, ... is used instead.
        """
        i = 1
        v = f'G{i:04d}{suffix}'
        while v in self._names:
            i += 1
            v = f'G{i:04d}{suffix}'
        return self[v]

    def pop(self) -> None:
        raise NotImplementedError()

    def push(self) -> None:
        raise NotImplementedError()

    def reset(self) -> None:
        self._names = set()


VV: Final[VariableSet] = VariableSet()


@dataclass
@functools.total_ordering
class SortKey(Generic[τ]):
    """A sort key for terms. Implements the abstract class
    :class:`.firstorder.term.Term.SortKey`.
    """

    term: τ
    """The term for which this is a sort key.
    """

    def __eq__(self, other: object) -> bool:
        """Return :obj:`True` if the underlying terms are equivalent.
        """
        if not isinstance(other, SortKey):
            return False
        return self.term.normal_ast.sort_key() == other.term.normal_ast.sort_key()

    def __hash__(self) -> int:
        """Return the hash value of the underlying term.

        >>> z = VV['x']
        >>> hash(SortKey(z)) == hash(z)
        True
        """
        return hash(self.term)

    def __le__(self, other: SortKey) -> bool:
        """Comparison of terms based on :class:`.ast.SortKey`.
        """
        return self.term.normal_ast.sort_key() <= other.term.normal_ast.sort_key()


class Term(firstorder.Term['Term', 'Variable', Number, SortKey]):

    _normalizer: ClassVar[Callable[[ast.AST], ast.AST]] = conjugate_normal_form
    _ast: ast.AST
    _current_normal_form: Callable[[ast.AST], ast.AST]

    @property
    def normal_ast(self) -> ast.AST:
        """The normal form of this term as AST.
        """
        if self._current_normal_form != Term._normalizer:
            self._ast = Term._normalizer(self._ast)
            self._current_normal_form = Term._normalizer
        return self._ast

    def __init__(self, number: Number) -> None:
        """Initialize a term from a number.

        >>> Term(2)
        2
        >>> Term(1.5)
        3/2
        >>> Term(1 + 2j)
        1 + 2 * I
        """
        self._ast = Term._normalizer(ast.AST.from_number(number))
        self._current_normal_form = Term._normalizer

    def __add__(self, other: Number | Term) -> Term:
        """Add this term to another term or a number.

        >>> z = VV['z']
        >>> z + 2
        z + 2
        """
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast + other.normal_ast)
        return self + Term(other)

    def __eq__(self, other: Number | Term) -> Eq:  # type: ignore[override]
        if isinstance(other, Term):
            return Eq(self, other)
        return self == Term(other)

    def __ge__(self, other: Number | Term) -> Ge:
        if isinstance(other, Term):
            return Ge(self, other)
        return self >= Term(other)

    def __gt__(self, other: Number | Term) -> Gt:
        if isinstance(other, Term):
            return Gt(self, other)
        return self > Term(other)

    def __hash__(self) -> int:
        """Return the hash value of this term.
        """
        return hash(self.normal_ast)

    def __invert__(self) -> Term:
        """Return the complex conjugate of this term.

        >>> I.conjugate()
        -I
        """
        return self.conjugate()

    def __le__(self, other: Number | Term) -> Le:
        if isinstance(other, Term):
            return Le(self, other)
        return self <= Term(other)

    def __lt__(self, other: Number | Term) -> Lt:
        if isinstance(other, Term):
            return Lt(self, other)
        return self < Term(other)

    def __mul__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast * other.normal_ast)
        return self * Term(other)

    def __ne__(self, other: Number | Term) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        return self != Term(other)

    def __neg__(self) -> Term:
        return Term._from_ast(-self.normal_ast)

    def __pow__(self, other: int) -> Term:
        """Raise this term to a non-negative integer power.

        >>> I ** 2
        -1
        """
        return Term._from_ast(self.normal_ast ** other)

    def __radd__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term(other) + self

    def __repr__(self) -> str:
        """String representation of this term that can be evaluated to
        reconstruct the term. For a more human-readable string
        representation, use :meth:`.Term.__str__` or :meth:`.Term.as_latex`.
        """
        return self.normal_ast.accept(ReprFormatter())

    def __rmul__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term(other) * self

    def __rsub__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term(other) - self

    def __rtruediv__(self, other: Number | Term) -> Term:
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
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast - other.normal_ast)
        return self - Term(other)

    def __truediv__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            return Term._from_ast(self.normal_ast / other.normal_ast)
        return self / Term(other)

    def __xor__(self, other: Never) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")

    def as_latex(self) -> str:
        """Return a LaTeX representation as a string. Implements the abstract
        method :meth:`.firstorder.term.Term.as_latex`.
        """
        return self.normal_ast.as_latex()

    def as_variable(self) -> Variable:
        """Return this term as a variable. Raises a :class:`ValueError`
        if this term is not a variable.

        >>> x = VV['x']
        >>> x.as_variable()
        x
        >>> (x + 1).as_variable()
        Traceback (most recent call last):
        ...
        ValueError: Term x + 1 is not a variable
        """
        maybe_var = conjugate_normal_form(self.normal_ast)
        if isinstance(maybe_var, ast.Var):
            return VV[maybe_var.name]
        raise ValueError(f'Term {self} is not a variable')

    def conjugate(self) -> Term:
        """Return the complex conjugate of this term.

        >>> x = VV['x']
        >>> (x + 2).conjugate()
        ~x + 2
        >>> (2 * I).conjugate()
        -2 * I
        """
        return Term._from_ast(ast.Conj(self.normal_ast))

    def eval(self) -> tuple[mpq, mpq]:
        """Evaluate this term to a pair of rational numbers representing its
        real and imaginary parts. Raises a ValueError if this term is not
        constant.

        >>> (1 + 2 * I).eval()
        (mpq(1,1), mpq(2,1))
        >>> x = VV['x']
        >>> (x + 2).eval()
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate variable x
        """
        return self.normal_ast.eval()

    @classmethod
    def _from_ast(cls, ast: ast.AST) -> Self:
        """Construct a term from an AST.
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
        2
        >>> z = VV['z']
        >>> (z + 2).imaginary_part()
        -1/2 * I * z + 1/2 * I * ~z
        """
        return Term._from_ast(ast.Im(self.normal_ast))

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.

        >>> x = VV['x']
        >>> (x + 2).is_constant()
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

    def is_real(self) -> bool:
        """Return :obj:`True` if this term is real, i.e., its imaginary
        part is zero.

        >>> x = VV['x']
        >>> (x + 2).is_real()
        False
        >>> (x + x.conjugate()).is_real()
        True
        """
        return self.imaginary_part().is_zero()

    def is_variable(self) -> bool:
        """Return :obj:`True` if this term is a variable.

        >>> x = VV['x']
        >>> (x + 2).is_variable()
        False
        >>> x.is_variable()
        True
        >>> I.is_variable()
        False
        """
        return isinstance(conjugate_normal_form(self.normal_ast), ast.Var)

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term is zero.

        >>> x = VV['x']
        >>> (x + 2).is_zero()
        False
        >>> (x - x).is_zero()
        True
        """
        return self.normal_ast.is_zero()

    def lc(self) -> Term:
        """Return the leading coefficient of this term.

        >>> z = VV['z']
        >>> (3 * z - 2).lc()
        3
        >>> (-z * ~z).lc()
        -1
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
        0
        >>> z = VV['z']
        >>> z.real_part()
        1/2 * z + 1/2 * ~z
        """
        return Term._from_ast(ast.Re(self.normal_ast))

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks.
        """
        return self.normal_ast._repr_latex_()

    @classmethod
    def set_normal_form(cls, normalizer: Callable[[ast.AST], ast.AST]) -> Callable[[ast.AST], ast.AST]:
        """Return a function that normalizes an AST using the given normalizer.
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
        [({z: 2}, 1), ({z: 1}, 2), ({}, 1)]
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


class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

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
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.term.Variable.fresh`.

        >>> z = VV['z']
        >>> z.fresh()
        G0001_z
        """
        return VV.fresh(suffix=f'_{str(self)}')


I: Final[Term] = Term(1j)
"""The imaginary unit.

>>> I**2
-1
"""


def Re(term: Term) -> Term:
    """Return the real part of a term.

    >>> Re(2 * I)
    0
    >>> x = VV['x']
    >>> Re(x)
    1/2 * x + 1/2 * ~x
    """
    return term.real_part()


def Im(term: Term) -> Term:
    """Return the imaginary part of a term.

    >>> Im(2 * I)
    2
    >>> x = VV['x']
    >>> Im(x)
    -1/2 * I * x + 1/2 * I * ~x
    """
    return term.imaginary_part()


def Conj(term: Term) -> Term:
    """Return the complex conjugate of a term.

    >>> x = VV['x']
    >>> Conj(x + 2)
    ~x + 2
    >>> Conj(2 * I)
    -2 * I
    """
    return term.conjugate()


from logic1.theories.Complex.atomic import Eq, Ge, Gt, Le, Lt, Ne