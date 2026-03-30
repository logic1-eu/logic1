from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
import functools
from typing import ClassVar, Final, Generic, Never, Self, TypeVar

from gmpy2 import mpq

from logic1 import firstorder
from logic1.theories.Complex import ast
from logic1.theories.Complex.normalize import ComplexNormalizer
from logic1.theories.Complex.types import Number, RationalNumber

α = TypeVar('α')
τ = TypeVar('τ', bound='Term')


@dataclass
class VariableSet(firstorder.atomic.VariableSet['Variable']):
    
    _names: set[str] = field(default_factory=set)
     
    @property
    def stack(self) -> list[set[str]]:
        return [self._names]

    def __getitem__(self, index: str) -> Variable:
        """Implements the abstract method
        :meth:`.firstorder.atomic.VariableSet.__getitem__`.
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


VV: Final = VariableSet()


@dataclass
@functools.total_ordering
class SortKey(Generic[τ]):
    """A sort key for terms. Implements the abstract class
    :class:`.firstorder.atomic.Term.SortKey`.
    """

    term: τ
    """The term for which this is a sort key.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SortKey):
            return False
        return self.term._ast.sort_key() == other.term._ast.sort_key()

    def __hash__(self) -> int:
        return hash(self.term)

    def __le__(self, other: SortKey) -> bool:
        return self.term._ast.sort_key() <= other.term._ast.sort_key()
        

class Term(firstorder.Term['Term', 'Variable', Number, SortKey]):

    _ast: ast.AST
    _normalizer: ClassVar[ast.ASTVisitor[ast.AST]] = ComplexNormalizer()

    def __init__(self, number: Number) -> None:
        """Initialize a term from a number.
        
        >>> Term(2)
        2
        >>> Term(1.5)
        3/2
        >>> Term(1 + 2j)
        1 + 2 * I
        """
        self._ast = ast.AST.from_number(number).accept(self._normalizer)

    
    def __add__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            return Term._from_ast(self._ast + other._ast)
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
        return hash(self._ast)

    def __invert__(self) -> Term:
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
            return Term._from_ast(self._ast * other._ast)
        return self * Term(other)
        
    def __ne__(self, other: Number | Term) -> Ne:  # type: ignore[override]
        if isinstance(other, Term):
            return Ne(self, other)
        return self != Term(other)

    def __neg__(self) -> Term:
        return Term._from_ast(-self._ast)

    def __pow__(self, other: int) -> Term:
        return Term._from_ast(self._ast ** other)

    def __radd__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term(other) + self
    
    def __repr__(self) -> str:
        """String representation of this term that can be evaluated to
        reconstruct the term. For a more human-readable string
        representation, use :meth:`.Term.__str__` or :meth:`.Term.as_latex`.
        """
        return repr(self._ast)

    def __rmul__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term(other) * self

    def __rsub__(self, other: Number | Term) -> Term:
        assert not isinstance(other, Term)
        return Term(other) - self
    
    def __str__(self) -> str:
        return str(self._ast)

    def __sub__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            return Term._from_ast(self._ast - other._ast)
        return self - Term(other)
        
    def __truediv__(self, other: Number | Term) -> Term:
        if isinstance(other, Term):
            return Term._from_ast(self._ast / other._ast)
        return self / Term(other)

    def __xor__(self, other: Never) -> Term:
        raise NotImplementedError(
            "Use ** for exponentiation, not '^', which means xor "
            "in Python, and has the wrong precedence")
    
    def as_latex(self) -> str:
        """LaTeX representation as a string. Implements the abstract method
        :meth:`.firstorder.atomic.Term.as_latex`.
        """
        return self._ast.as_latex()

    def as_variable(self) -> Variable:
        """Return this term as a variable if it is one, and raise a ValueError
        otherwise.

        >>> x = VV['x']
        >>> x.as_variable()
        x
        >>> (x + 1).as_variable()
        Traceback (most recent call last):
        ...
        ValueError: Term x + 1 is not a variable
        """
        if isinstance(self._ast, ast.Var):
            return VV[self._ast.name]
        raise ValueError(f'Term {self} is not a variable')
    
    def conjugate(self) -> Term:
        """The complex conjugate of this term.

        >>> x = VV['x']
        >>> (x + 2).conjugate()
        ~x + 2
        >>> (2 * I).conjugate()
        -2 * I
        """
        return Term._from_ast(ast.Conj(self._ast))
    
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
        return self._ast.eval()
            
    @classmethod
    def _from_ast(cls, ast: ast.AST) -> Self:
        """Construct a term from an AST.
        """
        term = cls.__new__(cls)
        term._ast = ast.accept(cls._normalizer)
        return term

    @staticmethod
    def from_real_imag(real: RationalNumber, imag: RationalNumber) -> Term:
        """Convert a pair of real and imaginary parts to a term.

        >>> Term.from_real_imag(1, 2)
        1 + 2 * I
        """
        return Term(real) + Term(imag) * I
    
    def imaginary_part(self) -> Term:
        """The imaginary part of this term.

        >>> (2 * I).imaginary_part()
        2
        >>> x = VV['x']
        >>> (x + 2).imaginary_part()
        -1/2 * I * x + 1/2 * I * ~x
        """
        return Term._from_ast(ast.Im(self._ast))

    def is_constant(self) -> bool:
        """Return :obj:`True` if this term is constant.

        >>> x = VV['x']
        >>> (x + 2).is_constant()
        False
        >>> (2 * I).is_constant()
        True
        """
        return self._ast.is_constant()
        
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
        return isinstance(self._ast, ast.Var)

    def is_zero(self) -> bool:
        """Return :obj:`True` if this term is zero.

        >>> x = VV['x']
        >>> (x + 2).is_zero()
        False
        >>> (x - x).is_zero()
        True
        """
        return self._ast.is_zero()
    
    def lc(self) -> Term:
        """Return the leading coefficient of this term.
        """
        _, coeff = next(self.summands())
        return coeff
    
    def real_part(self) -> Term:
        """Return the real part of this term.

        >>> (2 * I).real_part()
        0
        >>> x = VV['x']
        >>> x.real_part()
        1/2 * x + 1/2 * ~x
        """
        return Term._from_ast(ast.Re(self._ast))

    def _repr_latex_(self) -> str:
        """LaTeX representation for Jupyter notebooks.
        """
        return self._ast._repr_latex_()
    
    def sort_key(self) -> SortKey[Self]:
        """A sort key suitable for ordering instances of this class. Implements
        the abstract method :meth:`.firstorder.atomic.Term.sort_key`.
        """
        return SortKey(self)

    def subs(self, sigma: Mapping[Variable, Number | Term]) -> Term:
        raise NotImplementedError()
    
    def summands(self) -> Iterator[tuple[Mapping[Term, int], Term]]:
        """An iterator that yields each summand of this term as a pair of a
        mapping from terms to their exponents, and a coefficient in 
        decreasing order of the leading term.
        """
        constant = Term(0)
        products = self._ast.args if isinstance(self._ast, ast.Add) else [self._ast]
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
        """An iterator that yields each variable of this term once. Implements
        the abstract method :meth:`.firstorder.atomic.Term.vars`.
        """
        result = set()
        for mapping, _ in self.summands():
            for term in mapping:
                _ast = term._ast
                if isinstance(_ast, ast.Var):
                    result.add(VV[_ast.name])
                elif isinstance(_ast, ast.Conj) and isinstance(_ast.arg, ast.Var):
                    result.add(VV[_ast.arg.name])
                elif isinstance(_ast, ast.Re) and isinstance(_ast.arg, ast.Var):
                    result.add(VV[_ast.arg.name])
                elif isinstance(_ast, ast.Im) and isinstance(_ast.arg, ast.Var):
                    result.add(VV[_ast.arg.name])
                else:
                    assert False, f'Unexpected term in summand: {term}'
        return iter(result)


class Variable(Term, firstorder.Variable['Variable', int, SortKey['Variable']]):

    @property    
    def name(self) -> str:
        """The name of this variable.
        """
        assert isinstance(self._ast, ast.Var)
        return self._ast.name

    def __init__(self) -> None:
        raise NotImplementedError("Use VV[...] to create variables")

    def fresh(self) -> Variable:
        """Returns a variable that has not been used so far. Implements
        abstract method :meth:`.firstorder.atomic.Variable.fresh`.
        """
        return VV.fresh(suffix=f'_{str(self)}')
    

I: Final[Term] = Term(1j)


def Re(term: Term) -> Term:
    """The real part of a term.

    >>> Re(2 * I)
    0
    >>> x = VV['x']
    >>> Re(x)
    1/2 * x + 1/2 * ~x
    """
    return term.real_part()


def Im(term: Term) -> Term:
    """The imaginary part of a term.

    >>> Im(2 * I)
    2
    >>> x = VV['x']
    >>> Im(x)
    -1/2 * I * x + 1/2 * I * ~x
    """
    return term.imaginary_part()


def Conj(term: Term) -> Term:
    """The complex conjugate of a term.

    >>> x = VV['x']
    >>> Conj(x + 2)
    ~x + 2
    >>> Conj(2 * I)
    -2 * I
    """
    return term.conjugate()


from logic1.theories.Complex.atomic import Eq, Ge, Gt, Le, Lt, Ne