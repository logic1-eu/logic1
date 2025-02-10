from __future__ import annotations

from typing import Generic, Iterable, Iterator, Optional, Self, TypeAlias, TypeVar

# Strangely, the following import is not made in
# sage/rings/polynomial/multi_polynomial_libsingular.pyx
from sage.rings.polynomial.multi_polynomial_ring_base import MPolynomialRing_base
from sage.rings.polynomial.polynomial_element import Polynomial_generic_dense
from sage.rings.ring import Ring
from sage.structure.factorization import Factorization


class MPolynomialRing_libsingular(MPolynomialRing_base):

    # The method resolution order is as follows:
    #
    # [sage.rings.polynomial.multi_polynomial_libsingular.MPolynomialRing_libsingular,
    #  sage.rings.polynomial.multi_polynomial_ring_base.MPolynomialRing_base,
    #  sage.rings.ring.CommutativeRing,
    #  sage.rings.ring.Ring,
    #  sage.structure.parent_gens.ParentWithGens,
    #  sage.structure.parent_base.ParentWithBase,
    #  sage.structure.parent_old.Parent,
    #  sage.structure.parent.Parent,
    #  sage.structure.category_object.CategoryObject,
    #  sage.structure.sage_object.SageObject,
    #  object]

    def gens(self) -> tuple[MPolynomial_libsingular]:
        # In reality, gens is inherited from
        # sage.structure.parent_gens.ParentWithGens. There is an attribute
        # element_class available, but I see no easy way to type this properly
        # at the place of its definition.
        ...


ρ = TypeVar('ρ')

_dict: TypeAlias = dict  # because dict will be shadowed by a class method


class MPolynomial_libsingular(Generic[ρ]):

    # The method resolution order is as follows:
    #
    # [sage.rings.polynomial.multi_polynomial_libsingular.MPolynomial_libsingular,
    #  sage.rings.polynomial.multi_polynomial.MPolynomial_libsingular,
    #  sage.rings.polynomial.multi_polynomial.MPolynomial,
    #  sage.rings.polynomial.commutative_polynomial.CommutativePolynomial,
    #  sage.structure.element.CommutativeAlgebraElement,
    #  sage.structure.element.CommutativeRingElement,
    #  sage.structure.element.RingElement,
    #  sage.structure.element.ModuleElement,
    #  sage.structure.element.Element,
    #  sage.structure.sage_object.SageObject,
    #  object]
    #
    # For the time being, we specify the whole interface down here.

    def __add__(self, other: object) -> Self:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __int__(self) -> int:
        ...

    def __iter__(self) -> Iterator[tuple[ρ, Self]]:
        ...

    def __le__(self, other: object) -> bool:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __mul__(self, other: object) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __pow__(self, other: object) -> Self:
        ...

    def __radd__(self, other: object) -> Self:
        ...

    def __rmul__(self, other: object) -> Self:
        ...

    def __rsub__(self, other: object) -> Self:
        ...

    def __sub__(self, other: object) -> Self:
        ...

    def __truediv__(self, other: object) -> Self:
        ...

    def change_ring(self, R: Ring) -> Self:
        # defined in sage.rings.polynomial.multi_polynomial.MPolynomial
        ...

    def coefficient(self, degrees: _dict[Self, int]) -> Self:
        ...

    def constant_coefficient(self) -> ρ:
        # Returns an element of the base ring of Self, which is always ZZ in
        # logic1.
        ...

    def content(self) -> ρ:
        ...

    def degree(self, x: Optional[Self] = None, std_grading: bool = False) -> int:
        ...

    def derivative(self, x: Self, n: int = 1) -> Self:
        ...

    def dict(self) -> _dict[tuple[int, ...], ρ]:
        ...

    def factor(self) -> Factorization[ρ]:
        ...

    def is_constant(self) -> bool:
        ...

    def is_generator(self) -> bool:
        ...

    def is_monomial(self) -> bool:
        ...

    def is_zero(self) -> bool:
        ...

    def lc(self) -> ρ:
        ...

    def monomial_coefficient(self, mon: Self) -> ρ:
        ...

    def monomials(self) -> list[Self]:
        ...

    def parent(self) -> MPolynomialRing_libsingular:
        ...

    def polynomial(self, var: Self) -> Polynomial_generic_dense:
        # defined in sage.rings.polynomial.multi_polynomial.MPolynomial
        ...

    def quo_rem(self, right: Self) -> tuple[Self, Self]:
        ...

    def reduce(self, i: Iterable[Self]) -> Self:
        ...

    # def subs(self, fixed: Optional[_dict[Self, ρ]] = None, **kw) -> Self:
    #     ...

    def subs(self, **kw) -> Self:
        ...

    def variables(self) -> tuple[Self]:
        ...
