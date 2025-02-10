from sage.rings.polynomial.multi_polynomial_libsingular import MPolynomial_libsingular
from typing import Generic, Iterator, TypeVar


ρ = TypeVar('ρ')


class Factorization(Generic[ρ]):

    def __iter__(self) -> Iterator[tuple[MPolynomial_libsingular[ρ], int]]:
        ...

    def unit(self) -> MPolynomial_libsingular[ρ]:
        ...
