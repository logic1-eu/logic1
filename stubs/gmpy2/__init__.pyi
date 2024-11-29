from typing import Self
from sage.all import Rational


class mpz:
    ...


class mpq:

    numerator: mpz
    denominator: mpz

    def __init__(self, arg: int | mpz | Rational) -> None:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __mul__(self, other: Self) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __rtruediv__(self, other: int) -> Self:
        ...
