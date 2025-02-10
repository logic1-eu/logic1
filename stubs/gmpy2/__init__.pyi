from __future__ import annotations

from typing import Literal, Self
from sage.all import Rational


class mpfr:

    def __gt__(self, other: object) -> bool:
        ...

    def __init__(self, *args) -> None:
        ...

    def __neg__(self) -> Self:
        ...


class mpq:

    numerator: mpz
    denominator: mpz

    def __add__(self, other: Self | mpfr) -> Self:
        ...

    def __ge__(self, other: object) -> bool:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __init__(self, arg: int | mpz | Rational) -> None:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __mul__(self, other: Self | mpfr) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __rtruediv__(self, other: int) -> Self:
        ...


class mpz:
    ...


def sign(arg: mpfr | mpq | mpz) -> Literal[-1, 0, 1]:
    ...
