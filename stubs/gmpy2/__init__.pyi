from __future__ import annotations


from typing import Literal, overload, Self, TypeVar
from sage.all import Rational


class InvalidOperationError(ValueError):
    ...


class context:
    ...

    def __enter__(self) -> None:
        ...

    def __exit__(self, type, value, traceback) -> None:
        ...

    def __init__(self, **args) -> None:
        ...


class mpfr:

    def __add__(self, other: mpfr | mpq) -> Self:
        ...

    def __ge__(self, other: object) -> bool:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __init__(self, *args) -> None:
        ...

    def __le__(self, other: object) -> bool:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __mul__(self, other: mpfr | mpq) -> Self:
        ...

    def __neg__(self) -> Self:
        ...

    def __pow__(self, n: int) -> Self:
        ...


μ = TypeVar('μ', bound='mpq | mpfr')


class mpq:

    numerator: mpz
    denominator: mpz

    def __add__(self, other: μ) -> μ:
        ...

    def __ge__(self, other: object) -> bool:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    @overload
    def __init__(self, arg: float | int | mpz | Rational) -> None:
        ...

    @overload
    def __init__(self, numer: int | mpz, denom: int | mpz) -> None:
        ...

    def __le__(self, other: object) -> bool:
        ...

    def __lt__(self, other: object) -> bool:
        ...

    def __mul__(self, other: μ) -> μ:
        ...

    def __neg__(self) -> Self:
        ...

    def __pow__(self, n: int) -> Self:
        ...

    def __sub__(self, other: Self) -> Self:
        ...

    def __truediv__(self, other: Self) -> Self:
        ...

    def __rtruediv__(self, other: int) -> Self:
        ...


class mpz:

    def __int__(self) -> int:
        ...
    
    def __gt__(self, other: object) -> bool:
        ...


def sign(arg: mpfr | mpq | mpz) -> Literal[-1, 0, 1]:
    ...
