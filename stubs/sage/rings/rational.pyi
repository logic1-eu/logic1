from fractions import Fraction
from typing import Any, Literal, Self

from gmpy2 import mpq
from sage.rings.integer import Integer


class Rational:
    # The method resolution order is as follows:
    #
    # [sage.rings.rational.Rational,
    #  sage.structure.element.FieldElement,
    #  sage.structure.element.CommutativeRingElement,
    #  sage.structure.element.RingElement,
    #  sage.structure.element.ModuleElement,
    #  sage.structure.element.Element,
    #  sage.structure.sage_object.SageObject,
    #  object]
    #
    # For the time being, we specify the whole interface down here.

    # The following do not exist in sage. See line 1070 in
    # sage/structure/element.pyx, which says:
    ####################################################################
    # In a Cython or a Python class, you must define _richcmp_
    #
    # Rich comparisons (like a < b) will use _richcmp_
    #
    # In the _richcmp_ method, you can assume that both arguments have
    # identical parents.
    ####################################################################
    def __ge__(self, other: object) -> bool:
        ...

    def __gt__(self, other: object) -> bool:
        ...

    def __lt__(self, other: object) -> bool:
        ...
    ####################################################################

    def __add__(self, other: object) -> Any:
        # Don't quite know how to deal with the infinities.
        ...

    def __init__(self, arg: int | mpq | Fraction | tuple[int | Integer, int | Integer]) -> None:
        ...

    def __invert__(self) -> Self:
        ...

    def __mul__(self, other: object) -> Any:
        ...

    def __neg__(self) -> Self:
        ...

    def __radd__(self, other: object) -> Any:
        ...

    def __rmul__(self, other: object) -> Any:
        ...

    def __truediv__(self, other: object) -> Any:
        ...

    def denom(self) -> Integer:
        ...

    def numer(self) -> Integer:
        ...

    def sign(self) -> Literal[-1, 0, 1]:
        ...
