"""Number types shared by the RCF polynomial backends."""

from fractions import Fraction
from typing import Final, TypeAlias

from gmpy2 import mpq


Number: TypeAlias = int | float | Fraction | mpq

_NUMBER_TYPES: Final = (int, float, Fraction, mpq)
