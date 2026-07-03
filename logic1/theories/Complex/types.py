from fractions import Fraction
from typing import Final, TypeAlias

from gmpy2 import mpq

from logic1 import firstorder

RationalNumber: TypeAlias = int | float | Fraction | mpq
"""Type alias for rational number types.
"""

_RATIONAL_NUMBER_TYPES: Final = (int, float, Fraction, mpq)

Number: TypeAlias = RationalNumber | complex
"""Type alias for complex number types.
"""

_NUMBER_TYPES: Final = _RATIONAL_NUMBER_TYPES + (complex,)

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable, Number]
"""
Type alias for formulas in the theory of complex numbers.
"""


from logic1.theories.Complex.term import Term, Variable
from logic1.theories.Complex.atomic import AtomicFormula