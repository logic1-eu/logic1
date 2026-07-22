from fractions import Fraction
from typing import Final, TypeAlias, TypeVar

from gmpy2 import mpq

import logic1


RationalNumber: TypeAlias = int | float | Fraction | mpq
"""Type alias for rational number types.
"""

_RATIONAL_NUMBER_TYPES: Final = (int, float, Fraction, mpq)
"""Tuple of all rational number types used for instance checking.
"""

Number: TypeAlias = RationalNumber | complex
"""Type alias for complex number types.
"""

_NUMBER_TYPES: Final = _RATIONAL_NUMBER_TYPES + (complex,)
"""Tuple of all complex number types used for instance checking.
"""

Formula: TypeAlias = logic1.firstorder.Formula[
    'logic1.theories.Complex.atomic.AtomicFormula',
    'logic1.theories.Complex.term.Term',
    'logic1.theories.Complex.term.Variable',
    Number]
"""
Type alias for formulas in the theory of complex numbers.
"""

α = TypeVar('α')
"""Generic type variable.
"""

η = TypeVar('η', bound='AST')
"""
Type variable for AST nodes used to parameterize :class:`.ast.SortKey`.
"""

τ = TypeVar('τ', bound='Term')
"""
Type variable for terms used to parameterize :class:`.term.SortKey`.
"""


from logic1.theories.Complex.ast import AST
from logic1.theories.Complex.term import Term