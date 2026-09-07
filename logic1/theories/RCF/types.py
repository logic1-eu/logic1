from __future__ import annotations

from fractions import Fraction
from typing import Final, TypeAlias

from gmpy2 import mpq

import logic1

Formula: TypeAlias = logic1.firstorder.Formula[
    'logic1.theories.RCF.atomic.AtomicFormula',
    'logic1.theories.RCF.term.Term',
    'logic1.theories.RCF.term.Variable',
    'logic1.theories.RCF.types.Number']
"""Type alias for :class:`.firstorder.formula.Formula` in the theory of Real Closed Fields.
"""

Prefix: TypeAlias = logic1.firstorder.Prefix['logic1.theories.RCF.term.Variable']
"""Type alias for :class:`.firstorder.quantified.Prefix` in the theory of Real Closed Fields.
"""

Number: TypeAlias = int | float | Fraction | mpq
"""Type alias for real number types.
"""

_NUMBER_TYPES: Final = (int, float, Fraction, mpq)
"""Tuple of all real types used for instance checking.
"""
