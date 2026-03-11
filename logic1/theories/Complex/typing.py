from fractions import Fraction
from typing import Final

from gmpy2 import mpq

from logic1 import firstorder

type Number = int | float | Fraction | mpq | complex
_NUMBER_TYPES: Final = (int, float, Fraction, mpq, complex)

type Formula = firstorder.Formula[AtomicFormula, Term, Variable, Number]

from logic1.theories.Complex.term import Term, Variable
from logic1.theories.Complex.atomic import AtomicFormula