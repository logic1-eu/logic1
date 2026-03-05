from typing import Never

from logic1 import firstorder
from logic1.theories.Complex.atomic import AtomicFormula, Term, Variable

type Formula = firstorder.Formula[AtomicFormula, Term, Variable, Never]