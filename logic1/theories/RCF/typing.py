from fractions import Fraction
from typing import TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Term, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable, int | Fraction]
Prefix: TypeAlias = firstorder.Prefix[Variable]
