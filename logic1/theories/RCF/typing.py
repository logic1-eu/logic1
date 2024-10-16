from typing import TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Term, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable, int]
Prefix: TypeAlias = firstorder.Prefix[Variable]
