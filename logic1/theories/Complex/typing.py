from typing import Never, TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Term, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable, Never]