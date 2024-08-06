from typing import Never, TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Variable, Variable, Never]
Prefix: TypeAlias = firstorder.Prefix[AtomicFormula, Variable, Variable, Never]
