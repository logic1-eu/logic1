from typing import TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Variable, Variable]
Prefix: TypeAlias = firstorder.Prefix[AtomicFormula, Variable, Variable]
