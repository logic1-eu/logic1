from typing import TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Variable

Sets_Formula: TypeAlias = firstorder.Formula[AtomicFormula, Variable, Variable]
Sets_Prefix: TypeAlias = firstorder.Prefix[AtomicFormula, Variable, Variable]
