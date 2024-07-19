from typing import TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Term, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable]
Prefix: TypeAlias = firstorder.Prefix[AtomicFormula, Term, Variable]
