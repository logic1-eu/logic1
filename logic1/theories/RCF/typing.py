from typing import TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Term, Variable

RCF_Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable]
RCF_Prefix: TypeAlias = firstorder.Prefix[AtomicFormula, Term, Variable]
