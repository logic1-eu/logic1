from typing import TypeAlias

from logic1 import firstorder
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable, int]
Prefix: TypeAlias = firstorder.Prefix[Variable]
