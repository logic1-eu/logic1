from typing import TypeAlias

from logic1 import firstorder
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.atomic import AtomicFormula

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Term, Variable, int]
"""Type alias for :class:`.firstorder.formula.Formula` in the theory of Real Closed Fields.
"""

Prefix: TypeAlias = firstorder.Prefix[Variable]
"""Type alias for :class:`.firstorder.quantified.Prefix` in the theory of Real Closed Fields.
"""
