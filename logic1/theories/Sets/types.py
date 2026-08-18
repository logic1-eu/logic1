from typing import Never, TypeAlias

from ... import firstorder
from .atomic import AtomicFormula, Variable

Formula: TypeAlias = firstorder.Formula[AtomicFormula, Variable, Variable, Never]
"""Type alias for :class:`.firstorder.formula.Formula` in the theory of Sets.
"""

Prefix: TypeAlias = firstorder.Prefix[Variable]
"""Type alias for :class:`.firstorder.quantified.Prefix` in the theory of Sets.
"""