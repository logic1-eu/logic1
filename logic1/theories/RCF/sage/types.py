from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import logic1
from logic1.theories.RCF.abc.types import Number, _NUMBER_TYPES

if TYPE_CHECKING:
    from logic1.theories.RCF.sage.atomic import AtomicFormula
    from logic1.theories.RCF.sage.term import Term, Variable

Formula: TypeAlias = logic1.firstorder.Formula[
    'logic1.theories.RCF.sage.atomic.AtomicFormula',
    'logic1.theories.RCF.sage.term.Term',
    'logic1.theories.RCF.sage.term.Variable',
    'logic1.theories.RCF.sage.types.Number']
"""Type alias for :class:`.firstorder.formula.Formula` in the theory of Real Closed Fields.
"""

Prefix: TypeAlias = logic1.firstorder.Prefix['logic1.theories.RCF.sage.term.Variable']
"""Type alias for :class:`.firstorder.quantified.Prefix` in the theory of Real Closed Fields.
"""

