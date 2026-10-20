from abc import abstractmethod

import logic1
from logic1.theories.RCF.abc.term import Term, Variable
from logic1.theories.RCF.abc.types import Number

class AtomicFormula[
        α: AtomicFormula,
        τ: Term,
        χ: Variable,
      ](logic1.firstorder.AtomicFormula[α, τ, χ, Number]):

    @abstractmethod
    def __init__(self, lhs: Number | τ, rhs: Number | τ) -> None:
        """Initialize an atomic formula with the given left- and right-hand side
        terms."""
        super().__init__()
