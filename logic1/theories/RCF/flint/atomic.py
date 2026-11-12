from __future__ import annotations

from logic1.theories import RCF
from logic1.theories.RCF.abc.types import Number


class AtomicFormula(RCF.abc.AtomicFormula["RCF.flint.atomic.AtomicFormula",
                                          "RCF.flint.term.Term",
                                          "RCF.flint.term.Variable",
                                          "Eq",
                                          "Ge",
                                          "Gt",
                                          "Le",
                                          "Lt",
                                          "Ne"]):
    """Flint-specific class for atomic formulas over real closed fields. The
    class is the common parent of :class:`Eq <.RCF.flint.atomic.Eq>`, :class:`Ne
    <.RCF.flint.atomic.Ne>`, :class:`Le <.RCF.flint.atomic.Le>`, :class:`Ge
    <.RCF.flint.atomic.Ge>`, :class:`Lt <.RCF.flint.atomic.Lt>`, and :class:`Gt
    <.RCF.flint.atomic.Gt>`. It is not intended to be instantiated directly. Use
    one of the concrete relation classes instead.
    """

    @staticmethod
    def _eq_type() -> type[Eq]:
        return Eq

    @staticmethod
    def _ge_type() -> type[Ge]:
        return Ge

    @staticmethod
    def _gt_type() -> type[Gt]:
        return Gt

    @staticmethod
    def _le_type() -> type[Le]:
        return Le

    @staticmethod
    def _lt_type() -> type[Lt]:
        return Lt

    @staticmethod
    def _ne_type() -> type[Ne]:
        return Ne

    @staticmethod
    def _term_type() -> type[Term]:
        return Term


class Eq(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(lhs, rhs)

class Ne(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(lhs, rhs)


class Ge(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(lhs, rhs)


class Le(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(lhs, rhs)


class Gt(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(lhs, rhs)


class Lt(AtomicFormula):

    def __init__(self, lhs: Number | Term, rhs: Number | Term) -> None:
        super().__init__(lhs, rhs)


from logic1.theories.RCF.flint.term import Term