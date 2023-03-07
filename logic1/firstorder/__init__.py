"""A package providing first-order formulas.

This package provides first-order formulas as a class :class:`formula.Formula`
to the outside world. Technically, :class:`formula.Formula` is an abstract base
class with a hierarchy of subclasses underneath.

Amnong those subclasses, there are subclasses :class:`quantified.Ex`,
:class:`quantified.All`, :class:`boolean.Equivalent`, :class:`boolean.Implies`,
:class:`boolean.And`, :class:`boolean.Or`, :class:`boolean.Not`,
:class:`truth._T`, and :class:`truth._F` for constructing subformulas with the
respective toplevel operator.

Furthermore, there is another abstract base class
:class:`atomic.AtomicFormula`, from which modules outside
:mod:`logic1.firstorder` derive classes that implement selections of atomic
formulas with certain relation symbols as their toplevel operators. The package
:mod:`logic1.firstorder` itself makes no assumptions on admissible relation
symbols and function symbols, or on the representation of atomic formulas
and terms. Of course, the abstract methods specified by
:class:`atomic.AtomicFormula` must be implemented.
"""

from .formula import Formula  # noqa

from .quantified import QuantifiedFormula, Ex, All  # noqa

from .boolean import BooleanFormula, Equivalent, Implies, AndOr, And, Or, Not  # noqa

from .truth import _T, T, _F, F  # noqa

from .atomic import AtomicFormula  # noqa

__all__ = [
    'Formula',

    'QuantifiedFormula', 'Ex', 'All',

    'BooleanFormula', 'AndOr', 'Equivalent', 'Implies', 'And', 'Or', 'Not',

    '_T', 'T', '_F', 'F',

    'AtomicFormula'
]
