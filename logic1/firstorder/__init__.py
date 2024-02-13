"""A package providing first-order formulas.

This package provides first-order formulas as a class :class:`Formula` to the
outside world. Technically, :class:`Formula` is an abstract base class with a
hierarchy of subclasses underneath.

Amnong those subclasses, there are subclasses :class:`Ex`, :class:`All`,
:class:`Equivalent`, :class:`Implies`, :class:`And`, :class:`Or`, :class:`Not`,
:class:`_T`, and :class:`_F` for constructing subformulas with the respective
toplevel operator.

Furthermore, there is another abstract base class :class:`AtomicFormula`, from
which modules outside :mod:`logic1.firstorder` derive classes that implement
selections of atomic formulas with certain relation symbols as their toplevel
operators. The package :mod:`logic1.firstorder` itself makes no assumptions on
admissible relation symbols and function symbols, or on the representation of
atomic formulas and terms. Of course, the abstract methods specified by
:class:`AtomicFormula` must be implemented.
"""

from .formula import Formula  # noqa

from .atomic import AtomicFormula, Term, Variable  # noqa

from .boolean import BooleanFormula, Equivalent, Implies, And, Or, Not, _T, T, _F, F  # noqa

from .quantified import QuantifiedFormula, Ex, All  # noqa

from .pnf import pnf  # noqa


__all__ = [
    'Formula',

    'QuantifiedFormula', 'Ex', 'All',

    'BooleanFormula', 'Equivalent', 'Implies', 'And', 'Or', 'Not',

    '_T', 'T', '_F', 'F',

    'AtomicFormula',

    'pnf'
]
