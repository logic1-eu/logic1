from importlib.metadata import version as _version

__version__ = _version("logic1")

from . import firstorder

from .firstorder import (Formula, AtomicFormula, Term, Variable,  # noqa
                         BooleanFormula, Equivalent, Implies, And, Or, Not,
                         T, F, QuantifiedFormula, Ex, All, Prefix)

from . import theories

from .theories import Complex, RCF, Sets  # noqa

__all__ = firstorder.__all__ + theories.__all__
