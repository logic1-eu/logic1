from importlib.metadata import version as _version

try:
    __version__ = _version("logic1")
except Exception:
    from setuptools_scm import get_version as _get_version
    try:
        __version__ = _get_version()
    except Exception:
        __version__ = "unknown"

from . import firstorder

from .firstorder import (Formula, AtomicFormula, Term, Variable,  # noqa
                         BooleanFormula, Equivalent, Implies, And, Or, Not,
                         T, F, QuantifiedFormula, Ex, All, Prefix)

from . import theories

from .theories import Complex, RCF, Sets  # noqa

__all__ = firstorder.__all__ + theories.__all__
