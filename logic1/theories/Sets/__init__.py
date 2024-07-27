"""A theory package for Sets.
"""

from .atomic import C, C_, Eq, Ne, oo, Variable, VV  # noqa
from .bnf import cnf, dnf  # noqa
from .qe import quantifier_elimination, qe  # noqa
from .simplify import simplify  # noqa

__all__ = [
    'C', 'C_', 'Eq', 'Ne', 'oo', 'VV',

    'cnf', 'dnf',

    'qe',

    'simplify'
]
