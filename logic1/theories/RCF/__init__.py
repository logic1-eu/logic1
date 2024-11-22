"""A theory package for Real Closed Fields.
"""

from .atomic import VV, Eq, Ne, Ge, Le, Gt, Lt, polynomial_ring, Term, Variable  # noqa
from .bnf import cnf, dnf  # noqa
from .parser import l1  # noqa
from .qe import CLUSTERING, GENERIC, qe  # noqa
from . import redlog  # noqa
from .simplify import is_valid, simplify  # noqa

__all__ = [
    'Eq', 'Ne', 'Ge', 'Le', 'Gt', 'Lt', 'Term', 'VV',

    'cnf', 'dnf',

    'l1',

    'CLUSTERING', 'GENERIC', 'qe',

    'redlog',

    'simplify',

    'is_valid'
]
