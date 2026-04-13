"""A theory package for Real Closed Fields.
"""

from .atomic import (POLYLIB, AtomicFormula, cache_clear, cache_info, init_env,
                     init_env_arg, Eq, Ne, Ge, Le, Gt, Lt, Term, Variable, VV)
from .bnf import cnf, dnf
from .parser import l1
from .qe import Clustering, Generic, qe
from . import redlog
from .simplify import is_valid, simplify
from .types import Formula

from . import node

__all__ = [
    'POLYLIB', 'AtomicFormula', 'Eq', 'Ne', 'Ge', 'Le', 'Gt', 'Lt', 'Term',
    'Variable', 'VV',

    'cnf', 'dnf',

    'l1',

    'Clustering', 'Generic', 'qe',

    'redlog',

    'is_valid', 'simplify',

    'cache_clear', 'cache_info',

    'node',

    'Formula'
]
