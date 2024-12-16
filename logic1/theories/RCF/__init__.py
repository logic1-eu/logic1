"""A theory package for Real Closed Fields.
"""
from . import atomic
from . import simplify as module_simplify

from .atomic import Eq, Ne, Ge, Le, Gt, Lt, Term, Variable, VV
from .bnf import cnf, dnf
from .parser import l1
from .qe import CLUSTERING, GENERIC, qe
from . import redlog
from .simplify import is_valid, simplify

__all__ = [
    'Eq', 'Ne', 'Ge', 'Le', 'Gt', 'Lt', 'Term', 'Variable', 'VV',

    'cnf', 'dnf',

    'l1',

    'CLUSTERING', 'GENERIC', 'qe',

    'redlog',

    'is_valid', 'simplify',

    'cache_clear', 'cache_info'
]


_caches = [atomic.Term.factor,
           module_simplify._SubstValue.as_term,
           module_simplify.Simplify._simpl_at]


def cache_clear():
    for cache in _caches:
        cache.cache_clear()


def cache_info():
    return {cache.__wrapped__: cache.cache_info() for cache in _caches}
