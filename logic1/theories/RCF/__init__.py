"""A theory package for Real Closed Fields.
"""

from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ne, Ge, Le, Gt, Lt
from logic1.theories.RCF.term import (POLYLIB, cache_clear, cache_info, init_env, init_env_arg,
                                      Term, Variable, VV)
from logic1.theories.RCF.bnf import cnf, dnf
from logic1.theories.RCF.parser import l1
from logic1.theories.RCF.qe import Clustering, Generic, qe
from logic1.theories.RCF import redlog
from logic1.theories.RCF.simplify import is_valid, simplify
from logic1.theories.RCF.types import Formula
from logic1.theories.RCF import node

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
