"""A theory package for Sets.
"""

from .atomic import AtomicFormula, C, C_, Eq, Ne, oo, Variable, VV  # noqa
from .bnf import cnf, dnf  # noqa
from .qe import quantifier_elimination, qe  # noqa
from .simplify import is_valid, simplify  # noqa
from .types import Formula  # noqa

__all__ = [
    'AtomicFormula', 'C', 'C_', 'Eq', 'Ne', 'oo', 'Variable', 'VV',

    'cnf', 'dnf',
    
    'qe',
    
    'is_valid', 'simplify',
    
    'Formula'
]
