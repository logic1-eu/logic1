"""A first-order theory of complex numbers following [FarossSturm-2026]_.
"""

# order to avoid circular imports:
# types, ast, format, normalize, term, atomic, qe, simplify

from .types import Formula
from .normalize import cartesian_normal_form, conjugate_normal_form
from .term import Conj, I, Im, Term, Re, Variable, VV
from .atomic import AtomicFormula, Eq, Ne, Ge, Le, Gt, Lt
from .qe import qe, real_normal_form
from .simplify import is_valid, simplify

__all__ = [
    'AtomicFormula', 'Eq', 'Ne', 'Ge', 'Le', 'Gt', 'Lt',

    'Conj', 'I', 'Im', 'Term', 'Re', 'Variable', 'VV',

    'cartesian_normal_form', 'conjugate_normal_form',

    'qe', 'real_normal_form',

    'is_valid', 'simplify',

    'Formula'
]
