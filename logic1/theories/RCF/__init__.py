"""The module :mod:`logic1.theories.RCF` implements the theory of Real Closed
Fields based on the :mod:`logic1.firstorder` framework. Real Closed Fields are
the first-order theory of the real numbers with ring arithmetic, equality, and
order. This can be naturally axiomatized as follows:

1. the axioms of ordered fields,
2. every positive element has a square root,
3. every polynomial of odd degree has a root.

:ref:`Variables <api-RCF-atomic>` are elements of an infinite set :data:`RCF.VV
<.RCF.term.VV>`, indexed by their name as a string.

>>> from logic1.theories.RCF import VV
>>> x = VV['x']
>>> isinstance(x, Variable)
True
>>> isinstance(x, Term)
True

Methods :meth:`VV.get <.firstorder.term.VariableSet.get>` and :meth:`VV.imp
<.firstorder.term.VariableSet.imp>` allow to retrieve several variables at once,
given their names. :ref:`Terms <api-RCF-atomic>` are built from variables and
rational numbers using ring arithmetic. All terms are implicitly converted to
polynomials with rational coefficients. We use sparse distributive
representation and deglex monomial ordering.

>>> from logic1.theories.RCF import *
>>> from gmpy2 import mpq
>>> x, y = VV.get('x', 'y')
>>> (2 * x - 3 * y) ** 3
8*x**3 - 36*x**2*y + 54*x*y**2 - 27*y**3

Admissible number types are :class:`.int`, :class:`gmpy2.mpq`, and
:class:`.float`. Fractions are entered as GNU multi-precision rational numbers
:class:`gmpy2.mpq`, while :class:`.float` provides a convenient interface
for external input with decimal numbers.

.. attention::
    Python division of integers yields a float, which can cause precision
    issues:

    >>> from logic1.theories.RCF import *
    >>> x, = VV.get('x')
    >>> x + 0.1
    x + 1/10
    >>> x + (1/10 + 2/10)
    x + 415716888680356/1385722962267853

    Use GNU multi-precision rational numbers for exact arithmetic:

    >>> from logic1.theories.RCF import *
    >>> from gmpy2 import mpq
    >>> x, = VV.get('x')
    >>> x + (mpq(1, 10) + mpq(2, 10))
    x + 3/10

The module :mod:`.logic1.interactive.RCF` provides a convenient interface for
interactive use, e.g., pre-defining single letter variables.

:ref:`Atoms <api-RCF-atomic>` are built from polynomials using equality,
disequality, and order relations. :ref:`Formulas <api-firstorder-firstorder>`
are built from atoms using Boolean connectives and quantifiers. Real closed
fields are complete, decidable, and admit :ref:`quantifier elimination
<api-RCF-qe>`.

>>> from logic1.interactive.RCF import *
>>> a1 = -(1 - 3*r) * (a**2 + b**2) + 2*a*r
>>> a2 = -(2 - 3*r) * (a**2 + b**2) + 4*a*r - 2*a - r
>>> collins_johnson = Ex(r, And(0 < r, r < 1, a >= 1/2, b > 0, a1 < 0, a2 > 0))
>>> qe(collins_johnson)
 And(b > 0,
     2*a - 1 >= 0,
     3*a**4 + 6*a**2*b**2 + 3*b**4 + 7*a**3 + 7*a*b**2 + 3*a**2 - b**2 - a > 0,
     3*a**4 + 6*a**2*b**2 + 3*b**4 + 10*a**3 + 10*a*b**2 + 4*a**2 - 4*b**2 - 6*a + 1 > 0,
     9*a**6 + 27*a**4*b**2 + 27*a**2*b**4 + 9*b**6 + 30*a**5 + 60*a**3*b**2 + 30*a*b**4
        + 36*a**4 + 36*a**2*b**2 + 14*a**3 - 2*a*b**2 - 5*a**2 - b**2 < 0)

Numbers obtained from terms or formulas using :mod:`.theories.RCF` operations
are generally of type :class:`gmpy2.mpq`.

>>> from logic1.interactive.RCF import *
>>> t = 0.1 * x + 2
>>> t.lc()
mpq(1,10)
>>> t.constant_coefficient()
mpq(2,1)
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
