r"""The module :mod:`logic1.theories.Sets` implements the theory of Sets with
Cardinality Constraints, based on the :mod:`logic1.firstorder` framework. The
name "Sets" emphasizes that the models considered are plain sets without any
constants or functions. We have equality and disequality, where the former is
generally not considered a formal relation in interpreted first-order logic, and
the latter can be considered a shorthand for the logical negation of equality.

The only formal structure actually available is an infinite set of constant
relations :math:`\{C_n\}_{n \in \mathbb{N} \cup \{\infty\}}`,
:math:`\{\overline{C_n}\}_{n \in \mathbb{N} \cup \{\infty\}}`, which are defined
as follows:

#. :math:`C_n` holds if and only if the cardinality of the universe is at least
   :math:`n`, for :math:`n \in \mathbb{N}`;
#. :math:`C_\infty` holds if and only if the universe is infinite;
#. :math:`\overline{C_n}` holds if and only if :math:`C_n` does not hold, for
   :math:`n \in \mathbb{N} \cup \{\infty\}`.

:ref:`Variables <api-Sets-atomic>` are elements of an infinite set :data:`Sets.VV
<.Sets.atomic.VV>`, indexed by their name as a string. Methods :meth:`VV.get
<.firstorder.term.VariableSet.get>` and :meth:`VV.imp
<.firstorder.term.VariableSet.imp>` allow to retrieve several variables at once,
given their names.

>>> from logic1.theories.Sets import VV
>>> x = VV['x']
>>> type(x)
<class 'logic1.theories.Sets.atomic.Variable'>

The module :mod:`.logic1.interactive.Sets` provides a convenient interface for
interactive use, e.g., pre-defining single letter variables.

>>> from logic1.interactive.Sets import *
>>> a
a
>>> type(a)
<class 'logic1.theories.Sets.atomic.Variable'>

In the absence of constants and functions, all terms in this theory are
variables, and there is no explicit class for terms.

:ref:`Atoms <api-Sets-atomic>` are built from Variables using equality,
disequality, and cardinality constraints.

>>> from logic1.theories.Sets import *
>>> x, y, z = VV.get('x', 'y' , 'z')
>>> x == y
x == y
>>> y != z
y != z
>>> C(2)
C(2)
>>> C_(99)
C_(99)

.. attention::

   The numbers ``2`` and ``99`` are not terms but indices that are used here for
   denoting two out of the infinitely many existing cardinality  constraints,
   namely :math:`C_2` and :math:`\overline{C_{99}}`. Both these constraints are
   relation symbols with arity zero.

:ref:`Formulas <api-firstorder-firstorder>` are built from atoms using Boolean
connectives and quantifiers. Sets with cardinality constraints are decidable but
not complete. They admit :ref:`quantifier elimination <api-Sets-qe>`.

>>> from logic1.firstorder import *
>>> from logic1.theories.Sets import *
>>> x, y, z = VV.get('x', 'y', 'z')
>>> qe(Ex([x, y, z], And(x == y, y != z)))
C(2)
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
