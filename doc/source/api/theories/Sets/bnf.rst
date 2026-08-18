.. _api-Sets-bnf:

*Sets with Cardinality Constraints*

********************
Boolean Normal Forms
********************

.. automodule:: logic1.theories.Sets.bnf

  .. function:: cnf(f: Sets.types.Formula) -> Sets.types.Formula
                dnf(f: Sets.types.Formula) -> Sets.types.Formula

    Compute a conjunctive or disjunctive normal form of ``f``. If ``f`` contains
    quantifiers, then the result is an equivalent prenex normal form whose
    matrix is in CNF or DNF, respectively.

    .. rubric:: Some examples

    >>> from logic1 import *
    >>> from logic1.theories.Sets import *
    >>> a, b, c, d = VV.get('a', 'b', 'c', 'd')

    >>> f = Equivalent(a == d, b == d)
    >>> cnf(f)
    And(Or(a == b, a != d), Or(a == b, b != d))
    >>> dnf(f)
    Or(a == b, And(a != d, b != d))

    >>> f = And(Or(a == d, b != d), Or(a != d, b == d))
    >>> cnf(f)
    And(Or(a == b, a != d), Or(a == b, b != d))
    >>> dnf(f)
    Or(a == b, And(a != d, b != d))

    >>> f = And(Or(a != d, b == d), Or(a == d, b == d))
    >>> cnf(f)
    And(Or(a == b, a != d), Or(a == d, b == d))
    >>> dnf(f)
    Or(And(a == b, a == d), And(b == d, a != b))

    .. seealso::

      :class:`.BooleanNormalForm`
        Its inherited methods :meth:`.BooleanNormalForm.cnf` and
        :meth:`.BooleanNormalForm.dnf` are wrapped by the functions
        :func:`.cnf` and :func:`.dnf`, respectively.

      :ref:`Simplification <api-Sets-simplify>`
        for the simplifier that is used to simplify intermediate results
        throughout the CNF and DNF computation.

  Details
  *******

  .. attention::
    The material below addresses implementers rather than users.

  .. autoclass:: BooleanNormalForm
    :exclude-members: __init__, __new__
