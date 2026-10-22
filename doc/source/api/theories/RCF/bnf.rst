.. _api-RCF-bnf:

*Real Closed Fields*

********************
Boolean Normal Forms
********************

.. automodule:: logic1.theories.RCF.bnf

  .. function:: cnf(f, method='pyeda', *, simplification=Simplification.NONE, max_pair_checks=500)
                dnf(f, method='pyeda', *, simplification=Simplification.NONE, max_pair_checks=500)

    Compute a conjunctive or disjunctive normal form of ``f``. If ``f`` contains
    quantifiers, then the result is an equivalent prenex normal form whose
    matrix is in CNF or DNF, respectively.

    ``method`` is ``'pyeda'`` (the default) or ``'cartesian'``. The separate
    ``simplification`` option accepts a :class:`Simplification` member or its
    integer value. The default, ``NONE`` (0), retains ordinary preprocessing
    and conservative final simplification without additional DNF reductions.
    The other schedules are currently supported only by ``'cartesian'``:

    * ``FINAL`` (1): reduce the complete strict DNF once.
    * ``AFTER_STEP`` (2): reduce each completed Boolean node.
    * ``DURING_STEP`` (3): reduce cubes and partial products during construction
      at each Boolean node.

    These schedules share atomic rules and final semantic subsumption; larger
    integer values do not guarantee better or faster results.
    ``max_pair_checks`` bounds the final semantic comparisons for these three
    schedules: 500 by default, zero to skip them, and ``None`` for no limit.
    It does not bound atomic reductions or Cartesian construction. Unsupported
    combinations of algorithms and schedules raise an exception.

    .. rubric:: Some examples

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c = VV.get('a', 'b', 'c')

    >>> f = Equivalent(a == 0, b == 0)
    >>> cnf(f)
    And(Or(b == 0, a != 0), Or(b != 0, a == 0))
    >>> dnf(f)
    Or(And(b == 0, a == 0), And(b != 0, a != 0))

    >>> f = And(Or(a > 0, b == 0), Or(a <= 0, b == 0))
    >>> cnf(f)
    b == 0
    >>> dnf(f)
    b == 0

    >>> f = Ex([a, b], All([c], Equivalent(c == 0, And(a == 0, b == 0))))
    >>> cnf(f)
    Ex(a, Ex(b, All(c, And(Or(c == 0, b != 0, a != 0),
                           Or(c != 0, b == 0),
                           Or(c != 0, a == 0)))))
    >>> dnf(f)
    Ex(a, Ex(b, All(c, Or(And(c == 0, b == 0, a == 0),
                          And(c != 0, b != 0),
                          And(c != 0, a != 0)))))

    >>> f = Equivalent(Ex([a, b], And(a == 0, b == 0)),
    ...                All(c, And(a == 0, c == 0)))
    >>> dnf(f)
    Ex(G0001_c, Ex(G0002_a, Ex(G0001_b, All(G0001_a, All(b, All(c,
        Or(And(c == 0, a == 0, G0002_a == 0, G0001_b == 0),
           And(c == 0, a == 0, G0001_c != 0),
           And(b != 0, a != 0),
           And(b != 0, G0002_a == 0, G0001_b == 0),
           And(b != 0, G0001_c != 0),
           And(a != 0, G0001_a != 0),
           And(G0002_a == 0, G0001_b == 0, G0001_a != 0),
           And(G0001_c != 0, G0001_a != 0))))))))
    >>> cnf(f)
    All(G0003_a, All(b, All(c, Ex(G0002_c, Ex(G0004_a, Ex(G0002_b,
        And(Or(c == 0, b != 0, G0003_a != 0),
            Or(b != 0, a == 0, G0003_a != 0),
            Or(a != 0, G0004_a == 0, G0002_c != 0),
            Or(a != 0, G0002_c != 0, G0002_b == 0))))))))

    .. seealso::

      :ref:`Simplification <api-RCF-simplify>`
        for the RCF simplifier used before and after normal form construction,
        and for final semantic subsumption in the simplifying schedules.

      ``doc/notes/RCF/dnf.md``
        for the strict DNF simplification rules and their growth limits.


  Details
  *******

  .. attention::
    The material below addresses implementers rather than users.

  .. autoclass:: CartesianBooleanNormalForm
    :exclude-members: __init__, __new__

  .. autoclass:: Simplification
    :members:

  .. autoclass:: SimplifyingCartesianBooleanNormalForm
    :exclude-members: __new__

  .. autoclass:: FinalSimplifyingCartesianBooleanNormalForm
    :exclude-members: __new__

  .. autoclass:: AfterStepSimplifyingCartesianBooleanNormalForm
    :exclude-members: __new__

  .. autoclass:: DuringStepSimplifyingCartesianBooleanNormalForm
    :exclude-members: __new__

  .. autoclass:: PyedaBooleanNormalForm
    :exclude-members: __init__, __new__
