.. _api-RCF-bnf:

*Real Closed Fields*

********************
Boolean Normal Forms
********************

.. automodule:: logic1.theories.RCF.bnf

  .. function:: cnf(f: RCF.types.Formula) -> RCF.types.Formula
                dnf(f: RCF.types.Formula) -> RCF.types.Formula

    Compute a conjunctive or disjunctive normal form of ``f``. If ``f`` contains
    quantifiers, then the result is an equivalent prenex normal form whose
    matrix is in CNF or DNF, respectively.

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
        * :class:`.BooleanNormalForm` -- Its inherited methods
          :meth:`.BooleanNormalForm.cnf` and :meth:`.BooleanNormalForm.dnf` are
          wrapped by the functions :func:`.cnf` and :func:`.dnf`, respectively.
        * :ref:`Simplification <api-RCF-simplify>` -- for the simplifier that is
          used to simplify intermediate results throughout the CNF and DNF
          computation.


  Details
  *******

  .. attention::
    The material below addresses implementers rather than users.

  .. autoclass:: BooleanNormalForm
    :exclude-members: __init__, __new__