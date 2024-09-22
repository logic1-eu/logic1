.. _api-RCF-bnf:

*Real Closed Fields*

********************
Boolean Normal Forms
********************

.. automodule:: logic1.theories.RCF.bnf

  .. autoclass:: BooleanNormalForm
    :special-members:

  User Interface
  **************

  .. function:: cnf(f: Formula) -> Formula
                dnf(f: Formula) -> Formula

    Compute a conjunctive or disjunctive normal form, respectively.

    :param f:
      The input formula

    :returns:
      Returns a CNF or DNF of `f`, respectively. If `f` contains quantifiers,
      then the result is an equivalent prenex normal form whose matrix is in CND
      or DNF, respectively.

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
