.. _api-Sets-bnf:

*Sets*

********************
Boolean Normal Forms
********************

.. automodule:: logic1.theories.Sets.bnf

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