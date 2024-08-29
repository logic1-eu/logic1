.. _api-Sets-qe:

*Sets*

**********************
Quantifier Elimination
**********************

.. automodule:: logic1.theories.Sets.qe

  Theories
  ********

  .. autoclass:: Theory
    :special-members:

  Nodes
  *****

  .. autoclass:: Node
    :special-members:

  Quantifier Elimination
  **********************

  .. autoclass:: QuantifierElimination
    :special-members:

  User Interface
  **************

  .. autofunction:: qe

  .. rubric:: Some examples

  >>> from logic1.firstorder import *
  >>> from logic1.theories.Sets import *
  >>> a, u, v, w, x, y, z = VV.get('a', 'u', 'v', 'w', 'x', 'y', 'z')

  For the following input formula to hold, there must be at least two
  different elements in the universe. We derive this information via
  quantifier elimination:

  >>> qe(Ex([x, y], x != y))
  C(2)

  .. seealso:: The documentation of :class:`.C` and :class:`.C_`.

  In the next example, we learn that there must be at most one element in the
  universe:

  >>> qe(All(u, Ex(w, All(x, Ex([y, v],
  ...   And(Or(u == v, v != w), ~ Equivalent(u == x, u != w), y == a))))))
  C_(2)

  In the next example, the cardinality of the universe must be exactly three:

  >>> qe(Ex([x, y, z],
  ...   And(x != y, x != z, y != z, All(u, Or(u == x, u == y, u == z)))))
  And(C(3), C_(4))

  In our final example, the cardinality of the universe must be exactly one or
  at least 4:

  >>> qe(Implies(Ex([w, x], w != x),
  ...            Ex([w, x, y, z],
  ...              And(w != x, w != y, w != z, x != y, x != z, y != z))))
  Or(C_(2), C(4))
