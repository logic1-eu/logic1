.. _api-Sets-simplify:

*Sets*

**************
Simplification
**************

.. automodule:: logic1.theories.Sets.simplify

  Internal Representations
  ************************

  .. autoclass:: InternalRepresentation
    :special-members:

  Simplification and Validity
  ***************************

  .. autoclass:: Simplify
    :special-members:

  User Interface
  **************

  .. autofunction:: simplify

    .. rubric:: An example

    >>> from logic1.firstorder import *
    >>> from logic1.theories.Sets import *
    >>> a, b, c, d = VV.get('a', 'b', 'c', 'd')
    >>> simplify(And(a == b, b == c, c == d,  d == c), assume=[a == b])
    And(a == c, a == d)

  .. autofunction:: is_valid

    .. rubric:: Some examples

    >>> from logic1.firstorder import *
    >>> from logic1.theories.Sets import *
    >>> a, b, c, d = VV.get('a', 'b', 'c', 'd')

    >>> is_valid(a == d, assume=[a == b, b == c, c == d])
    True

    >>> is_valid(a == d, assume=[a == b, b != c, c == d])
    False

    >>> is_valid(a == d, assume=[a != b, b != c, c != d])  # Returns None
