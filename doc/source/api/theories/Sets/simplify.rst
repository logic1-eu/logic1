.. _api-Sets-simplify:

*Sets*

**************
Simplification
**************

.. automodule:: logic1.theories.Sets.simplify

  Theories
  ********

  .. autoclass:: Theory
    :special-members:

  Simplification and Validity
  ***************************

  .. autoclass:: Simplify
    :special-members:

  User Interface
  **************

  .. autofunction:: simplify

  .. autofunction:: is_valid

    >>> from logic1.firstorder import *
    >>> from logic1.theories.Sets import *
    >>> a, b, c, d = VV.get('a', 'b', 'c', 'd')
    >>> is_valid(a == d, assume=[a == b, b == c, c == d])
    True
    >>> is_valid(a == d, assume=[a == b, b != c, c == d])
    False
    >>> is_valid(a == d, assume=[a != b, b != c, c != d])  # Returns None
