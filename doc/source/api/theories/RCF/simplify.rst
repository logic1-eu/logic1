.. _api-RCF-simplify:

*Real Closed Fields*

**************
Simplification
**************

.. automodule:: logic1.theories.RCF.simplify

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
    >>> from logic1.theories.RCF import *
    >>> a, b, c = VV.get('a', 'b', 'c')
    >>> is_valid(3 * b**2 + c**2 >= 0)
    True
    >>> is_valid(3 * b**2 + c**2 < 0)
    False
    >>> is_valid(a * b**2 + c**2 >= 0, assume=[a > 0])  # returns None

    The last test returns :data:`None` because the simplifier is not strong
    enough to deduce :data:`.T`.
