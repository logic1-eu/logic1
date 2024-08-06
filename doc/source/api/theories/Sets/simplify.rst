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
    :members:
    :undoc-members:
    :exclude-members: __new__,  simpl_at,
      AtomicSortKey, SortKey

    .. automethod:: __call__

  Interface Functions
  *******************

  .. autofunction:: simplify

  .. autofunction:: is_valid

    This function establishes the user interface to a heuristic validity test.
    Technically, it is the method :meth:`.is_valid` of an instance of the
    callable class :class:`.Sets.simplify.Simplify`.

    :param f:
      The formula to be tested for validity

    :param assume:
      A list of atomic formulas that are assumed to hold. The result of the
      validity test is correct modulo those assumptions.