.. _api-Sets-simplify:

*Sets with Cardinality Constraints*

**************
Simplification
**************

.. automodule:: logic1.theories.Sets.simplify

  .. autofunction:: simplify(f: Sets.types.Formula, assume: Iterable[Sets.atomic.AtomicFormula] = []) -> Sets.types.Formula


  .. autofunction:: is_valid(f: Sets.types.Formula, assume: Iterable[Sets.atomic.AtomicFormula] = []) -> Optional[bool]


  Details
  *******

  .. attention::
    The material below addresses implementers rather than users.

  .. autoclass:: Simplify
    :exclude-members: __init__, __new__

  .. autoclass:: InternalRepresentation
    :exclude-members: __init__, __new__