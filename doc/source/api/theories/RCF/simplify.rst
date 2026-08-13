.. _api-RCF-simplify:

*Real Closed Fields*

**************
Simplification
**************

.. automodule:: logic1.theories.RCF.simplify

  .. autofunction:: simplify(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], **options) -> RCF.types.Formula

  .. autofunction:: is_valid(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], **options) -> Optional[bool]

  .. autoclass:: Options
    :members: explode_always, implicit_ranges, lift, prefer_order, prefer_weak, substitute
    :exclude-members: __init__, __new__


  Details
  *******

  .. attention::
    The material below addresses implementers rather than users.

  .. autoclass:: Simplify
    :exclude-members: __init__, __new__

  .. autoclass:: InternalRepresentation
    :exclude-members: __init__, __new__