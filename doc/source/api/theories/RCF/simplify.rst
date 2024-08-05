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

  Simplification
  **************

  .. autoclass:: Simplify
    :members:
    :undoc-members:
    :exclude-members: explode_always, prefer_order, prefer_weak

    .. automethod:: __call__

  Validity
  ********

  .. autoclass:: IsValid
    :private-members: _simplify
    :special-members:

  Interface Functions
  *******************

  .. autofunction:: simplify

  .. autofunction:: is_valid
