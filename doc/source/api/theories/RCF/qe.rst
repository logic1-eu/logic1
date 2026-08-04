.. _api-RCF-qe:

*Real Closed Fields*

**********************
Quantifier Elimination
**********************

.. automodule:: logic1.theories.RCF.node

  Assumptions
  ***********

  .. autoclass:: Assumptions
    :exclude-members: __init__, __new__

  Nodes
  *****

  .. autoclass:: Node
    :exclude-members: __init__, __new__

  Options
  *******

  .. autoclass:: Clustering
    :members:

  .. autoclass:: Generic
    :members:

.. automodule:: logic1.theories.RCF.qe

  .. autoclass:: Options
    :members: clustering, generic, traditional_guards
    :exclude-members: __init__, __new__

  Quantifier Elimination
  **********************

  .. autoclass:: VirtualSubstitution
    :exclude-members: __init__, __new__

  User Interface
  **************

  .. autofunction:: qe(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], **options) -> Optional[RCF.types.Formula]

.. discuss: autofunction qe gets its signature from
.. VirtualSubstitution.__call__. However, it shows
.. firstorder.atomic.AtomicFormula instead of RCF.atomic.AtomicFormula. This is
.. also the origin of a Warning when building the documentation.
