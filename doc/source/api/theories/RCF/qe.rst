.. _api-RCF-qe:

*Real Closed Fields*

**********************
Quantifier Elimination
**********************

.. automodule:: logic1.theories.RCF.qe

  Assumptions
  ***********

  .. autoclass:: Assumptions
    :special-members:

  Nodes
  *****

  .. autoclass:: Node
    :special-members:

  Options
  *******

  .. autoclass:: CLUSTERING
    :members:

  .. autoclass:: GENERIC
    :members:

  .. autoclass:: Options
    :members: clustering, generic, traditional_guards
    :special-members:

  Quantifier Elimination
  **********************

  .. autoclass:: VirtualSubstitution
    :special-members:

  User Interface
  **************

  .. autofunction:: qe

.. discuss: autofunction qe gets its signature from
.. VirtualSubstitution.__call__. However, it shows
.. firstorder.atomic.AtomicFormula instead of RCF.atomic.AtomicFormula. This is
.. also the origin of a Warning when building the documentation.
