.. _api-RCF-qe:

*Real Closed Fields*

**********************
Quantifier Elimination
**********************

.. automodule:: logic1.theories.RCF.qe

  .. autofunction:: qe(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], **options) -> Optional[RCF.types.Formula]

  .. autoclass:: Options
    :members: clustering, elimination_order, generic, log_level, log_rate, traditional_guards, workers, xopt
    :exclude-members: __init__, __new__
    :member-order: alphabetical

.. autoclass:: logic1.theories.RCF.node.Clustering
  :members:
  :undoc-members:

.. autoclass:: logic1.theories.RCF.node.Generic
  :members:
  :undoc-members:


Details
*******

.. attention::
  The material below addresses implementers rather than users.

.. autoclass:: logic1.theories.RCF.qe.VirtualSubstitution
  :exclude-members: __init__, __new__

.. autoclass:: logic1.theories.RCF.node.Node
  :exclude-members: __init__, __new__

.. autoclass:: logic1.theories.RCF.node.Assumptions
  :exclude-members: __init__, __new__
