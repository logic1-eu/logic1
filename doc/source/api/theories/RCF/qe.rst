.. _api-RCF-qe:

*Real Closed Fields*

**********************
Quantifier Elimination
**********************

.. automodule:: logic1.theories.RCF.qe

  .. autoclass:: Node
    :special-members:

  .. autoclass:: Theory
    :special-members:

  .. autoclass:: Options
    :special-members:

  .. autoclass:: VirtualSubstitution
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__, collect_success_nodes, final_simplification,
      parallel_process_block, parallel_process_block_worker, pop_block,
      process_block, sequential_process_block, setup, status, timings,
      virtual_substitution

    .. automethod:: __call__

  .. autofunction:: qe

.. discuss: autofunction qe gets its signature from
.. VirtualSubstitution.__call__. However, it shows
.. firstorder.atomic.AtomicFormula instead of RCF.atomic.AtomicFormula. This is
.. also the origin of a Warning when building the documentation.
