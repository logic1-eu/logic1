.. _api-abc-qe:

*Abstract Base Classes*

**********************
Quantifier Elimination
**********************

.. attention::

  This documentation page addresses implementers rather than users. Concrete
  implemtations of the abstract classes described here are documented in the
  corresponding sections of the various domains:

  * :ref:`Quantifier elimination in Real Closed Fields <api-RCF-qe>`
  * :ref:`Quantifier elimination in the theory of Sets <api-Sets-qe>`

.. automodule:: logic1.abc.qe

    Generic Types
    *************

    We use type variables :data:`.qe.α`, :data:`.qe.τ`,
    :data:`.qe.χ`, :data:`.qe.σ` in anology to their counterparts in
    the module :mod:`logic1.firstorder.formula`.

    .. data:: α
      :value: TypeVar('α', bound='AtomicFormula')
      :canonical: logic1.abc.qe.α

    .. data:: τ
      :value: TypeVar('τ', bound='Term')
      :canonical: logic1.abc.qe.τ

    .. data:: χ
      :value: TypeVar('χ', bound='Variable')
      :canonical: logic1.abc.qe.χ

    .. data:: σ
      :value: TypeVar('σ')
      :canonical: logic1.abc.qe.σ

    We introduce the following additional type variables.

    .. data:: φ
      :value: TypeVar('φ', bound=Formula)
      :canonical: logic1.abc.qe.φ

      A type variable denoting a formula with upper bound
      :class:`logic1.firstorder.formula.Formula`.

    .. data:: ν
      :value: TypeVar('ν', bound='Node')
      :canonical: logic1.abc.qe.ν

      A type variable denoting a node with upper bound
      :class:`.Node`.

    .. data:: ι
      :value: TypeVar('ι')
      :canonical: logic1.abc.qe.ι

      A type variable denoting the type of the principal argument of the
      abstract method :meth:`.QuantifierElimination.init_env`.

    .. data:: λ
      :value: TypeVar('λ', bound='Assumptions')
      :canonical: logic1.abc.qe.λ

      A type variable denoting a assumptions with upper bound
      :class:`.Assumptions`.

    .. data:: ω
      :value: TypeVar('ω', bound='Options')
      :canonical: logic1.abc.qe.ω

      A type variable denoting a options for
      :meth:`.QuantifierElimination.__call__` with upper bound
      :class:`.Options`.

    Assumptions
    ***********

    .. autoclass:: Assumptions
      :members: atoms, append, extend, simplify
      :special-members:

      .. autoclass:: logic1.abc.qe.Assumptions.Inconsistent
        :special-members:

    Nodes
    *****

    .. autoclass:: Node
      :members: variables, formula, copy, process
      :undoc-members:
      :special-members:

    Options
    *******

    .. autoclass:: Options
      :members: log_level, log_rate, workers
      :special-members:

    Quantifier Elimination
    **********************

    .. autoclass:: QuantifierElimination
      :exclude-members: __init__, __new__

      A first group of attributes holds the state of the computation:

      .. autoproperty:: assumptions
      .. autoattribute:: blocks
      .. autoattribute:: matrix
      .. autoattribute:: negated
      .. autoattribute:: root_nodes
      .. autoattribute:: working_nodes
      .. autoattribute:: success_nodes
      .. autoattribute:: failure_nodes
      .. autoattribute:: result

      Note that the parameter `f` of :meth:`__call__` inizializes
      :attr:`.blocks` and :attr:`matrix`, and the parameter `assume` inizializes
      :attr:`.assumptions`. The next group of attributes corresponds to
      read-only input parameters of
      :meth:`__call__`:

      .. autoattribute:: options

      The third and last group of attributes holds comprehensive timing
      information on the last computation. All times are wall times in seconds:

      .. autoattribute:: time_syncmanager_enter
      .. autoattribute:: time_start_first_worker
      .. autoattribute:: time_start_all_workers
      .. autoattribute:: time_multiprocessing
      .. autoattribute:: time_import_failure_nodes
      .. autoattribute:: time_import_success_nodes
      .. autoattribute:: time_import_working_nodes
      .. autoattribute:: time_syncmanager_exit
      .. autoattribute:: time_final_simplification
      .. autoattribute:: time_total

      .. automethod:: __call__
      .. automethod:: create_options
      .. automethod:: create_root_nodes
      .. automethod:: create_assumptions
      .. automethod:: create_true_node
      .. automethod:: final_simplify
      .. automethod:: init_env
      .. automethod:: init_env_arg
