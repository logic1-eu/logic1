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

**********************
Gröbner Simplification
**********************

.. automodule:: logic1.theories.RCF.gsimplify

  By default, Gröbner simplification computes CNF using Logic1's
  :func:`~logic1.theories.RCF.bnf.cnf` with its default options. No Redlog
  installation is required.

  Pass ``use_redlog_cnf=True`` to :func:`gsimplify` to use Redlog's
  :func:`~logic1.theories.RCF.redlog.cnf` instead. The options
  :attr:`Options.bnfsac` and :attr:`Options.bnfsm` affect only Redlog CNF
  computation and are ignored when using native CNF.

  .. autofunction:: gsimplify(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], **options) -> RCF.types.Formula

  .. autoclass:: Options
    :members: bnfsac, bnfsm, radical, use_redlog_cnf
    :exclude-members: __init__, __new__

  .. autoclass:: GSimplify
    :members: Inconsistent
    :exclude-members: __call__, __init__, __new__
