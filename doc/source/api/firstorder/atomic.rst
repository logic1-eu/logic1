.. _api-firstorder-atomic:

*Generic*

***************************
Terms, Variables, and Atoms
***************************

.. automodule:: logic1.firstorder.atomic

  Generic Types
  *************

  We use type variables :data:`.atomic.α`, :data:`.atomic.τ`, and
  :data:`.atomic.χ` with the same names and definitions as in module
  :mod:`.formula`.

  .. data:: α
    :value: TypeVar('α', bound='AtomicFormula')
    :canonical: logic1.firstorder.formula.α

    A type variable denoting a type of atomic formulas with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: τ
    :value: TypeVar('τ', bound='AtomicFormula')
    :canonical: logic1.firstorder.formula.τ

    A type variable denoting a type of terms with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: χ
    :value: TypeVar('χ', bound='AtomicFormula')
    :canonical: logic1.firstorder.formula.χ

    A type variable denoting a type of variables with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.


  Set of Variables
  ********************

  .. autoclass:: _VariableSet
    :members:
    :undoc-members:
    :special-members: __getitem__


  Terms
  *****

  .. autoclass:: Term
    :members: as_latex, sort_key, vars
    :special-members:


  Variables
  *********

  .. autoclass:: Variable
    :members:
    :undoc-members:

  Atomic Formulas
  ***************

  .. autoclass:: AtomicFormula
    :members:
    :private-members: _bvars, _fvars
    :undoc-members:
    :exclude-members: atoms
    :special-members: __le__
