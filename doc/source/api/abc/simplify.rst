.. _api-abc-simplify:

*Abstract Base Classes*

**************
Simplification
**************

.. attention::

  This documentation page addresses implementors rather than users. Concrete implemtations of the abstract classes described here are documented in the corresponding sections of the various domains:

  * :ref:`Simplification in Real Closed Fields <api-RCF-simplify>`
  * :ref:`Simplification in the theory of Sets <api-Sets-simplify>`

.. automodule:: logic1.abc.simplify


  Generic Types
  *************

  We use type variables :data:`.simplify.α`, :data:`.simplify.τ`, and
  We use type variables :data:`.simplify.α`, :data:`.simplify.τ`, and
  :mod:`.formula`.

  .. data:: α
    :value: TypeVar('α', bound='AtomicFormula')
    :canonical: logic1.abc.simplify.α

    A type variable denoting a type of atomic formulas with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: τ
    :value: TypeVar('τ', bound='AtomicFormula')
    :canonical: logic1.abc.simplify.τ

    A type variable denoting a type of terms with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: χ
    :value: TypeVar('χ', bound='AtomicFormula')
    :canonical: logic1.abc.simplify.χ

    A type variable denoting a type of variables with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: σ
    :value: TypeVar('σ')
    :canonical: logic1.abc.simplify.σ

    A type variable denoting a type that is admissible in addition to terms as a
    dictionary entry in :meth:`.formula.subs`. Instances of type :data:`.σ` that
    are passed to :meth:`.formula.subs` must not contain any variables. A
    typical example is setting :data:`σ` to :class:`int` in the theory of real
    closed fields.

  Additionally, we introduce a type variable :data:`θ` for theories internally
  used by the simplifier.

  .. data:: θ
    :value: TypeVar('θ', bound='Theory')
    :canonical: logic1.abc.simplify.θ

    A type variable denoting a type of variables with upper bound
    :class:`logic1.abc.simplify.Theory`.


  Theories
  ********

  .. autoclass:: Theory
    :members: add, extract, next_
    :special-members:


  Simplification and Validity
  ***************************

  .. autoclass:: Simplify
    :members:
