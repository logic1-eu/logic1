.. _api-abc-simplify:

*Abstract Base Classes*

**************
Simplification
**************

.. attention::

  This documentation page addresses implementers rather than users. Concrete
  implemtations of the abstract classes described here are documented in the
  corresponding sections of the various domains:

  * :ref:`Simplification in Real Closed Fields <api-RCF-simplify>`
  * :ref:`Simplification in the InternalRepresentation of Sets <api-Sets-simplify>`

.. automodule:: logic1.abc.simplify


  Generic Types
  *************

  We use type variables :data:`.simplify.α`, :data:`.simplify.τ`,
  :data:`.simplify.χ`, :data:`.simplify.σ` in anology to their counterparts in
  :mod:`.formula`.

  .. data:: α
    :value: TypeVar('α', bound='AtomicFormula')
    :canonical: logic1.abc.simplify.α

  .. data:: τ
    :value: TypeVar('τ', bound='Term')
    :canonical: logic1.abc.simplify.τ

  .. data:: χ
    :value: TypeVar('χ', bound='Variable')
    :canonical: logic1.abc.simplify.χ

  .. data:: σ
    :value: TypeVar('σ')
    :canonical: logic1.abc.simplify.σ

  Additionally, we introduce a type variable :data:`ρ` for internal
  representations used by the simplifier.

  .. data:: ρ
    :value: TypeVar('ρ', bound='InternalRepresentation')
    :canonical: logic1.abc.simplify.ρ

    A type variable denoting a type of variables with upper bound
    :class:`logic1.abc.simplify.InternalRepresentation`.


  Internal Representations
  ************************

  .. autoclass:: InternalRepresentation
    :members: add, extract, next_
    :special-members:


  Simplification and Validity
  ***************************

  .. autoclass:: Simplify
    :members:
