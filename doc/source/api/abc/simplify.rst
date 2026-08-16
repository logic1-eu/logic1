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
  representations and a type variable :data:`ω` for options used by the
  simplifier.

  .. data:: ρ
    :value: TypeVar('ρ', bound='InternalRepresentation')
    :canonical: logic1.abc.simplify.ρ

    A type variable denoting a type of variables with upper bound
    :class:`logic1.abc.simplify.InternalRepresentation`.

  .. data:: ω
    :value: TypeVar('ω', bound='Options')
    :canonical: logic1.abc.simplify.ω

    A type variable denoting a options for :meth:`.Simplify.simplify` with upper
    bound :class:`.Options`.


  Internal Representations
  ************************

  .. autoclass:: RESTART
    :members:

  .. autoclass:: InternalRepresentation
    :members:
    :exclude-members: __init__, __new__


  Simplification and Validity
  ***************************

  .. autoclass:: Options
    :exclude-members: __init__, __new__

  .. autoclass:: Simplify
    :members:
    :exclude-members: __init__, __new__
