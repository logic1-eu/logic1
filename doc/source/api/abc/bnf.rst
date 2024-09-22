.. _api-abc-bnf:

*Abstract Base Classes*

********************
Boolean Normal Forms
********************

.. attention::

  This documentation page addresses implementers rather than users. Concrete
  implemtations of the abstract classes described here are documented in the
  corresponding sections of the various domains:

  * :ref:`Boolean normal forms in Real Closed Fields <api-RCF-bnf>`
  * :ref:`Boolean normal forms in the theory of Sets <api-Sets-bnf>`

.. automodule:: logic1.abc.bnf

  Generic Types
  *************

  We use type variables :data:`.bnf.α`, :data:`.bnf.τ`, :data:`.bnf.χ`,
  :data:`.bnf.σ` in anology to their counterparts in :mod:`.formula`.

  .. data:: α
    :value: TypeVar('α', bound='AtomicFormula')
    :canonical: logic1.abc.bnf.α

  .. data:: τ
    :value: TypeVar('τ', bound='Term')
    :canonical: logic1.abc.bnf.τ

  .. data:: χ
    :value: TypeVar('χ', bound='Variable')
    :canonical: logic1.abc.bnf.χ

  .. data:: σ
    :value: TypeVar('σ')
    :canonical: logic1.abc.bnf.σ

  Boolean Normal Forms
  ********************

  .. autoclass:: BooleanNormalForm
    :members:
    :exclude-members: __init__
