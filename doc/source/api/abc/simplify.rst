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

  .. [DS97]
    A. Dolzmann, T. Sturm. Simplification of Quantifier-Free Formulae over
    Ordered Fields.  J. Symb. Comput. 24(2):209–231, 1997. Open access at
    `doi:10.1006/ jsco.1997.0123 <https://doi.org/10.1006/jsco.1997.0123>`_

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


  Simplification
  **************

  .. autoclass:: Simplify
    :members:


  Validity
  ********

  .. autoclass:: IsValid
    :members:
    :private-members: _simplify
    :undoc-members:
