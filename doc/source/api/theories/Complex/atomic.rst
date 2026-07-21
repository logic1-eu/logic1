.. _api-Complex-atomic:

*Complex*

***********************
Variables, Terms, Atoms
***********************

Terms and Variables
*******************

.. automodule:: logic1.theories.Complex.term

  .. autodata:: κ

  .. autoclass:: Term
    :members:
    :special-members:
    :exclude-members: __hash__, __radd__, __rsub__, __rmul__, __rtruediv__

  .. autoclass:: Variable
    :members:
    :special-members:

  .. autoclass:: VariableSet
    :members:
    :special-members:

  .. autodata:: VV

  .. autofunction:: Re

  .. autofunction:: Im

  .. autofunction:: Conj

  .. autodata:: I

Atomic Formulas
***************

.. automodule:: logic1.theories.Complex.atomic

  .. autoclass:: AtomicFormula
    :members:
    :special-members:

  .. autoclass:: RealAtomicFormula
    :members:
    :special-members:

  .. autoclass:: Eq
    :members:
    :special-members:
    :exclude-members:

  .. autoclass:: Ne
    :members:
    :special-members:
    :exclude-members:

  .. autoclass:: Le
    :members:
    :special-members:
    :exclude-members:

  .. autoclass:: Ge
    :members:
    :special-members:
    :exclude-members:

  .. autoclass:: Lt
    :members:
    :special-members:
    :exclude-members:

  .. autoclass:: Gt
    :members:
    :special-members:
    :exclude-members: