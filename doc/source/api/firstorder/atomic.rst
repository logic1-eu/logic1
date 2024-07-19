.. _api-firstorder-atomic:

***************
Terms and Atoms
***************

.. automodule:: logic1.firstorder.atomic


  .. autoclass:: Term
    :members:
    :undoc-members:

  .. autoclass:: Variable
    :members:
    :undoc-members:


  .. autoclass:: AtomicFormula
    :members:
    :undoc-members:
    :exclude-members: complement

    .. property:: complement
      :classmethod:

    The complement relation of an atomic formula, i.e.,
    :code:`a.complement(*a.args)` is an atomic formula equivalent to
    :code:`Not(a.op(*a.args))`.

    The implementation here raises :exc:`NotImplementedError` as a workaround
    for abstract class properties art not properly supported in Python.

    .. seealso::
      :attr:`logic1.theories.RCF.atomic.AtomicFormula.complement`
