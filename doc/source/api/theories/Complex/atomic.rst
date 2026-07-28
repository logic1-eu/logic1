.. _api-Complex-atomic:

*Complex*

***********************
Variables, Terms, Atoms
***********************

.. _api-Complex-atomic-terms:

Terms and Variables
*******************

.. automodule:: logic1.theories.Complex.term

  .. note::
    The global variable set :data:`VV <.Complex.term.VV>` is used to obtain
    :class:`Variables <.Complex.term.Variable>`. These can then be combined with
    the usual arithmetic operations  (:meth:`+ <.Complex.term.Term.__add__>`,
    :meth:`- <.Complex.term.Term.__sub__>`, :meth:`* <.Complex.term.Term.__mul__>`,
    :meth:`/ <.Complex.term.Term.__truediv__>`, :meth:`** <.Complex.term.Term.__pow__>`,
    :meth:`~ <.Complex.term.Term.__invert__>`), the imaginary unit
    :data:`I <.Complex.term.I>`, and the functions :func:`Re <.Complex.term.Re>`,
    :func:`Im <.Complex.term.Im>`, and :func:`Conj <.Complex.term.Conj>` to build
    larger :class:`Terms <.Complex.term.Term>`.

    >>> z = VV['z']
    >>> Re(z + I)
    1/2 * z + 1/2 * ~z

  .. autodata:: VV

  .. autofunction:: Re

  .. autofunction:: Im

  .. autofunction:: Conj

  .. autodata:: I

  .. autoclass:: Term
    :members:
    :special-members:
    :exclude-members: __hash__, __radd__, __rsub__, __rmul__, __rtruediv__

  .. autoclass:: SortKey
    :members:
    :special-members:
    :exclude-members:  __eq__, __hash__, __ge__, __init__, __gt__, __lt__, __ne__, __repr__, __weakref__

  .. autoclass:: Variable
    :members:
    :special-members:

  .. autoclass:: VariableSet
    :members:
    :special-members:

.. _api-Complex-atomic-atoms:

Atomic Formulas
***************

.. automodule:: logic1.theories.Complex.atomic

  .. note::
    Atomic formulas can be constructed from terms using the standard comparison
    operators (:class:`== <.Complex.atomic.Eq>`,
    :class:`\!= <.Complex.atomic.Ne>`, :class:`< <.Complex.atomic.Lt>`,
    :class:`<= <.Complex.atomic.Le>`, :class:`> <.Complex.atomic.Gt>`,
    :class:`>= <.Complex.atomic.Ge>`). Note that inequalities can only be
    constructed if both sides are real, otherwise a :class:`ValueError` is
    raised.

    >>> z = VV['z']
    >>> z == 1
    z == 1
    >>> Re(z) >= 0
    1/2 * z + 1/2 * ~z >= 0
    >>> z > 0
    Traceback (most recent call last):
    ...
    ValueError: Cannot create atomic formula z > 0 because it is not real

  .. autoclass:: AtomicFormula
    :members:
    :special-members:
    :exclude-members: __eq__,__hash__, __init__

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