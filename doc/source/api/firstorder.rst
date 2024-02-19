.. _api-firstorder:

********************
First-order Formulas
********************

.. automodule:: logic1.firstorder


Formula Base Class
******************

.. automodule:: logic1.firstorder.formula

  .. autoclass:: Formula
    :members:
    :undoc-members:
    :exclude-members: func, args, __init__

    .. automethod:: __init__

    .. property:: func
      :type: Self
      :classmethod:

      This class property is supposed to be used read-only on instances of
      subclasses of :class:`Formula`. It yields the respective subclass.

    .. autoproperty:: args

    The properties :attr:`func` and :attr:`args` are useful for decomposing and
    reconstructing formulas, using the invariant :code:`f = f.func(*f.args)`,
    which holds for all formulas :code:`f`. This approach has been adopted from
    the `SymPy <https://www.sympy.org/>`_ project:

    .. doctest::

       >>> from logic1.firstorder import *
       >>> f = And(Implies(F, T), Or(T, Not(T)))
       >>> # The class of f:
       >>> f.func
       <class 'logic1.firstorder.boolean.And'>
       >>> # The argument tuple of f:
       >>> f.args
       (Implies(F, T), Or(T, Not(T)))
       >>> # The invariant:
       >>> f == f.func(*f.args)
       True
       >>> # Construction of a new formula using components of f:
       >>> f.func(Equivalent(T, T), *f.args)
       And(Equivalent(T, T), Implies(F, T), Or(T, Not(T)))

    The subclass constructors
    :class:`Not <logic1.firstorder.boolean.Not>`,
    :class:`And <logic1.firstorder.boolean.And>`,
    :class:`Or <logic1.firstorder.boolean.Or>`,
    :class:`Implies <logic1.firstorder.boolean.Implies>` are alternatively
    available as overloaded operators :code:`~`, :code:`&`, :code:`|`,
    :code:`>>`, respectively:

    .. method:: ~, &, |, >>, <<
                __invert__(other: Formula) -> Formula
                __and__(other: Formula) -> Formula
                __or__(other: Formula) -> Formula
                __rshift__(other: Formula) -> Formula
                __lshift__(other: Formula) -> Formula


Boolean Formulas
****************

.. automodule:: logic1.firstorder.boolean

  .. autoclass:: BooleanFormula
    :members:
    :undoc-members:

  .. autoclass:: Equivalent
    :members:
    :undoc-members:

  .. autoclass:: Implies
    :members:
    :undoc-members:

  .. autoclass:: And
    :members:
    :undoc-members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`Or`, which implements
      the dual operator :math:`\lor` or :math:`\land`.

  .. autoclass:: Or
    :members:
    :undoc-members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`And`, which implements
      the dual operator :math:`\land` or :math:`\lor`.

  .. autoclass:: Not
    :members:
    :undoc-members:

  .. autoclass:: _T
    :members:
    :undoc-members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`_F`, which implements the dual
      operator :math:`\bot` or :math:`\top`.

  .. autodata:: T
    :annotation: = _T()

  .. autoclass:: _F
    :members:
    :undoc-members:

      A class property yielding the class :class:`_F` itself.

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`_T`, which implements
      the dual operator :math:`\top` or :math:`\bot`.

  .. autodata:: F
    :annotation: = _F()


Atomic Formulas
***************

.. automodule:: logic1.firstorder.atomic

  .. autoclass:: AtomicFormula
    :members:
    :undoc-members:
    :exclude-members: complement_func

    .. property:: complement_func
      :classmethod:

      The complement func of an atomic formula. Let :code:`A` be a
      subclass of :class:`AtomicFormula`. Then
      :code:`A.complement_func(*args)` is equivalent to
      :code:`Not(A.func(*args))`.

      The implementation in here raises :exc:`NotImplementedError`, which is
      a workaround for missing abstract class properties. Relevant subclasses
      are implemented in various theories, e.g.,
      :class:`logic1.theories.RCF.rcf.AtomicFormula`

  .. autoclass:: Term
    :members:
    :undoc-members:


Quantified Formulas
*******************

.. automodule:: logic1.firstorder.quantified

  .. autoclass:: QuantifiedFormula
    :members:
    :undoc-members:

  .. autoclass:: Ex
    :members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the dual class :class:`All` of :class:`Ex`.

  .. autoclass:: All
    :members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the dual class :class:`Ex` of :class:`All`.

  .. autodata:: QuantifierBlock
