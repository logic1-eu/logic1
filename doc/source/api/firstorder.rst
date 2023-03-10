.. _api-firstorder:

********************
First-order Formulas
********************

.. automodule:: logic1.firstorder


.. autoclass:: Formula
  :members:
  :special-members: __and__, __invert__, __lshift__, __or__, __rshift__


.. autoclass:: QuantifiedFormula
  :members:
  :undoc-members:

.. autoclass:: Ex
  :members: latex_symbol, text_symbol

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`Ex` itself.

  .. property:: dual_func
    :classmethod:

    A class property yielding the dual class :class:`All` of :class:`Ex`.

.. autoclass:: All
  :members: latex_symbol, text_symbol

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`All` itself.

  .. property:: dual_func
    :classmethod:

    A class property yielding the dual class :class:`Ex` of :class:`All`.


.. autoclass:: BooleanFormula
  :members:
  :undoc-members:

.. autoclass:: Equivalent
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`Equivalent` itself.

.. autoclass:: Implies
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`Implies` itself.

.. autoclass:: AndOr
  :members:
  :undoc-members:

.. autoclass:: And
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`And` itself.

  .. property:: dual_func
    :classmethod:

    A class property yielding the class :class:`Or`, which implements
    the dual operator :math:`\lor` or :math:`\land`.

.. autoclass:: Or
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`Or` itself.

  .. property:: dual_func
    :classmethod:

    A class property yielding the class :class:`And`, which implements
    the dual operator :math:`\land` or :math:`\lor`.

.. autoclass:: Not
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`Not` itself.


.. autoclass:: TruthValue
  :members:
  :undoc-members:

.. autoclass:: _T
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`_T` itself.

  .. property:: dual_func
    :classmethod:

    A class property yielding the class :class:`_F`, which implements the dual
    operator :math:`\bot` or :math:`\top`.

.. autodata:: T
  :annotation: = _T()

.. autoclass:: _F
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding the class :class:`_F` itself.

  .. property:: dual_func
    :classmethod:

    A class property yielding the class :class:`_T`, which implements
    the dual operator :math:`\top` or :math:`\bot`.

.. autodata:: F
  :annotation: = _F()


.. autoclass:: AtomicFormula
  :members:
  :undoc-members:

  .. property:: func
    :classmethod:

    A class property yielding this class or the derived subclass itself.
