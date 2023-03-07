.. _api-firstorder:


Package :mod:`logic1.firstorder`
================================

.. automodule:: logic1.firstorder

  The following code block from :file:`__init__.py` documents the re-export of
  relevant objects:

  .. code-block::

    from .formula import Formula
    from .quantified import QuantifiedFormula, Ex, All
    from .boolean import BooleanFormula, AndOr, Equivalent, Implies, And, Or, Not
    from .truth import T, F
    from .atomic import AtomicFormula


Module :mod:`formula <logic1.firstorder.formula>`
-------------------------------------------------

.. automodule:: logic1.firstorder.formula

.. autoclass:: Formula
  :members:
  :undoc-members:
  :special-members: __and__, __invert__, __lshift__, __or__, __rshift__


Module :mod:`quantified <logic1.firstorder.quantified>`
-------------------------------------------------------

.. automodule:: logic1.firstorder.quantified

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


Module :mod:`boolean <logic1.firstorder.boolean>`
-------------------------------------------------

.. automodule:: logic1.firstorder.boolean

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


Module :mod:`truth <logic1.firstorder.truth>`
---------------------------------------------

.. automodule:: logic1.firstorder.truth

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


Module :mod:`atomic <logic1.firstorder.atomic>`
-----------------------------------------------

.. automodule::
  logic1.firstorder.atomic

.. autoclass:: AtomicFormula
  :members:
  :undoc-members:
