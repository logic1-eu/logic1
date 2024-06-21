.. _api-firstorder:

********************
First-order Formulas
********************

.. automodule:: logic1.firstorder

The following picture summarizes the inheritance hierarchy. Next, we are going
to describe the occurring classes starting at the top.

.. graphviz::

   digraph foo {
      bgcolor="#f3f4f5";
      edge [dir=back, arrowtail=empty];
      node [shape=plaintext, fontname="monospace"];
      "Formula" -> "QuantifiedFormula";
      "Formula" -> "BooleanFormula" ;
      "Formula" -> "AtomicFormula";
      QuantifiedFormula -> "Ex | All";
      "BooleanFormula" -> "Equivalent | Implies | And | Or | Not | _T | _F";
      AtomicFormula -> "RCF.AtomicFormula | ...";
   }

Formula Base Class
******************

.. automodule:: logic1.firstorder.formula

  .. autoclass:: Formula
    :members:
    :undoc-members:
    :exclude-members: func, args, __init__

    .. automethod:: __init__

    .. automethod:: _repr_latex_

    .. property:: func
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

    .. method:: ~, &, |, >>, <<
                __invert__(other: Formula) -> Formula
                __and__(other: Formula) -> Formula
                __or__(other: Formula) -> Formula
                __rshift__(other: Formula) -> Formula
                __lshift__(other: Formula) -> Formula

      The subclass constructors
      :class:`Not <logic1.firstorder.boolean.Not>`,
      :class:`And <logic1.firstorder.boolean.And>`,
      :class:`Or <logic1.firstorder.boolean.Or>`,
      :class:`Implies <logic1.firstorder.boolean.Implies>` are alternatively
      available as overloaded operators :code:`~`, :code:`&`, :code:`|`,
      :code:`>>`, respectively:

      >>> from logic1.firstorder import *
      >>> (F >> T) &  (T |  ~ T)
      And(Implies(F, T), Or(T, Not(T)))

      Furthermore, :code:`f1 << f2` constructs :code:`Implies(f2, f1)`. Note
      that the :external+python:ref:`Operator Precedence <operator-summary>` of
      Python applies and cannot be changed. In particular, the originally
      bitwise logical operators bind stronger than the comparison operators so
      that, unfortunately, equations and inequalities must be parenthesized:

      >>> from logic1.theories import RCF
      >>> a, b, x = RCF.VV.get('a', 'b', 'x')
      >>> f = Ex(x, (x >= 0) & (a*x + b == 0))
      >>> f
      Ex(x, And(x >= 0, a*x + b == 0))


Boolean Formulas
****************

.. automodule:: logic1.firstorder.boolean

  .. autoclass:: BooleanFormula
    :members:
    :undoc-members:

  .. autoclass:: Equivalent
    :members:
    :undoc-members:
    :exclude-members: __init__

  .. autoclass:: Implies
    :members:
    :undoc-members:
    :exclude-members: __init__

  .. autoclass:: And
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`Or`, which implements
      the dual operator :math:`\lor` or :math:`\land`.

    .. property:: definite_func
      :classmethod:

      A class property yielding the class :class:`_F`, which implements the
      definite operator :math:`\bot` of :math:`\wedge`. The definite operator is
      the dual of the neutral. Note that the return value :class:`_F` is the
      bare operator, in contrast to the formula :data:`F`.

    .. property:: neutral_func
      :classmethod:

      A class property yielding the class :class:`_T`, which implements the
      neutral operator :math:`\top` of :math:`\wedge.` Note that the return
      value :class:`_T` is the bare operator, in contrast to the formula
      :data:`T`.

  .. autoclass:: Or
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`And`, which implements
      the dual operator :math:`\land` or :math:`\lor`.

    .. property:: definite_func
      :classmethod:

      A class property yielding the class :class:`_T`, which implements the
      definite operator :math:`\top` of :math:`\vee`. The definite operator is
      the dual of the neutral. Note that the return value :class:`_T` is the
      bare operator, in contrast to the formula :data:`T`.

    .. property:: neutral_func
      :classmethod:

      A class property yielding the class :class:`_F`, which implements the
      neutral operator :math:`\bot` of :math:`\vee.` Note that the return
      value :class:`_F` is the bare operator, in contrast to the formula
      :data:`F`.

  .. autoclass:: Not
    :members:
    :undoc-members:
    :exclude-members: __init__

  .. autofunction:: involutive_not

  .. autoclass:: _T
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`_F`, which implements the dual
      operator :math:`\bot` of :math:`\top`.

  .. autodata:: T
    :annotation: = _T()

  .. autoclass:: _F
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`_T`, which implements
      the dual operator :math:`\top` of :math:`\bot`.

  .. autodata:: F
    :annotation: = _F()


Quantified Formulas
*******************

.. automodule:: logic1.firstorder.quantified

  .. autoclass:: QuantifiedFormula
    :special-members:

  .. autoclass:: Ex
    :members: var, arg
    :special-members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`All`, which implements the
      dual operator :math:`\forall` of :math:`\exists`.

  .. autoclass:: All
    :members: var, arg
    :special-members:

    .. property:: dual_func
      :classmethod:

      A class property yielding the class :class:`Ex`, which implements the
      dual operator :math:`\exists` of :math:`\forall`.

  .. autodata:: QuantifierBlock
