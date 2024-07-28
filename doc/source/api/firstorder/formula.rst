.. _api-firstorder-firstorder:

.. |blank| unicode:: U+0020

*Generic*

********************
First-order Formulas
********************

The following class diagram summarizes the inheritance hierarchy. Next, we are
going to describe the occurring classes starting at the top.

.. topic:: |blank|

  .. graphviz::
    :class: only-light

     digraph foo {
        rankdir="RL";
        bgcolor="transparent";
        edge [arrowhead=empty, arrowsize=0.75, penwidth=0.8];
        node [shape=box, fixedsize=true, width=1.6, height=0.3,
              fontsize="10pt", fontname="monospace", penwidth=0.8];
        Ex, All -> QuantifiedFormula;
        Equivalent, Implies, And, Or, Not, _T, _F -> BooleanFormula;
        "RCF.AtomicFormula", "Sets.AtomicFormula" -> AtomicFormula;
        QuantifiedFormula, BooleanFormula, AtomicFormula -> Formula;
        dots [shape=plaintext, height=0.1, label="..."];
        dots -> AtomicFormula;
     }

  .. graphviz::
    :class: only-dark

     digraph foo {
        rankdir="RL";
        bgcolor="transparent";
        edge [arrowhead=empty, arrowsize=0.75, penwidth=0.8, color=white];
        node [shape=box, fixedsize=true, width=1.6, height=0.3,
              fontsize="10pt", fontname="monospace", penwidth=0.8,
              color=white, fontcolor=white];
        Ex, All -> QuantifiedFormula;
        Equivalent, Implies, And, Or, Not, _T, _F -> BooleanFormula;
        "RCF.AtomicFormula", "Sets.AtomicFormula" -> AtomicFormula;
        QuantifiedFormula, BooleanFormula, AtomicFormula -> Formula;
        dots [shape=plaintext, height=0.1, label="..."];
        dots -> AtomicFormula;
     }


.. automodule:: logic1.firstorder.formula

  Generic Types
  *************

  .. data:: logic1.firstorder.formula.α
    :value: TypeVar('α', bound='AtomicFormula')

    A type variable denoting a type of atomic formulas with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: logic1.firstorder.formula.τ
    :value: TypeVar('τ', bound='AtomicFormula')

    A type variable denoting a type of terms with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: logic1.firstorder.formula.χ
    :value: TypeVar('χ', bound='AtomicFormula')

    A type variable denoting a type of variables with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.


  Formula Base Class
  ******************

  .. autoclass:: Formula
    :members:
    :undoc-members:
    :exclude-members: op, args, __init__
    :private-members: _repr_latex_
    :special-members: __le__, __str__

    .. autoproperty:: op

    .. autoproperty:: args

      The properties :attr:`op` and :attr:`args` are useful for decomposing and
      reconstructing formulas, using the invariant :code:`f = f.op(*f.args)`,
      which holds for all formulas :code:`f`. This approach has been adopted from
      the `SymPy <https://www.sympy.org/>`_ project:

      .. doctest::

        >>> from logic1.firstorder import *
        >>> f = And(Implies(F, T), Or(T, Not(T)))
        >>> # The class of f:
        >>> f.op
        <class 'logic1.firstorder.boolean.And'>
        >>> # The argument tuple of f:
        >>> f.args
        (Implies(F, T), Or(T, Not(T)))
        >>> # The invariant:
        >>> f == f.op(*f.args)
        True
        >>> # Construction of a new formula using components of f:
        >>> f.op(Equivalent(T, T), *f.args)
        And(Equivalent(T, T), Implies(F, T), Or(T, Not(T)))

    .. automethod:: __init__

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

  .. autoclass:: Or
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

  .. autoclass:: Not
    :members:
    :undoc-members:
    :exclude-members: __init__

  .. autofunction:: involutive_not

  .. autoclass:: _T
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

  .. autodata:: T
    :annotation: = _T()

  .. autoclass:: _F
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

  .. autodata:: F
    :annotation: = _F()


Quantified Formulas
*******************

.. automodule:: logic1.firstorder.quantified

  .. autoclass:: QuantifiedFormula
    :members:
    :undoc-members:
    :exclude-members: __init__, __new__

  .. autoclass:: Ex
    :members:
    :special-members:

  .. autoclass:: All
    :members:
    :special-members:

  .. autoclass:: Prefix
    :members:
    :undoc-members:
    :exclude-members: __init__
