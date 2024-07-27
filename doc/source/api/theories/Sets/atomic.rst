.. _api-Sets-atomic:

*Sets*

***********************
Variables and Atoms
***********************

.. automodule:: logic1.theories.Sets.atomic

  .. autoclass:: VariableSet
    :special-members:

    .. automethod:: __getitem__

    .. automethod:: fresh

    .. method::
      pop() -> None
      push() -> None
      :abstractmethod:

      Implement abstract methods
      :meth:`.logic1.firstorder.atomic.VariableSet.pop` and
      :meth:`.logic1.firstorder.atomic.VariableSet.push`.


  .. data:: VV
    :value: VariableSet()

    The unique instance of :class:`.VariableSet`.

  .. autoclass:: Variable
    :special-members:

    .. method:: ==, !=
                __eq__(other: Variable) -> Eq
                __ne__(other: Variable) -> Ne

      Construction of instances of :class:`Eq` and :class:`Ne` is available via
      overloaded operators.

    .. automethod:: as_latex

    .. automethod:: fresh

    .. automethod:: sort_key

    .. automethod:: subs

    .. automethod:: vars


  .. autoclass:: AtomicFormula
    :special-members:

    .. automethod:: __le__

    .. automethod:: __str__

    .. automethod:: as_latex

    .. automethod:: bvars

    .. automethod:: complement

    .. automethod:: fvars

    .. automethod:: simplify

    .. automethod:: subs


  .. class:: Eq
             Ne

    Bases: :class:`.AtomicFormula`

    Equations and inequalities between variables.

    .. property:: lhs
                  rhs
      :type: Variable

      The left hand side variable and the right hand side variable of an
      equation or inequation, respectively.


  .. autodata:: oo

  .. autodata:: Index


  .. class:: C
             C_

    Cardinality constraints. From a mathematical perspective, the instances are
    constant relation symbols with an index, which is either a positive integer
    or the float `inf`, represented as ``oo``. ``C(n)`` holds iff there are at
    least ``n`` different elements in the universe. This is not a statement
    about the index ``n`` but about a range of models where this constant
    relation holds.

    In the following example, ``f`` states that there should be at least 2
    elements but not 3 elements or more:

    >>> from logic1.firstorder import *
    >>> from logic1.theories.Sets import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> f = Ex([x, y], x != y) & All([x, y, z], Or(x == y, y == z, z == x))
    >>> qe(f)  # quantifier elimination:
    And(C(2), C_(3))

    The class :class:`C_` is dual to :class:`C`; more precisely, for every index
    ``n``, we have that ``C_(n)`` is the dual relation of ``C(n)``, and vice
    versa.

    The class constructors take care that instances with equal indices are
    identical:

    >>> C(1) is C(1)
    True
    >>> C(1) == C(2)
    False

    .. property:: index
      :type: Index

      The index of the constant relation symbol