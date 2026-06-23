.. _api-Complex-ast:

*Complex*

*********************
Abstract Syntax Trees
*********************

.. automodule:: logic1.theories.Complex.ast

  Abstract AST Nodes, SortKey
  ******************************

  .. autoclass:: AST
    :special-members:

    .. autoproperty:: op

    .. autoproperty:: args

    .. automethod:: __init__

    .. method:: +, ~, *, -, **, /
                __add__(other: Number | AST) -> Add
                __invert__() -> Conj
                __mul__(other: Number | AST) -> Mul
                __neg__() -> Neg
                __pow__(other: int) -> Pow
                __radd__(other: Number | AST) -> Add
                __rmul__(other: Number | AST) -> Mul
                __rsub__(other: Number | AST) -> Add
                __sub__(other: Number | AST) -> Add
                __truediv__(self, other: Number | AST) -> AST

      Arithmetic operations on AST nodes are available as overloaded operators.

    .. method:: ==, >=, >, <=, <, !=
                __eq__(other: object) -> bool
                __ge__(other: AST) -> bool
                __gt__(other: AST) -> bool
                __le__(other: AST) -> bool
                __lt__(other: AST) -> bool
                __ne__(other: object) -> bool

      Comparison of AST nodes via their :class:`SortKey`. See also :meth:`sort_key`.

    .. automethod:: accept
    .. automethod:: as_latex
    .. automethod:: lc
    .. automethod:: eval
    .. automethod:: factors
    .. automethod:: from_real_imag
    .. automethod:: from_number
    .. automethod:: is_constant
    .. automethod:: is_variable
    .. automethod:: is_zero
    .. automethod:: _repr_latex_
    .. automethod:: sort_key
    .. automethod:: subs

  .. autoclass:: SortKey
    :special-members:

  .. autoclass:: MonoidalOperation
    :special-members:

    .. automethod:: __init__

  .. autoclass:: UnaryOperation
    :special-members:


  Concrete AST Nodes
  ******************

  .. autoclass:: Rat
    :special-members:

    .. automethod:: __init__

  .. autoclass:: _I
    :special-members:

  .. autoclass:: Var
    :special-members:

  .. autoclass:: Add
    :special-members:

  .. autoclass:: Mul
    :special-members:

  .. autoclass:: Pow
    :special-members:

  .. autoclass:: Neg
    :special-members:

  .. autoclass:: Conj
    :special-members:

  .. autoclass:: Re
    :special-members:

  .. autoclass:: Im
    :special-members:

  AST Visitors
  ************

  .. autoclass:: ASTVisitor

  .. autoclass:: IdentityASTVisitor

  .. autoclass:: VariableSubstitutor


Printing
********

.. automodule:: logic1.theories.Complex.format