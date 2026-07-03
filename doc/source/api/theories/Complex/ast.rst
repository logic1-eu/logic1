.. _api-Complex-ast:

*Complex*

*********************
Abstract Syntax Trees
*********************


.. automodule:: logic1.theories.Complex.ast

  .. autodata:: α
  .. autodata:: η

  Base Classes
  ************

  .. autoclass:: AST
    :members:
    :special-members:
    :exclude-members: __lt__, __gt__, __ge__, __ne__, __hash__, __weakref__

  .. autoclass:: MonoidalOperation
    :members:
    :special-members:

  .. autoclass:: UnaryOperation
    :members:
    :special-members:


  AST Nodes
  *********

  .. autoclass:: Rat
    :members:
    :special-members:

  .. autoclass:: _I
    :members:
    :special-members:

  .. autodata:: I

  .. autoclass:: Var
    :members:
    :special-members:

  .. autoclass:: Add
    :members:
    :special-members:

  .. autoclass:: Mul
    :members:
    :special-members:

  .. autoclass:: Pow
    :members:
    :special-members:

  .. autoclass:: Neg
    :members:
    :special-members:

  .. autoclass:: Conj
    :members:
    :special-members:

  .. autoclass:: Re
    :members:
    :special-members:

  .. autoclass:: Im
    :members:
    :special-members:

  Sort Key
  ********

  .. autoclass:: SortKey
    :members:
    :special-members:
    :exclude-members: __init__, __lt__, __gt__, __ge__, __ne__, __hash__, __weakref__


  Visitors
  ********

  .. autoclass:: ASTVisitor
    :members:
    :special-members:
    :exclude-members: __weakref__

  .. autoclass:: IdentityASTVisitor
    :members:
    :special-members:

  .. autoclass:: VariableSubstitutor
    :members:
    :special-members:


Printing
********

.. automodule:: logic1.theories.Complex.format

  .. autoclass:: BaseFormatter
    :members:
    :special-members:

  .. autoclass:: ReprFormatter
    :members:
    :special-members:

  .. autoclass:: StrFormatter
    :members:
    :special-members:

  .. autoclass:: LatexFormatter
    :members:
    :special-members:

Normalization
*************

.. automodule:: logic1.theories.Complex.normalize

  .. autoclass:: ArithmeticEvaluator
    :members:
    :special-members:

  .. autoclass:: ConstantEvaluator
    :members:
    :special-members:

  .. autoclass:: WeakNormalizer
    :members:
    :special-members:

  .. autoclass:: AddSortKey
    :members:
    :special-members:
    :exclude-members: __lt__, __gt__, __ge__, __ne__, __hash__, __repr__, __weakref__

  .. autoclass:: MulSortKey
    :members:
    :special-members:
    :exclude-members: __lt__, __gt__, __ge__, __ne__, __hash__, __repr__, __weakref__

  .. autoclass:: Normalizer
    :members:
    :special-members:

  .. autoclass:: ComplexNormalizer
    :members:
    :special-members:

  .. autoclass:: RealNormalizer
    :members:
    :special-members:

  .. autofunction:: conjugate_normal_form

  .. autofunction:: cartesian_normal_form

SortKeys
********