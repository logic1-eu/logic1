.. _api-firstorder-atomic:

*Generic*

***********************
Variables, Terms, Atoms
***********************

.. automodule:: logic1.firstorder.atomic

  Generic Types
  *************

  We use type variables :data:`.atomic.α`, :data:`.atomic.τ`, and
  :data:`.atomic.χ` with the same names and definitions as in module
  :mod:`.formula`.

  .. data:: α
    :value: TypeVar('α', bound='AtomicFormula')
    :canonical: logic1.firstorder.atomic.α

    A type variable denoting a type of atomic formulas with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: τ
    :value: TypeVar('τ', bound='AtomicFormula')
    :canonical: logic1.firstorder.atomic.τ

    A type variable denoting a type of terms with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.

  .. data:: χ
    :value: TypeVar('χ', bound='AtomicFormula')
    :canonical: logic1.firstorder.atomic.χ

    A type variable denoting a type of variables with upper bound
    :class:`logic1.firstorder.atomic.AtomicFormula`.


  Set of Variables
  ********************

  .. autoclass:: VariableSet
    :special-members:

    .. automethod:: __getitem__

    .. automethod:: get

    .. automethod:: imp

    .. method::
      pop() -> None
      push() -> None
      :abstractmethod:

      :meth:`.push` pushes the current status regarding used variables to a
      private stack and resets to the initial state where all variables are
      marked as unused. :meth:`.pop` recovers the status from the stack and
      thereby overwrites the current status.

      .. attention::
        :attr:`.stack`, :meth:`.push`, and :meth:`pop` are reserved for special
        situations.

        They allow to obtain variables from :meth:`fresh()
        <.firstorder.atomic.VariableSet.fresh>` within asychronous doctests in a
        reproducable way. In the following example,
        :meth:`.Formula.to_pnf` indirectly uses
        :meth:`.RCF.atomic.VariableSet.fresh`:

        >>> from logic1.firstorder import *
        >>> from logic1.theories.RCF import *
        >>> x, a = VV.get('x', 'a')
        >>> f = And(x == 0, Ex(x, x == a))
        >>> f.to_pnf()
        Ex(G0001_x, And(x == 0, -G0001_x + a == 0))
        >>> f.to_pnf()
        Ex(G0002_x, And(x == 0, -G0002_x + a == 0))
        >>> VV.push()
        >>> x, a = VV.get('x', 'a')
        >>> g = And(x == 0, Ex(x, x == a))
        >>> g.to_pnf()
        Ex(G0001_x, And(x == 0, -G0001_x + a == 0))
        >>> VV.pop()
        >>> f.to_pnf()
        Ex(G0003_x, And(x == 0, -G0003_x + a == 0))

        Notice that we are not using any previously existing variables
        between ``VV.push()`` and ``VV.pop()`` above.

        See :file:`logic1/theories/RCF/test_pnf.txt` for a complete doctest
        file using this approach

  Terms
  *****

  .. autoclass:: Term
    :members: as_latex, sort_key, vars
    :special-members:


  Variables
  *********

  .. autoclass:: Variable
    :members:
    :undoc-members:

  Atomic Formulas
  ***************

  .. autoclass:: AtomicFormula
    :members:
    :undoc-members:
    :exclude-members: atoms
    :special-members: __le__, __str__
