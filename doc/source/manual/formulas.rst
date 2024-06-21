.. _formulas-index:

.. |logic1| replace:: Logic1
.. |Card| replace:: :math:`n \in \mathbb{N} \cup \{\infty\}`

.. default-domain:: py

##############################
First-order Logic and Formulas
##############################

.. index::
  variable
  term
  formula
  constant

.. index::
  atomic formula
  relation


First-order logic recursively builds *terms* from *variables* and a specified
set of *function* symbols with specified arities, which includes *constant*
symbols with arity zero. Next, *atomic formulas* are built from terms and a
specified set of *relation* symbols with specified arities. Finally,
*first-order formulas* formulas are recursively built from atomic formulas and a
fixed set of *logical operators*.

|logic1| focuses on *interpreted* first-order logic, where the above-mentioned
function and relation symbols have an implicit semantics, which is not
explicitly expressed via axioms within the logical framework.


.. index::
  SymPy

Terms
=====
|logic1| supports `SymPy <https://www.sympy.org/>`_ expressions as terms with
SymPy symbols as variables. Find more details in the `SymPy documentation
<https://docs.sympy.org/>`_. Get started with the
:external:ref:`intro-tutorial`.

SymPy supports several ways to define :external:ref:`symbols` and
:external:ref:`symbolic-expressions`. The most basic and flexible one is the use
of the function :external:func:`sympy.core.symbol.symbols`. We give an
example:

.. doctest::

  >>> import sympy
  >>> x, y = sympy.symbols('x, y')
  >>> isinstance(x, sympy.core.symbol.Symbol)
  True
  >>> x
  x
  >>> term = x**2 - 2*x*y + y**2
  >>> isinstance(term, sympy.core.expr.Expr)
  True
  >>> term
  x**2 - 2*x*y + y**2

Latin and Greek one-letter symbols can be conveniently imported from
:external:mod:`sympy.abc`:

.. doctest::

  >>> from sympy.abc import x, epsilon
  >>> (x + epsilon) ** 2
  (epsilon + x)**2

Another approach uses
:external:func:`sympy.core.sympify.sympify`. This function parses
strings into SymPy expressions. Symbols newly introduced this way are not
implicitly available as Python variables. However, one can generally
use :external:func:`sympy.core.sympify.sympify`, since symbols with
the same name are identical:

.. doctest::

  >>> import sympy
  >>> s = sympy.sympify
  >>> term = s('y + delta')
  >>> term
  delta + y
  >>> y
  Traceback (most recent call last):
  ...
  NameError: name 'y' is not defined
  >>> s('y')
  y
  >>> s('y') is term.args[1]
  True

Atomic Formulas
===============
|logic1| does not use any relations from SymPy but implements its own relations.
We distinguish between *relation symbols*, like :math:`\leq`, and *atomic
formulas* built from relation symbols with a suitable number of argument terms,
like :math:`x \leq x+1`.

|logic1| primarily aims at interpreted first-order logic. For many algorithms in
this setting the choice of admissible relation symbols, and also of admissible
function symbols, plays a crucial rule in the sense that slight variations of
those choices lead to entirely different algorithmic approaches for the same
problems. Examples for such algorithms are simplification, quantifier
elimination, and decision procedures. The choice of admissible function and
relation symbols takes place with the implementation of theories, which we
discuss later on. It is also theories that fix a domain of computation and give
function and relation symbol a semantics in that domain.

The module :mod:`logic1.atomlib.sympy` provides a library of classes for atomic
formulas with SymPy terms. Theories can select a set classes from this
collection, possibly modify via subclassing, and implement additional atomic
formulas on their own. In particular, subclassing allows to restrict the set of
admissible function symbols via argument checking in the constructors and
initializers of the derived classes.

.. currentmodule:: logic1.atomlib.sympy

Equations and Inequalities
--------------------------

Equations and inequalities are a popular kind of atomic formulas. They appear in
many interesting theories including Ordered Sets, Presburger Arithmetic, and
Real Closed Fields. The classes in the following table allow to realize
equations and inequalities as their instances.

.. table::
  :widths: 80 50 80 40

  +--------------+-------+---------------+-------------+
  |relation      | arity | in words      | class       |
  +==============+=======+===============+=============+
  | :math:`=`    | 2     | equal         | :class:`Eq` |
  +--------------+-------+---------------+-------------+
  | :math:`\neq` | 2     | not equal     | :class:`Ne` |
  +--------------+-------+---------------+-------------+
  | :math:`\geq` | 2     | greater-equal | :class:`Ge` |
  +--------------+-------+---------------+-------------+
  | :math:`\leq` | 2     | less-equal    | :class:`Le` |
  +--------------+-------+---------------+-------------+
  | :math:`>`    | 2     | greater than  | :class:`Gt` |
  +--------------+-------+---------------+-------------+
  | :math:`<`    | 2     | less than     | :class:`Lt` |
  +--------------+-------+---------------+-------------+

We have a closer look at the class :class:`Lt`. The other classes are similar.
The following snippet is not complete and it contains attributes that are
actually inherited from super classes. Compare the documentation of :class:`Lt`
in the API Reference.

.. class:: Lt
  :noindex:

  Bases: :class:`logic1.atomlib.sympy.BinaryAtomicFormula`

  A class for atomic formulas with relation :math:`<`.

  # Class data attributes:

  .. attribute:: op
    :noindex:
    :value: Lt

    The relation symbol of the atomic formula.

  .. attribute:: complement
    :noindex:
    :value: Ge

    The complement relation symbol of :attr:`op`.

  .. attribute:: converse
    :noindex:
    :value: Gt

    The converse relation symbol of :attr:`op`.

  .. attribute:: dual
    :noindex:
    :value: Le

    The dual relation symbol of :attr:`op`.

  # Instance data attributes:

  .. attribute:: args
    :noindex:
    :type: tuple

    The tuple of arguments of :attr:`op`, which equals ``(lhs, rhs)``.

  .. property:: lhs
    :noindex:
    :type: sympy.core.expr.Expr

    The left hand side term of the atomic formula, which is ``args[0]``.

  .. property:: rhs
    :noindex:
    :type: sympy.core.expr.Expr

    The right hand side term of the atomic formula, which is ``args[1]``.

The attributes :attr:`op`, :attr:`lhs`, and :attr:`rhs` give
access to the mathematical components of an atomic formula:

.. doctest::

  >>> from logic1.atomlib.sympy import *
  >>> atom = Lt(0, 1)
  >>> atom
  Lt(0, 1)
  >>> atom.op, atom.lhs, atom.rhs
  (<class 'logic1.atomlib.sympy.Lt'>, 1, 0)

Since :attr:`op` is the class :class:`Lt` itself, it is callable as a
constructor, and we have everything together to decompose and construct atomic
formulas:

.. doctest::

  >>> new_atom = atom.op(atom.lhs, atom.rhs)
  >>> new_atom
  Lt(0, 1)
  >>> new_atom == atom
  True
  >>> new_atom is atom
  False
  >> Gt(2, new_atom.rhs)
  Gt(2, 1)

More generally, one can use the argument tuple :attr:`args` instead of
:attr:`lhs` and :attr:`rhs`:

.. important::

  If ``f`` is any of our atomic formulas, then ``f == f.op(*f.args)``.

The same holds for SymPy expressions, which is discussed in the section
`Recursing through an Expression Tree
<https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html#recursing-through-an-expression-tree>`_
of the SymPy documentation. This explains our generic attribute name
:attr:`op` when actually referring to relations.
The attributes :attr:`op` and :attr:`args` will be available throughout
|logic1| also for non-atomic formulas.

The constructors and initializers of our classes here check the validity of
their arguments and raise :exc:`ValueError` with additional information when
problems are detected:

.. doctest::

  >>> oops = Lt('1', 0)
  Traceback (most recent call last):
  ...
  ValueError: '1' is not a Term

It remains to clarify the values of :attr:`complement`,
:attr:`converse`, and :attr:`dual` in general.

.. admonition:: Mathematical definitions

  Let :math:`\varrho \subseteq A^n` be an  :math:`n`-ary relation. Then
  the *complement relation* is defined as

  .. math::
    \overline{\varrho} = A^n \setminus \varrho.

  It follows that :math:`\overline{\varrho}(a_1, \dots, a_n)` is equivalent to
  :math:`\lnot
  \varrho(a_1, \dots, a_n)`, which is an important property for |logic1|.

  If :math:`\varrho` is binary, then the *converse relation* is defined as

  .. math::
    \varrho^{-1} = \{\,(y, x) \in A^2 \mid (x, y) \in \varrho\,\}.

  In other words, the converse swaps sides. It is the inverse with respect to
  composition, i.e., :math:`\varrho \circ \varrho^{-1} = \varrho^{-1}
  \circ \varrho = \Delta_A`. The diagonal :math:`\Delta_A = \{\,(x, y) \in A^2
  \mid x = y\,\}` is equality on :math:`A`.

  Finally, the *dual relation* is defined as

  .. math::
    \varrho^d = \overline{\varrho^{-1}},

  which generally equals :math:`(\overline{\varrho})^{-1}`. For our relations
  here, dualization amounts to turning strict relations into weak relations, and
  vice versa.

  Each of these transformations of relations is involutive in the sense that
  :math:`\overline{\overline{\varrho}} = (\varrho^{-1})^{-1} = (\varrho^d)^d =
  \varrho`.

.. table::

  +-------------+--------------------+------------------+--------------+
  | :attr:`op`  | :attr:`complement` | :attr:`converse` | :attr:`dual` |
  +=============+====================+==================+==============+
  | :class:`Eq` | :class:`Ne`        | :class:`Eq`      | :class:`Ne`  |
  +-------------+--------------------+------------------+--------------+
  | :class:`Ne` | :class:`Eq`        | :class:`Ne`      | :class:`Eq`  |
  +-------------+--------------------+------------------+--------------+
  | :class:`Ge` | :class:`Lt`        | :class:`Le`      | :class:`Gt`  |
  +-------------+--------------------+------------------+--------------+
  | :class:`Le` | :class:`Gt`        | :class:`Ge`      | :class:`Lt`  |
  +-------------+--------------------+------------------+--------------+
  | :class:`Gt` | :class:`Le`        | :class:`Lt`      | :class:`Ge`  |
  +-------------+--------------------+------------------+--------------+
  | :class:`Lt` | :class:`Ge`        | :class:`Gt`      | :class:`Le`  |
  +-------------+--------------------+------------------+--------------+

Of course, :attr:`complement`, :attr:`converse`, and
:attr:`dual` work as constructors in the same way as :attr:`op`:

.. doctest::

  >>> atom
  Lt(1, 0)
  >>> atom.dual(atom.lhs, atom.rhs)
  Le(1, 0)

.. note::

  Recall that we are working at a syntactic level here, where domains and
  semantics of the relations are not specified yet. The concrete specifications
  of complement, converse, and dual relation symbols aims to have this set of
  relation symbols used in theories that correspond to it.

  As a counterexample consider a theory of partially ordered sets, in which all
  of our relation symbols are meaningful. However, with a partial order,
  :math:`\geq` is not the complement relation of :math:`<`. Consequently, such a
  theory would find another suitable set of relations somewhere, or implement it
  itself.


Cardinality Constraints
-----------------------
Cardinality constraints are constant atomic formulas, built from relation
symbols with arity 0. A cardinality constraint holds in a given theory if the
cardinality of the corresponding domain satisfies the requirements imposed by
the constraint. For instance, a constraint :math:`C_2` can be used to state that
the domain has at least two elements. It holds in the field
:math:`\mathbb{R}` but not in the field :math:`\mathbb{Z}/3`. A typical use case
is the theory of :ref:`sets-index`, which requires infinitely many constant
relation symbols :math:`C_n` for :math:`n \in \mathbb{N}` in order to admit
quantifier elimination.

The classes in the following table allow to realize cardinality constraints as
their instances.

.. table::
  :widths: 5 5 10 4 5

  +------------------------+---------+---------------------------------+-------------+---------+
  | relation               | m-arity | in words                        | class       | p-arity |
  +========================+=========+=================================+=============+=========+
  | :math:`C_n`            | 0       | cardinality at least :math:`n`  | :class:`C`  | 1       |
  +------------------------+---------+---------------------------------+-------------+---------+
  | :math:`\overline{C}_n` | 0       | cardinality less than :math:`n` | :class:`C_` | 1       |
  +------------------------+---------+---------------------------------+-------------+---------+

.. class:: C
  :noindex:

  Bases: :class:`logic1.atomlib.sympy.Cardinality`

  A class for atomic formulas with relation :math:`C_n` for :math:`n \in
  \mathbb{N} \cup \{\infty\}`.

  # Class data attributes:

  .. attribute:: op
    :noindex:
    :value: C

    The relation symbol of the atomic formula.

  .. attribute:: complement
    :noindex:
    :value: C_

    The complement relation symbol of :attr:`op`.

  # Instance data attributes:

  .. attribute:: args
    :noindex:
    :type: tuple

    The tuple of arguments of the constructor :attr:`op`, which equals ``(index,)``.

  .. property:: index
    :noindex:
    :type: int | sympy.core.numbers.Infinity

    The index :math:`n` of :math:C_n`, which is :attr:`args [0]`.

Infinity :math:`\infty` denotes the cardinal :math:`\aleph_0`. It is available
as

.. data:: oo
  :noindex:
  :value: sympy.oo

Note that we are dealing with two different notions of arity here:

m-arity:
  From a mathematical viewpoint we have two families :math:`\{C_n\}_{n \in
  \mathbb{N} \cup \{\infty\}}` and :math:`\{\overline{C}_n\}_{n \in \mathbb{N}
  \cup \{\infty\}}` of relations, where each relation has aritiy 0.

p-arity:
  From a Python viewpoint we have two classes :class:`C` and :class:`C_`.
  Each class has an instance data attribute :data:`n`, which is a
  non-negative integer or :data:`oo`. There is a corresponding constructor of
  arity 1.

We will encounter a similar situation with quantifiers in
the next section on :ref:`first-order-formulas`, where
both the m-arity and the p-arity will be positive.

Cardinality Constraints of the same class and with the same index are identical:

.. doctest::

    >>> from logic1.atomlib.sympy import C, oo
    >>> a1 = C(0)
    >>> a2 = C(0)
    >>> a3 = C(oo)
    >>> a1 is a2
    True
    >>> a1 == a3
    False

.. _first-order-formulas:

First-order Formulas
====================

the following *logical* operators:

.. table::
  :widths: 5 5 6 6 5 5

  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | operator                    | m-arity | in words   | class               | p-arity | short     |
  +=============================+=========+============+=====================+=========+===========+
  | :math:`\bot`                | 0       | false      | :class:`_F`         | 0       | :obj:`F`  |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\top`                | 0       | true       | :class:`_T`         | 0       | :obj:`T`  |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\lnot`               | 1       | not        | :class:`Not`        | 1       | :obj:`~~` |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\land`               | \*      | and        | :class:`And`        | \*      | :obj:`&`  |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\lor`                | \*      | or         | :class:`Or`         | \*      | :obj:`|`  |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\longrightarrow`     | 2       | implies    | :class:`Implies`    | 2       | :obj:`>>` |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\longleftrightarrow` | 2       | equivalent | :class:`Equivalent` | 2       |           |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\exists x`           | 1       | exists     | :class:`Ex`         | 2       |           |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
  | :math:`\forall x`           | 1       | for all    | :class:`All`        | 2       |           |
  +-----------------------------+---------+------------+---------------------+---------+-----------+
