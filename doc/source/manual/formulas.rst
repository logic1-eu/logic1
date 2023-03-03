.. _formulas-index:

.. |logic1| replace:: Logic1
.. |Card| replace:: :math:`n \in \mathbb{N} \cup \{\infty\}`

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

.. index::
  SymPy

Terms
=====
|logic1| supports `SymPy <https://www.sympy.org/>`_ expressions as terms with
SymPy symbols as variables. Here is an example:

>>> import sympy
>>> x, y = sympy.symbols('x, y')
>>> term = x**2 - 2*x*y + y**2

.. seealso::
  We refer to two short sections in the  `SymPy documentation
  <https://docs.sympy.org/>`_ for further details: Get started with the section
  on `Symbols
  <https://docs.sympy.org/latest/tutorials/intro-tutorial/gotchas.html#symbols>`_,
  and maybe also have a look at `Basic Operations
  <https://docs.sympy.org/latest/tutorials/intro-tutorial/basic_operations.html>`_.


Atomic Formulas
===============
|logic1| does *not* use any relations from SymPy but implements its own
relations, along with corresponding atomic formulas. The module
:py:mod:`logic1.atomlib.sympy` provides a library of atomic formulas based on
SymPy terms.

.. py:currentmodule:: logic1.atomlib.sympy

Equations and Inequalities
--------------------------

+--------------+-------+---------------+----------------+---------------+
|relation      | arity | in words      | class          | interactive   |
+==============+=======+===============+================+===============+
| :math:`=`    | 2     | equal         | :py:class:`Eq` | :py:func:`EQ` |
+--------------+-------+---------------+----------------+---------------+
| :math:`\neq` | 2     | not equal     | :py:class:`Ne` | :py:func:`NE` |
+--------------+-------+---------------+----------------+---------------+
| :math:`\geq` | 2     | greater-equal | :py:class:`Ge` | :py:func:`GE` |
+--------------+-------+---------------+----------------+---------------+
| :math:`\leq` | 2     | less-equal    | :py:class:`Le` | :py:func:`LE` |
+--------------+-------+---------------+----------------+---------------+
| :math:`>`    | 2     | greater than  | :py:class:`Gt` | :py:func:`GT` |
+--------------+-------+---------------+----------------+---------------+
| :math:`<`    | 2     | less than     | :py:class:`Lt` | :py:func:`LT` |
+--------------+-------+---------------+----------------+---------------+

Equations and inequalities as atomic formulas  are instances of the classes
given in the table. Let us have a closer look at :py:class:`Lt`. The other
classes are analogous:

.. py:class:: logic1.atomlib.sympy.Lt

  Bases: :py:class:`logic1.atomlib.sympy.BinaryAtomicFormula`

  A class for atomic formulas with relation :math:`<`.

  # Class data attributes:

  .. py:attribute:: func
    :canonical: logic1.atomlib.sympy.func
    :value: Lt

    The relation symbol of the atomic formula.

  .. py:attribute:: complement_func
    :canonical: logic1.atomlib.sympy.complement_func
    :value: Ge

    The complement relation symbol :py:attr:`func`'.

  .. py:attribute:: converse_func
    :canonical: logic1.atomlib.sympy.converse_func
    :value: Gt

    The converse relation symbol :py:attr:`func`:math:`^{-1}`.

  .. py:attribute:: dual_func
    :canonical: logic1.atomlib.sympy.dual_func
    :value: Le

    The dual relation symbol :math:`(`:py:attr:`func`':math:`)^{-1}`.

  # Instance data attributes:

  .. py:attribute:: args
    :type: tuple

    The tuple of arguments of the constructor :py:attr:`func`.

  .. py:property:: lhs
    :type: sympy.core.expr.Expr

    The left hand side term of the atomic formula, which is ``args[0]``.

  .. py:property:: rhs
    :type: sympy.core.expr.Expr

    The right hand side term of the atomic formula, which is ``args[1]``.

  # Class methods:

  .. py:classmethod:: interactive_new(cls, *args)

    Construct instances of :py:class:`Lt` dynamically checking `*args` and
    raising TypeError with an informative message for invalid arguments.

.. py:function:: logic1.atomlib.sympy.LT(*args) -> Lt

  Convenient access to Lt.interactive_new(*args)

The attributes :py:attr:`func`, :py:attr:`lhs`, and :py:attr:`rhs` allow to
access the components of a given atomic formula.

.. doctest::

  >>> from logic1.atomlib.sympy import *
  >>> atom = LT(1, 0)
  >>> atom
  Lt(1, 0)
  >>> atom.func, atom.lhs, atom.rhs
  (<class 'logic1.atomlib.sympy.Lt'>, 1, 0)

Since :py:attr:`func` is the class :py:class:`Lt` itself, it is callable, and we
have everything together to decompose and contruct atomic formulas:

.. doctest::

  >>> new_atom = atom.func(atom.lhs, atom.rhs)
  >>> new_atom
  Gt(1, 0)
  >>> new_atom == atom
  True
  >>> new_atom is atom
  False

More abstractly, one can use the argument tuple :py:attr:`args` instead of of
:py:attr:`lhs` and :py:attr:`rhs`:

.. admonition:: Important

  If ``f`` is any of our atomic formulas, then ``f == f.func(*f.args)``.

The attributes :py:attr:`func` and :py:attr:`args` are available throughout
|logic1| also for non-atomic formulas. This systematic has been adopted from
SymPy, which explains the generic attribute name :py:attr:`func` in the context
of relations.

.. seealso::

  The SymPy documentiation on `Recursing through an Expression Tree
  <https://docs.sympy.org/latest/tutorials/intro-tutorial/manipulation.html#recursing-through-an-expression-tree>`_.

Note that the regular constructor :py:meth:`Lt.__new__` of :py:class:`Lt` does
not check the validity of its arguments. This is desirable for the production of
efficient software based on |logic1|.

In contrast, the interactive constructor :py:meth:`Lt.interactive_new` of
:py:class:`Lt` does check its arguments and raises informative errors when
problems are detected. It is conveniently available es :py:func:`LT` and
should be prferred interactively or for code at a scripting level:

.. doctest::

  >>> bad = Lt('1', 0)  # slips through unnoticed
  >>> oops = LT('1', 0)
  Traceback (most recent call last):
  ...
  TypeError: '1' is not a Term

It remains to clarify the values of :py:attr:`complement_func`,
:py:attr:`converse_func`, and :py:attr:`dual_func` in general.

.. admonition:: Read about the mathematical background
  :class: dropdown

  Let :math:`\varrho \subseteq A^n` be an  :math:`n`-ary relation. Then
  the *complement relation* is defined as

  .. math::
    \varrho' = A^n \setminus \varrho.

  It follows that :math:`\varrho'(a_1, \dots, a_n)` is equivalent to :math:`\lnot
  \varrho(a_1, \dots, a_n)`, which is an important property for |logic1|. If
  :math:`\varrho` is binary, then the *converse relation* is defined as

  .. math::
    \varrho^{-1} = \{\,(y, x) \in A^2 \mid (x, y) \in \varrho\,\}.

  In other words, it swaps sides. The converse is the inverse with respect to
  composition, i.e., :math:`\varrho \circ \varrho^{-1} = \varrho^{-1}
  \circ \varrho = \Delta_A`, where the *diagonal* :math:`\Delta_A` is equality
  on :math:`A`. Finally, the *dual relation* is defined as

  .. math::
    \varrho^d = (\varrho')^{-1},

  which generally equals :math:`(\varrho^{-1})'`. For our relations here,
  dualization amounts to turning strict relations into weak relations, and vice
  versa.

  Each of these transformations of relations is involutive in the sense that
  :math:`(\varrho')' = (\varrho^{-1})^{-1} = (\varrho^d)^d = \varrho`.

+-----------------+----------------------------+--------------------------+----------------------+
| :py:attr:`func` | :py:attr:`complement_func` | :py:attr:`converse_func` | :py:attr:`dual_func` |
+=================+============================+==========================+======================+
| :py:class:`Eq`  | :py:class:`Ne`             | :py:class:`Eq`           | :py:class:`Ne`       |
+-----------------+----------------------------+--------------------------+----------------------+
| :py:class:`Ne`  | :py:class:`Eq`             | :py:class:`Ne`           | :py:class:`Eq`       |
+-----------------+----------------------------+--------------------------+----------------------+
| :py:class:`Ge`  | :py:class:`Lt`             | :py:class:`Le`           | :py:class:`Gt`       |
+-----------------+----------------------------+--------------------------+----------------------+
| :py:class:`Le`  | :py:class:`Gt`             | :py:class:`Ge`           | :py:class:`Lt`       |
+-----------------+----------------------------+--------------------------+----------------------+
| :py:class:`Gt`  | :py:class:`Le`             | :py:class:`Lt`           | :py:class:`Ge`       |
+-----------------+----------------------------+--------------------------+----------------------+
| :py:class:`Lt`  | :py:class:`Ge`             | :py:class:`Gt`           | :py:class:`Le`       |
+-----------------+----------------------------+--------------------------+----------------------+

Of course, :py:attr:`complement_func`, :py:attr:`converse_func`, and
:py:attr:`dual_func` work as constructors in the same way as :py:attr:`func`:

.. doctest::

  >>> atom
  Lt(1, 0)
  >>> atom.dual_func(atom.lhs, atom.rhs)
  Le(1, 0)

Cardinality Constraints
-----------------------
Next, we introduce infinitely many relation symbols of arity zero. The absence
of argument terms does not imply that those relations are semantically
equivalent to true or false. They can still make a statement about the domain of
the relation. For :math:`n \in \mathbb{N} \cup \{\infty\}` we have constant
relation symbols :math:`C_n` and :math:`\overline{C}_n`.

.. table::
  :widths: 10 10 12 10 10 10

  +------------------------+---------+---------------------------------+--------------------+------------------+---------+
  | relation               | m-arity | in words                        | class              | interactive      | p-arity |
  +========================+=========+=================================+====================+==================+=========+
  | :math:`C_n`            | 0       | cardinality at least :math:`n`  | :py:class:`_C`     | :py:func:`C`     | 1       |
  +------------------------+---------+---------------------------------+--------------------+------------------+---------+
  | :math:`\overline{C}_n` | 0       | cardinality less than :math:`n` | :py:class:`_C_bar` | :py:func:`C_bar` | 1       |
  +------------------------+---------+---------------------------------+--------------------+------------------+---------+

.. py:class:: logic1.atomlib.sympy._C

  Bases: :py:class:`logic1.atomlib.sympy.Cardinality`

  A class for atomic formulas with relation :math:`C_n` for :math:`n \in
  \mathbb{N} \cup \{\infty\}`.

  # Class data attributes:

  .. py:attribute:: func
    :canonical: logic1.atomlib.sympy.func
    :value: _C

    The relation symbol of the atomic formula.

  .. py:attribute:: complement_func
    :canonical: logic1.atomlib.sympy.complement_func
    :value: _C_bar

    The complement relation symbol :py:attr:`func`'.

  .. py:attribute:: args = ()
    :type: tuple

    The tuple of arguments of the constructor :py:attr:`func`.

  # Instance data attributes:

  .. py:attribute:: n
    :type: int

    The index :math:`n \in \mathbb{N} \cup \{\infty\}`.

  # Class methods:

  .. py:classmethod:: interactive_new(cls, *args)

    Construct instances of :py:class:`_C` dynamically checking `*args` and
    raising TypeError with an informative message for invalid arguments.

.. py:function:: logic1.atomlib.sympy.C(n) -> _C

  Convenient access to _C.interactive_new(n)


The idea is that the relation :math:`C_n` holds in its domain :math:`A` if and
only if :math:`|A| \geq n`. To be precise, :math:`\infty` represents the
cardinal :math:`\aleph_0`. Furthermore, :math:`\overline{C}_n` denotes the
complement relation :math:`C_n'` of :math:`C_n`. For instance, :math:`C_{12}`
holds in the field
:math:`\mathbb{R}` but not in the field :math:`\mathbb{Z}/3`.

First-order Formulas
====================

the following *logical* operators:

.. table::

  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | logical operator             | m-arity | in words   | class                  | interactive      | p-arity | Python operator |
  +==============================+=========+============+========================+==================+=========+=================+
  | :math:`\bot`                 | 0       | false      | :py:class:`_F`         | :py:obj:`F`      | 0       |                 |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\top`                 | 0       | true       | :py:class:`_T`         | :py:obj:`T`      | 0       |                 |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\lnot`                | 1       | not        | :py:class:`Not`        | :py:func:`NOT`   | 1       | :py:obj:`~~`    |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\land`                | \*      | and        | :py:class:`And`        | :py:func:`AND`   | \*      | :py:obj:`&`     |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\lor`                 | \*      | or         | :py:class:`Or`         | :py:func:`OR`    | \*      | :py:obj:`|`     |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\longrightarrow`      | 2       | implies    | :py:class:`Implies`    | :py:func:`IMPL`  | 2       | :py:obj:`>>`    |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\longleftrightarrow`  | 2       | equivalent | :py:class:`Equivalent` | :py:func:`EQUIV` | 2       |                 |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\exists x`            | 1       | exists     | :py:class:`Ex`         | :py:func:`EX`    | 2       |                 |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+
  | :math:`\forall x`            | 1       | for all    | :py:class:`All`        | :py:func:`ALL`   | 2       |                 |
  +------------------------------+---------+------------+------------------------+------------------+---------+-----------------+

Basic Operations on Formulas
============================

Data Extraction
---------------

.. autofunction:: logic1.firstorder.formula.Formula.count_alternations
  :noindex:

.. autofunction:: logic1.firstorder.formula.Formula.get_vars
  :noindex:

.. autofunction:: logic1.firstorder.formula.Formula.get_qvars
  :noindex:

.. index::
  Normal Forms

Transformations
---------------

.. index::
  substitution

.. autofunction:: logic1.firstorder.formula.Formula.subs
  :noindex:

.. index::
  simplification

.. autofunction:: logic1.firstorder.formula.Formula.simplify
  :noindex:

.. index::
  LaTeX; connversion to

.. autofunction:: logic1.firstorder.formula.Formula.to_latex
  :noindex:

.. autofunction:: logic1.firstorder.formula.Formula.to_sympy
  :noindex:

.. autofunction:: logic1.firstorder.formula.Formula.transform_atoms
  :noindex:

This often allows to write uniform code
for objects where the number or types of elements of :py:attr:`args` are not
known, in particular in recursions.

.. autofunction:: logic1.firstorder.formula.Formula.to_distinct_vars
  :noindex:


Normal Forms
============

.. index::
  negation normal form
  NNF

Negation Normal Form
--------------------

.. autofunction:: logic1.firstorder.formula.Formula.to_nnf
  :noindex:

.. index::
  prenex normal form
  PNF

Prenex Normal Form
------------------
.. autofunction:: logic1.firstorder.formula.Formula.to_pnf
   :noindex:


.. index::
  conjunctive normal form
  CNF
  disjunctive normal form
  DNF

Conjunctive and Disjunctive Normal Form
---------------------------------------
.. autofunction:: logic1.firstorder.boolean.BooleanFormula.to_cnf
   :noindex:
.. autofunction:: logic1.firstorder.boolean.BooleanFormula.to_dnf
   :noindex:

Remarks
=======
|logic1| focuses on *interpreted* first-order logic, where the specified
functions and relations mentioned above have a fixed semantics, which is not
explicitly expressed in the logical framework.
