.. _api-RCF-atomic:

*Real Closed Fields*

***********************
Variables, Terms, Atoms
***********************

There is an infinite set :data:`RCF.VV <.RCF.term.VV>` of all :mod:`.theories.RCF` variables, which is an instance of the class :class:`VariableSet <.RCF.term.VariableSet>`. The variables obtained from :data:`VV <.RCF.term.VV>` are instances of the class :class:`Variable <.RCF.term.Variable>`, which is a subclass of :class:`Term <.RCF.term.Term>`. Larger terms can be constructed from variables and numbers using ring arithmetic :meth:`+ <.RCF.term.Term.__add__>`, :meth:`* <.RCF.term.Term.__mul__>`, :meth:`- <.RCF.term.Term.__sub__>`, :meth:`** <.RCF.term.Term.__pow__>`, :meth:`/ <.RCF.term.Term.__truediv__>` implemented in :class:`Term <.RCF.term.Term>`.

There is a class :class:`AtomicFormula <.RCF.atomic.AtomicFormula>` with subclasses :class:`Eq <.RCF.atomic.Eq>`, :class:`Ne <.RCF.atomic.Ne>`, :class:`Le <.RCF.atomic.Le>`, :class:`Ge <.RCF.atomic.Ge>`, :class:`Lt <.RCF.atomic.Lt>`, :class:`Gt <.RCF.atomic.Gt>`. Atoms are obtained as instances of these subclasses either using the class names as constructors, or using the corresponding operators :meth:`== <.RCF.term.Term.__eq__>`, :meth:`\!= <.RCF.term.Term.__ne__>`, :meth:`<= <.RCF.term.Term.__le__>`, :meth:`>= <.RCF.term.Term.__ge__>`, :meth:`< <.RCF.term.Term.__lt__>` :meth:`> <.RCF.term.Term.__gt__>` overloaded in the class :class:`Term <.RCF.term.Term>`.

Some examples can be found on the :ref:`landing page <api-RCF>` of this section.

.. graphviz::
  :class: only-light

   digraph RCF_terms {
      graph [layout=neato, overlap=false, splines=line, sep=0.5, ranksep=0.5, nodesep=0.5];
      bgcolor="transparent";

      node [shape=box, fontsize="10pt",
            fontname="monospace", penwidth=0.8];
      edge [arrowsize=0.75, penwidth=0.8];

      VV [ shape=box, label=<<U>RCF.VV: VariableSet</U>>, pos="4.0,6.0!"];

      Variable [pos="4,5.0!"];
      int      [pos="5.5,5.0!"];
      mpq      [pos="7,5.0!"];
      Fraction [pos="8.5,5.0!"];
      float    [pos="10.0,5.0!"];

      Arithmetic [style=rounded, label="Term methods  +  -  *  **  /", pos="7,4.0!"];
      Term       [pos="7,3.0!"];

      Comparison [style=rounded, label="Term methods  ==  !=  <=  >=  <  >", pos="7.0,2.0!"];

      Eq [pos="4.25,1.0!"];
      Ne [pos="5.35,1.0!"];
      Le [pos="6.45,1.0!"];
      Ge [pos="7.55,1.0!"];
      Lt [pos="8.65,1.0!"];
      Gt [pos="9.75,1.0!"];

      AtomicFormula [pos="7,0.0!"];

      VV -> Variable
         [xlabel=" yields  ",
          style=dashed, arrowhead=normal,
          fontsize="10pt", fontname="sans-serif"];

      Variable -> Term
         [arrowhead=empty, xlabel="subclass of       ", fontsize="10pt", fontname="sans-serif"];

      Variable -> Arithmetic [style=dashed, arrowhead=none];
      int      -> Arithmetic [style=dashed, arrowhead=none];
      mpq      -> Arithmetic [style=dashed, arrowhead=none,
                              xlabel="enter  ", fontsize="10pt", fontname="sans-serif"];
      Fraction -> Arithmetic [style=dashed, arrowhead=none];
      float    -> Arithmetic [style=dashed, arrowhead=none];

      Arithmetic -> Term
         [xlabel=" construct  ",
          style=dashed, arrowhead=normal,
          fontsize="10pt", fontname="sans-serif"];

      Term -> Comparison
         [style=dashed, arrowhead=none,
          xlabel="enters  ", fontsize="10pt", fontname="sans-serif"];

      Comparison -> Eq [style=dashed, arrowhead=normal,
                        xlabel="construct    ", fontsize="10pt", fontname="sans-serif"];
      Comparison -> Ne [style=dashed, arrowhead=normal];
      Comparison -> Le [style=dashed, arrowhead=normal];
      Comparison -> Ge [style=dashed, arrowhead=normal];
      Comparison -> Lt [style=dashed, arrowhead=normal];
      Comparison -> Gt [style=dashed, arrowhead=normal];

      Eq -> AtomicFormula
        [arrowhead=empty, xlabel="subclass of             ", fontsize="10pt", fontname="sans-serif"];
      Ne, Le, Ge, Lt, Gt -> AtomicFormula [arrowhead=empty];
   }

.. graphviz::
   :class: only-dark

   digraph RCF_terms {
      graph [layout=neato, overlap=false, splines=line, sep=0.5, ranksep=0.5, nodesep=0.5];
      bgcolor="transparent";

      node [shape=box, fontsize="10pt", fontname="monospace", penwidth=0.8,
            color=white, fontcolor=white];
      edge [arrowsize=0.75, penwidth=0.8,
            color=white, fontcolor=white];

      VV [ shape=box, label=<<U>RCF.VV: VariableSet</U>>, pos="4.0,6.0!"];

      Variable [pos="4,5.0!"];
      int      [pos="5.5,5.0!"];
      mpq      [pos="7,5.0!"];
      Fraction [pos="8.5,5.0!"];
      float    [pos="10.0,5.0!"];

      Arithmetic [style=rounded, label="Term methods  +  -  *  **  /", pos="7,4.0!"];
      Term       [pos="7,3.0!"];

      Comparison [style=rounded, label="Term methods  ==  !=  <=  >=  <  >", pos="7.0,2.0!"];

      Eq [pos="4.25,1.0!"];
      Ne [pos="5.35,1.0!"];
      Le [pos="6.45,1.0!"];
      Ge [pos="7.55,1.0!"];
      Lt [pos="8.65,1.0!"];
      Gt [pos="9.75,1.0!"];

      AtomicFormula [pos="7,0.0!"];

      VV -> Variable
         [xlabel=" yields  ",
          style=dashed, arrowhead=normal,
          fontsize="10pt", fontname="sans-serif"];

      Variable -> Term
         [arrowhead=empty, xlabel="subclass of       ", fontsize="10pt", fontname="sans-serif"];

      Variable -> Arithmetic [style=dashed, arrowhead=none];
      int      -> Arithmetic [style=dashed, arrowhead=none];
      mpq      -> Arithmetic [style=dashed, arrowhead=none,
                              xlabel="enter  ", fontsize="10pt", fontname="sans-serif"];
      Fraction -> Arithmetic [style=dashed, arrowhead=none];
      float    -> Arithmetic [style=dashed, arrowhead=none];

      Arithmetic -> Term
         [xlabel=" construct  ",
          style=dashed, arrowhead=normal,
          fontsize="10pt", fontname="sans-serif"];

      Term -> Comparison
         [style=dashed, arrowhead=none,
          xlabel="enters  ", fontsize="10pt", fontname="sans-serif"];

      Comparison -> Eq [style=dashed, arrowhead=normal,
                        xlabel="construct    ", fontsize="10pt", fontname="sans-serif"];
      Comparison -> Ne [style=dashed, arrowhead=normal];
      Comparison -> Le [style=dashed, arrowhead=normal];
      Comparison -> Ge [style=dashed, arrowhead=normal];
      Comparison -> Lt [style=dashed, arrowhead=normal];
      Comparison -> Gt [style=dashed, arrowhead=normal];

      Eq -> AtomicFormula
        [arrowhead=empty, xlabel="subclass of             ", fontsize="10pt", fontname="sans-serif"];
      Ne, Le, Ge, Lt, Gt -> AtomicFormula [arrowhead=empty];
   }

.. automodule:: logic1.theories.RCF.term

  ========================
  The Set of All Variables
  ========================

  .. autoclass:: VariableSet
    :special-members:
    :exclude-members: __init__, __new__

    .. autoproperty:: stack

    .. automethod:: __getitem__

    .. automethod:: fresh

    .. method::
      pop() -> None
      push() -> None

      Implements the abstract methods
      :meth:`.logic1.firstorder.term.VariableSet.pop` and
      :meth:`push() <logic1.firstorder.term.VariableSet.push>`.


  .. data:: VV
    :value: VariableSet()

    The unique instance of :class:`.VariableSet`.

  ===================
  Terms and Variables
  ===================

  .. autoclass:: Term
    :special-members:
    :exclude-members: __init__, __new__

    .. method:: +, *, -, **, /
                __add__(other: object) -> Term
                __mul__(other: object) -> Term
                __neg__() -> Term
                __pow__(other: object) -> Term
                __radd__(other: object) -> Term
                __rmul__(other: object) -> Term
                __rsub__(other: object) -> Term
                __sub__(other: object) -> Term
                __truediv__(other: object) -> Term

      Arithmetic operations on Terms are available as these overloaded operators.

    .. method:: ==, >=, >, <=, <, !=
                __eq__(other: Term | int) -> logic1.theories.RCF.atomic.Eq
                __ge__(other: Term | int) -> logic1.theories.RCF.atomic.Ge | logic1.theories.RCF.atomic.Le
                __gt__(other: Term | int) -> logic1.theories.RCF.atomic.Gt | logic1.theories.RCF.atomic.Lt
                __le__(other: Term | int) -> logic1.theories.RCF.atomic.Ge | logic1.theories.RCF.atomic.Le
                __lt__(other: Term | int) -> logic1.theories.RCF.atomic.Gt | logic1.theories.RCF.atomic.Lt
                __ne__(other: Term | int) -> logic1.theories.RCF.atomic.Ne

      Construction of instances of :class:`Eq <.RCF.atomic.Eq>`, :class:`Ge <.RCF.atomic.Ge>`, :class:`Gt <.RCF.atomic.Gt>`, :class:`Le <.RCF.atomic.Le>`, :class:`Lt <.RCF.atomic.Lt>`, :class:`Ne <.RCF.atomic.Ne>` is available via these overloaded operators.

    .. automethod:: __init__(self, arg: float | int | fractions.Fraction | gmpy2.mpq | sage.rings.integer.Integer | sage.rings.rational.Rational | MPolynomial[sage.rings.rational.Rational] | UPolynomial) -> None

    .. automethod:: __iter__

    .. automethod:: as_constant

    .. automethod:: as_latex

    .. automethod:: as_variable

    .. automethod:: coefficient

    .. automethod:: constant_coefficient

    .. automethod:: content

    .. automethod:: degree

    .. automethod:: derivative

    .. automethod:: factor

    .. automethod:: is_constant

    .. automethod:: is_definite

    .. automethod:: is_monomial

    .. automethod:: is_variable

    .. automethod:: is_weakly_parametric_linear

    .. automethod:: is_zero

    .. automethod:: lc

    .. automethod:: monomial_coefficient

    .. automethod:: monomials

    .. automethod:: normalize

    .. automethod:: primitive_part

    .. automethod:: pseudo_quo_rem

    .. automethod:: quo_rem

    .. automethod:: reduce

    .. automethod:: sort_key

    .. automethod:: subs

    .. automethod:: subs_linear_solution

    .. automethod:: summands

    .. automethod:: vars


  .. autoclass:: Variable
    :exclude-members: __init__, __new__

    .. automethod:: fresh


  .. autodata:: logic1.theories.RCF.term.term_sage.τ

  =========================
  Support Classes for Terms
  =========================

  .. autoclass:: DEFINITE
    :members:


  .. autoclass:: SortKey
    :exclude-members: __init__, __new__

.. automodule:: logic1.theories.RCF.atomic

  =====
  Atoms
  =====

  .. autoclass:: AtomicFormula
    :exclude-members: __init__, __new__

    .. property:: lhs
                  rhs
      :type: logic1.theories.RCF.term.Term

      The left hand side term and the right hand side term of an atomic formula.

    .. automethod:: __bool__

    .. automethod:: __le__

    .. automethod:: __str__

    .. automethod:: as_latex

    .. automethod:: bvars

    .. method:: complement(cls) -> type[AtomicFormula]
                converse(cls) -> type[AtomicFormula]
                dual(cls) -> type[AtomicFormula]
      :classmethod:

      Complement relation, converse relation, and dual relation.
      :meth:`complement` implements the abstract method
      :meth:`.firstorder.atomic.AtomicFormula.complement`.

      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
      |                    | :class:`Eq` | :class:`Ne` | :class:`Le` | :class:`Ge` | :class:`Lt` | :class:`Gt` |
      +====================+=============+=============+=============+=============+=============+=============+
      | :meth:`complement` | :class:`Ne` | :class:`Eq` | :class:`Gt` | :class:`Lt` | :class:`Ge` | :class:`Le` |
      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
      | :meth:`converse`   | :class:`Eq` | :class:`Ne` | :class:`Ge` | :class:`Le` | :class:`Gt` | :class:`Lt` |
      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+
      | :meth:`dual`       | :class:`Ne` | :class:`Eq` | :class:`Lt` | :class:`Gt` | :class:`Le` | :class:`Ge` |
      +--------------------+-------------+-------------+-------------+-------------+-------------+-------------+

      .. admonition:: Mathematical definitions

        Let :math:`\varrho \subseteq A^n` be an  :math:`n`-ary relation. Then
        the *complement relation* is defined as

        .. math::
          \overline{\varrho} = A^n \setminus \varrho.

        It follows that :math:`\overline{\varrho}(a_1, \dots, a_n)` is equivalent to
        :math:`\lnot
        \varrho(a_1, \dots, a_n)`, which is an important property for Logic1.

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

      .. seealso::
        Inherited method :meth:`.firstorder.atomic.AtomicFormula.to_complement`

    .. automethod:: fvars

    .. automethod:: simplify

    .. automethod:: strict_part

      +---------------------+-------------+-------------+-------------+-------------+
      |                     | :class:`Le` | :class:`Ge` | :class:`Lt` | :class:`Gt` |
      +=====================+=============+=============+=============+=============+
      | :meth:`strict_part` | :class:`Lt` | :class:`Gt` | :class:`Lt` | :class:`Gt` |
      +---------------------+-------------+-------------+-------------+-------------+

    .. automethod:: subs


  .. class:: Eq
             Ge
             Gt
             Le
             Lt
             Ne

    Bases: :class:`.AtomicFormula`
