.. _api-Sets-atomic:

*Sets with Cardinality Constraints*

***********************
Variables and Atoms
***********************

There is an infinite set :data:`Sets.VV <.Sets.atomic.VV>` of all :mod:`Sets <.theories.Sets>` variables, which is an instance of the class :class:`VariableSet <.Sets.atomic.VariableSet>`. The variables obtained from :data:`VV <.Sets.atomic.VV>` are instances of the class :class:`Variable <.Sets.atomic.Variable>`. There are no composite terms in this theory; in other words, every term is a variable.

There is a class :class:`AtomicFormula <.Sets.atomic.AtomicFormula>` with subclasses :class:`Eq <.Sets.atomic.Eq>`, :class:`Ne <.Sets.atomic.Ne>`, :class:`C <.Sets.atomic.C>`, :class:`C_ <.Sets.atomic.C_>`. Atoms are obtained as instances of these subclasses, using the class names as constructors. For equations and diseqalities, one can alternatively use the corresponding operators :meth:`== <.Sets.atomic.Variable.__eq__>` and :meth:`\!= <.Sets.atomic.Variable.__ne__>` overloaded in the class :class:`Variable <.Sets.atomic.Variable>`. :class:`C <.Sets.atomic.C>` and :class:`C_ <.Sets.atomic.C_>` accept positive numbers from :class:`int` and :data:`oo <.Sets.atomic.oo>` from :class:`float` as indices.

Some examples can be found on the :ref:`landing page <api-Sets>` of this section.

.. graphviz::
   :class: only-light

   digraph RCF_terms {
      graph [
         layout=neato,
         splines=line,
         sep=0.5,
         ranksep=0.5,
         nodesep=0.5
      ];
      bgcolor="transparent";

      node [shape=box, fontsize="10pt",
            fontname="monospace", penwidth=0.8];
      edge [arrowsize=0.75, penwidth=0.8];

      VV [shape=box, label = <<U>Sets.VV : VariableSet</U>>,
          pos="0.5,6.0!"];

      Variable [pos="0.5,4.5!"];
      int [pos="2.5,4.5!"];
      float [pos="3.5,4.5!"];
      intB [label="int", pos="5,4.5!"];
      floatB [label="float", pos="6,4.5!"];

      Comparison [style=rounded,
                  label="Variable methods  ==  !=",
                  pos="0.5,3!"];
      C [style=rounded, label="Construtor of C", pos="3,3!"];
      CB [style=rounded, label="Constructor of C_", pos="5.5,3!"];

      Eq [pos="0,1.5!"];
      Ne [pos="1,1.5!"];
      C1 [label = "C", pos="3,1.5!"];
      CB1 [label="C_", pos="5.5,1.5!"];

      AtomicFormula [pos="3,0!"];

      VV -> Variable
         [xlabel="yields   ",
          style=dashed, arrowhead=normal,
          fontsize="10pt", fontname="sans-serif"];

      Variable -> Comparison
         [style=dashed, arrowhead=none,
          xlabel="enters   ", fontsize="10pt", fontname="sans-serif"];

      int -> C
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      float -> C
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      intB -> CB
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      floatB -> CB
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      Comparison -> Eq
         [style=dashed, arrowhead=normal,
          xlabel="construct   ", fontsize="10pt", fontname="sans-serif"];

      Comparison -> Ne
         [style=dashed, arrowhead=normal];

      C -> C1
         [style=dashed, arrowhead=normal];

      CB -> CB1
         [style=dashed, arrowhead=normal];

      Eq -> AtomicFormula
         [arrowhead=empty, xlabel="subclass of               ",
          fontsize="10pt", fontname="sans-serif"];

      C1 -> AtomicFormula [arrowhead=empty];
      Ne -> AtomicFormula [arrowhead=empty];
      CB1 -> AtomicFormula [arrowhead=empty];
   }

.. graphviz::
   :class: only-dark

   digraph RCF_terms {
      graph [
         layout=neato,
         splines=line,
         sep=0.5,
         ranksep=0.5,
         nodesep=0.5
      ];
      bgcolor="transparent";

      node [shape=box, fontsize="10pt", fontname="monospace", penwidth=0.8,
            color=white, fontcolor=white];
      edge [arrowsize=0.75, penwidth=0.8
            color=white, fontcolor=white];

      VV [shape=box, label = <<U>Sets.VV : VariableSet</U>>,
          pos="0.5,6.0!"];

      Variable [pos="0.5,4.5!"];
      int [pos="2.5,4.5!"];
      float [pos="3.5,4.5!"];
      intB [label="int", pos="5,4.5!"];
      floatB [label="float", pos="6,4.5!"];

      Comparison [style=rounded,
                  label="Variable methods  ==  !=",
                  pos="0.5,3!"];
      C [style=rounded, label="Construtor of C", pos="3,3!"];
      CB [style=rounded, label="Constructor of C_", pos="5.5,3!"];

      Eq [pos="0,1.5!"];
      Ne [pos="1,1.5!"];
      C1 [label = "C", pos="3,1.5!"];
      CB1 [label="C_", pos="5.5,1.5!"];

      AtomicFormula [pos="3,0!"];

      VV -> Variable
         [xlabel="yields   ",
          style=dashed, arrowhead=normal,
          fontsize="10pt", fontname="sans-serif"];

      Variable -> Comparison
         [style=dashed, arrowhead=none,
          xlabel="enters   ", fontsize="10pt", fontname="sans-serif"];

      int -> C
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      float -> C
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      intB -> CB
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      floatB -> CB
         [style=dashed, arrowhead=none,
          fontsize="10pt", fontname="sans-serif"];

      Comparison -> Eq
         [style=dashed, arrowhead=normal,
          xlabel="construct   ", fontsize="10pt", fontname="sans-serif"];

      Comparison -> Ne
         [style=dashed, arrowhead=normal];

      C -> C1
         [style=dashed, arrowhead=normal];

      CB -> CB1
         [style=dashed, arrowhead=normal];

      Eq -> AtomicFormula
         [arrowhead=empty, xlabel="subclass of               ",
          fontsize="10pt", fontname="sans-serif"];

      C1 -> AtomicFormula [arrowhead=empty];
      Ne -> AtomicFormula [arrowhead=empty];
      CB1 -> AtomicFormula [arrowhead=empty];
   }

.. automodule:: logic1.theories.Sets.atomic

  ========================
  The Set of All Variables
  ========================

  .. autoclass:: VariableSet
    :exclude-members: __init__, __new__

    .. autoproperty:: stack

    .. automethod:: __getitem__

    .. automethod:: fresh

    .. method::
      pop() -> None
      push() -> None
      :abstractmethod:

      Implement abstract methods
      :meth:`.logic1.firstorder.term.VariableSet.pop` and
      :meth:`.logic1.firstorder.term.VariableSet.push`.


  .. data:: VV
    :value: VariableSet()

    The unique instance of :class:`.VariableSet`. This is a singleton.


  =========
  Variables
  =========

  .. autoclass:: Variable
    :exclude-members: __init__, __new__

    .. method:: ==, !=
                __eq__(other: Variable) -> Eq
                __ne__(other: Variable) -> Ne

      Construction of instances of :class:`Eq` and :class:`Ne` is available via
      these overloaded operators.

    .. automethod:: as_latex

    .. automethod:: fresh

    .. automethod:: sort_key

    .. automethod:: subs

    .. automethod:: vars

  =====
  Atoms
  =====

  .. autoclass:: AtomicFormula
    :exclude-members: __init__, __new__

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
    or ``float('inf')``, represented as ``oo``. ``C(n)`` holds iff there are at
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

      The index of the constant relation symbol.