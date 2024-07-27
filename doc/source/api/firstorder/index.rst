.. _api-firstorder-index:

.. |blank| unicode:: U+0020

*********************
First-order Framework
*********************

.. automodule:: logic1.firstorder

The following class diagram summarizes the inheritance hierarchy.

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


.. toctree::
   :hidden:

   formula.rst
   atomic.rst
