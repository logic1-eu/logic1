.. _api-firstorder-index:

.. |blank| unicode:: U+0020

*********************
First-order Framework
*********************

.. automodule:: logic1.firstorder

The following picture summarizes the inheritance hierarchy.

.. topic:: |blank|

  .. graphviz::
    :class: only-light

     digraph foo {
        bgcolor="transparent";
        edge [dir=back, arrowtail=empty];
        node [shape=plaintext, fontname="monospace"];
        "Formula" -> "QuantifiedFormula";
        "Formula" -> "BooleanFormula" ;
        "Formula" -> "AtomicFormula";
        QuantifiedFormula -> "Ex | All";
        "BooleanFormula" -> "Equivalent | Implies | And | Or | Not | _T | _F";
        AtomicFormula -> "RCF.AtomicFormula | ...";
     }

  .. graphviz::
    :class: only-dark

     digraph foo {
        bgcolor="transparent";
        edge [dir=back, arrowtail=empty, color="white"];
        node [shape=plaintext, fontname="monospace", fontcolor="white"];
        "Formula" -> "QuantifiedFormula";
        "Formula" -> "BooleanFormula" ;
        "Formula" -> "AtomicFormula";
        QuantifiedFormula -> "Ex | All";
        "BooleanFormula" -> "Equivalent | Implies | And | Or | Not | _T | _F";
        AtomicFormula -> "RCF.AtomicFormula | ...";
     }


.. toctree::
   :hidden:

   formula.rst
   atomic.rst
