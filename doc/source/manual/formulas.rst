.. _formulas-index:

########
Formulas
########

Atomic Formulas
===============

First-order Formulas
====================

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
