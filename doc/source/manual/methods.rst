.. _formulamethods-index:

.. default-domain:: py

###################
Methods on Formulas
###################

Data Extraction
===============

.. autofunction:: logic1.firstorder.Formula.count_alternations
  :noindex:

.. autofunction:: logic1.firstorder.Formula.bvars
  :noindex:

.. autofunction:: logic1.firstorder.Formula.fvars
  :noindex:

.. autofunction:: logic1.firstorder.Formula.qvars
  :noindex:

.. index::
  Normal Forms

Transformations
===============

.. index::
  substitution

.. autofunction:: logic1.firstorder.Formula.subs
  :noindex:

.. index::
  simplification

.. autofunction:: logic1.firstorder.Formula.simplify
  :noindex:

.. index::
  LaTeX; connversion to

.. autofunction:: logic1.firstorder.Formula.as_latex
  :noindex:

.. autofunction:: logic1.firstorder.Formula.transform_atoms
  :noindex:

This often allows to write uniform code
for objects where the number or types of elements of :attr:`args` are not
known, in particular in recursions.


Normal Forms
============

.. index::
  negation normal form
  NNF

Negation Normal Form
--------------------

.. autofunction:: logic1.firstorder.Formula.to_nnf
  :noindex:

.. index::
  prenex normal form
  PNF

Prenex Normal Form
------------------
.. autofunction:: logic1.firstorder.pnf
   :noindex:


.. index::
  conjunctive normal form
  CNF
  disjunctive normal form
  DNF

.. Conjunctive and Disjunctive Normal Form
.. ---------------------------------------
.. .. autofunction:: logic1.firstorder.boolean.BooleanFormula.to_cnf
..    :noindex:
.. .. autofunction:: logic1.firstorder.boolean.BooleanFormula.to_dnf
..    :noindex:
