.. _api-Sets-interactive:

*Sets*

***************
Interactive Use
***************

.. module:: logic1.interactive.Sets
   :synopsis: Interactive use of the theory of Sets

Module for automatically importing the first-order framework, theories, and
variables into the global namespace. This module is intended to be used in an
interactive Python session, e.g. in a Jupyter notebook.

>>> from logic1.interactive.Sets import *
>>> And(a == b, z != y)
And(a == b, z != y)