.. _api-Complex-interactive:

*Complex*

***************
Interactive Use
***************

.. module:: logic1.interactive.Complex
   :synopsis: Interactive use of the theory of complex numbers

Module for automatically importing the first-order framework, theories, and
variables into the global namespace. This module is intended to be used in an
interactive Python session, e.g. in a Jupyter notebook.

>>> from logic1.interactive.Complex import *
>>> And(a * ~a >= 0, Re(z) == mpq(1, 2))
And(a * ~a >= 0, 1/2 * z + 1/2 * ~z == 1/2)
