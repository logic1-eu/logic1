.. _api-Complex-simplify:

*Complex*

**************
Simplification
**************

.. automodule:: logic1.theories.Complex.simplify

    .. note::
        The function :func:`.simplify` implements simplification for the theory
        of complex numbers. It takes a formula as input and returns an equivalent
        simplified formula modulo optional assumptions.

        >>> from logic1.firstorder import *
        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> simplify(And(z**2 == 1, Re(z) > 0))
        z - 1 == 0
        >>> simplify(Re(z) == 0, assume=[Im(z) == 1])
        z - I == 0

        The function :func:`.is_valid` heuristically checks whether a formula is
        valid, i.e. the formula holds for all possible values of its free
        variables. In case the validity of the formula cannot be determined, the
        function returns :obj:`None`.

        >>> from logic1.firstorder import *
        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> is_valid(z * ~z >= 0)
        True
        >>> is_valid(Re(z) > 0, assume=[z == 0])
        False
        >>> print(is_valid(z == 0))
        None

    User Interface
    **************

    .. autofunction:: simplify

    .. autofunction:: is_valid

    .. attention::
        The following functions are not intended to be
        used directly by the user. Instead, they are used internally by the
        functions :func:`.simplify` and :func:`.is_valid`.
        However, they might be useful for advanced users who want to reconstruct
        complex formulas.

    Internal Representation and Simplify
    ************************************

    .. autoclass:: Options
        :exclude-members: __init__, __new__

    .. autoclass:: InternalRepresentation
        :members:
        :exclude-members: __init__

    .. autoclass:: Simplify
        :members:
        :exclude-members: __init__