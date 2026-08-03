.. _api-Complex-qe:

*Complex*

**********************
Quantifier Elimination
**********************

.. automodule:: logic1.theories.Complex.qe

     .. note::
        The function :func:`qe <.Complex.qe.qe>` implements quantifier
        elimination for the theory of complex numbers. It takes a formula as
        input and returns an equivalent quantifier-free formula modulo
        optional assumptions.

        >>> from logic1.firstorder import *
        >>> from logic1.theories.Complex import *
        >>> z = VV['z']
        >>> qe(Ex(z, z**2 + 1 == 0))
        T
        >>> a = VV['a']
        >>> qe(Ex(z, Re(z)**2 == a), assume=[Im(a) == 0])
        a + ~a >= 0

        Internally, quantifier elimination is performed by first converting the
        formula into :mod:`RCF <logic1.theories.RCF>` and then applying
        quantifier elimination :func:`qe <logic1.theories.RCF.qe.qe>`
        for real closed fields.

    User Interface
    **************

    .. autofunction:: qe

    RCF--Complex Conversion
    ***********************

    .. attention::
        The following functions are not intended to be
        used directly by the user. Instead, they are used internally by the
        function :func:`qe <.Complex.qe.qe>`. However, they might be useful for
        advanced users who want to convert formulas and terms between the theory
        of complex numbers and the theory of real closed fields.

    .. autofunction:: real_formula_to_rcf

    .. autofunction:: real_normal_form

    .. autofunction:: formula_to_rcf

    .. autofunction:: term_to_complex

    .. autofunction:: formula_to_complex

    .. autoclass:: RCF_Evaluator
        :members:
        :special-members:
