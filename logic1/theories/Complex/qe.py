from typing import Iterable, Optional

from logic1.firstorder.boolean import _F, _T, And, Equivalent, Implies, Not, Or
from logic1.firstorder.quantified import All, Ex
from logic1.theories import RCF

from logic1.theories.Complex.types import Formula
from logic1.theories.Complex import ast
from logic1.theories.Complex.normalize import ArithmeticEvaluator, cartesian_normal_form, conjugate_normal_form
from logic1.theories.Complex.term import VV, Term
from logic1.theories.Complex.atomic import AtomicFormula, Eq, Ne, Ge, Gt, Le, Lt


class RCF_Evaluator(ArithmeticEvaluator[RCF.Term]):
    """Visitor that evaluates a :class:`.ast.AST` to a term in the
    theory of real closed fields. Raise a :class:`ValueError` if the AST
    contains any complex-specific operations that cannot be evaluated.
    Implements the abstract class :class:`.ast.ArithmeticEvaluator`.

    >>> z = ast.Var('z')
    >>> (ast.Re(z) + 1).accept(RCF_Evaluator())
    z_re + 1
    >>> (z + 1).accept(RCF_Evaluator())
    Traceback (most recent call last):
    ...
    ValueError: Cannot evaluate complex variable z in RCF
    """

    def add(self, a: RCF.Term, b: RCF.Term) -> RCF.Term:
        """Return the sum of two RCF terms. Implements
        abstract method :meth:`.Complex.ArithmeticEvaluator.add`.

        >>> x, y = RCF.VV.get('x', 'y')
        >>> RCF_Evaluator().add(x, y)
        x + y
        """
        return a + b

    def neg(self, a: RCF.Term) -> RCF.Term:
        """Return the negation of a RCF term. Implements the
        abstract method :meth:`.Complex.ArithmeticEvaluator.neg`.

        >>> x = RCF.VV['x']
        >>> RCF_Evaluator().neg(x)
        -x
        """
        return -a

    def mul(self, a: RCF.Term, b: RCF.Term) -> RCF.Term:
        """Return the product of two RCF terms. Implements the
        abstract method :meth:`.Complex.ArithmeticEvaluator.mul`.

        >>> x, y = RCF.VV.get('x', 'y')
        >>> RCF_Evaluator().mul(x, y)
        x*y
        """
        return a * b

    def visit_rat(self, num: ast.Rat) -> RCF.Term:
        """Return the RCF term corresponding to a rational number.
        Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_rat`.

        >>> RCF_Evaluator().visit_rat(ast.Rat(2))
        2
        """
        return RCF.Term(num.value)

    def visit_i(self, _: ast._I) -> RCF.Term:
        """Raise a :class:`ValueError` since the imaginary unit can not be
        evaluated in RCF. Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_i`.

        >>> RCF_Evaluator().visit_i(ast.I)
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate imaginary unit in RCF
        """
        raise ValueError("Cannot evaluate imaginary unit in RCF")

    def visit_var(self, var: ast.Var) -> RCF.Term:
        """Raise a :class:`ValueError` since complex variables can not be
        evaluated in RCF. Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_var`.

        >>> z = ast.Var('z')
        >>> RCF_Evaluator().visit_var(z)
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate complex variable z in RCF
        """
        raise ValueError(f"Cannot evaluate complex variable {var} in RCF")

    def visit_conj(self, conj: ast.Conj) -> RCF.Term:
        """Raise a :class:`ValueError` since complex conjugation can not be
        evaluated in RCF. Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_conj`.

        >>> z = ast.Var('z')
        >>> RCF_Evaluator().visit_conj(z)
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate complex conjugation in RCF
        """
        raise ValueError(f"Cannot evaluate complex conjugation in RCF")

    def visit_re(self, re: ast.Re) -> RCF.Term:
        """Return the RCF term corresponding to the real part of a complex
        variable. If the argument of :code:`re` is not a variable, raise a
        :class:`ValueError`. Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_re`.

        >>> z = ast.Var('z')
        >>> RCF_Evaluator().visit_re(ast.Re(z))
        z_re
        >>> RCF_Evaluator().visit_re(ast.Re(z + 1))
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate real part of non-variable term z + 1 in RCF
        """
        if isinstance(re.arg, ast.Var):
            return RCF.VV[f"{re.arg.name}_re"]
        else:
            raise ValueError(f"Cannot evaluate real part of non-variable term {re.arg} in RCF")

    def visit_im(self, im: ast.Im) -> RCF.Term:
        """Return the RCF term corresponding to the imaginary part of a complex
        variable. If the argument of :code:`im` is not a variable, raise a
        :class:`ValueError`. Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_im`.

        >>> z = ast.Var('z')
        >>> RCF_Evaluator().visit_im(ast.Im(z))
        z_im
        >>> RCF_Evaluator().visit_im(ast.Im(z + 1))
        Traceback (most recent call last):
        ...
        ValueError: Cannot evaluate imaginary part of non-variable term z + 1 in RCF
        """
        if isinstance(im.arg, ast.Var):
            return RCF.VV[f"{im.arg.name}_im"]
        else:
            raise ValueError(f"Cannot evaluate imaginary part of non-variable term {im.arg} in RCF")


def real_atom_to_rcf(atom: AtomicFormula) -> RCF.AtomicFormula:
    """Convert a real atomic formula in the theory of complex numbers to
    an equivalent atomic formula in the theory of real closed fields.

    >>> z = VV['z']
    >>> real_atom_to_rcf(z * ~z == 0)
    z_im**2 + z_re**2 == 0
    """
    assert atom.is_real()
    lhs = cartesian_normal_form((atom.lhs - atom.rhs).normal_ast).accept(RCF_Evaluator())
    if isinstance(atom, Eq):
        return RCF.Eq(lhs, 0)
    elif isinstance(atom, Ne):
        return RCF.Ne(lhs, 0)
    elif isinstance(atom, Ge):
        return RCF.Ge(lhs, 0)
    elif isinstance(atom, Gt):
        return RCF.Gt(lhs, 0)
    elif isinstance(atom, Le):
        return RCF.Le(lhs, 0)
    elif isinstance(atom, Lt):
        return RCF.Lt(lhs, 0)
    else:
        assert False, type(atom)

def real_formula_to_rcf(formula: Formula) -> RCF.Formula:
    """Convert a real formula in the theory of complex numbers to an
    equivalent formula in the theory of real closed fields.

    >>> z = VV['z']
    >>> phi = All(z, z * ~z == 0)
    >>> real_formula_to_rcf(phi)
    All(z_re, All(z_im, z_im**2 + z_re**2 == 0))
    """
    if isinstance(formula, AtomicFormula):
        return real_atom_to_rcf(formula)
    if isinstance(formula, (All, Ex)):
        var = conjugate_normal_form(formula.var.normal_ast)
        assert isinstance(var, ast.Var)
        var_re = RCF.VV[f"{var.name}_re"]
        var_im = RCF.VV[f"{var.name}_im"]
        arg = real_formula_to_rcf(formula.arg)
        return formula.op([var_re, var_im], arg)
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, _T, _F)):
        return formula.op(*(real_formula_to_rcf(arg) for arg in formula.args))
    assert False, type(formula)

def real_normal_form(formula: Formula) -> Formula:
    # TODO: rename?
    if isinstance(formula, AtomicFormula):
       return formula.real_normal_form()
    if isinstance(formula, (All, Ex)):
        return formula.op(formula.var, real_normal_form(formula.arg))
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, _T, _F)):
        return formula.op(*(real_normal_form(arg) for arg in formula.args))
    assert False, type(formula)

def formula_to_rcf(formula: Formula) -> RCF.Formula:
    """Convert a formula in the theory of complex numbers to an
    equivalent formula in the theory of real closed fields.

    >>> z = VV['z']
    >>> phi = Ex(z, z**2 == -1)
    >>> formula_to_rcf(phi)
    Ex(z_re, Ex(z_im, And(-z_im**2 + z_re**2 + 1 == 0, 2*z_im*z_re == 0)))
    """
    formula = real_normal_form(formula)
    return real_formula_to_rcf(formula)


def assume_to_rcf(assume: Iterable[AtomicFormula]) -> list[RCF.AtomicFormula]:
    """Convert a list of atomic formulas in the theory of complex numbers to
    a list of atomic formulas in the theory of real closed fields. The new list
    of atomic formulas is not necessarily equivalent to the original list but
    implied by it.

    >>> z = VV['z']
    >>> assume = [z * ~z == 1, Or(z == 1, z == -1)]
    >>> assume_to_rcf(assume)
    [z_im**2 + z_re**2 - 1 == 0, Eq(0, 0)]
    """
    assumption = formula_to_rcf(And(*assume))
    if isinstance(assumption, RCF.AtomicFormula):
        return [assumption]
    elif isinstance(assumption, And):
        return [arg for arg in assumption.args if isinstance(arg, RCF.AtomicFormula)]
    else:
        return []

def variable_to_complex(var: RCF.Variable) -> Term:
    """Convert a RCF variable to a complex variable. The RCF variable
    name must be of the form :code:`*_re` or :code:`*_im`.

    >>> z_re, z_im = RCF.VV.get('z_re', 'z_im')
    >>> variable_to_complex(z_re)
    1/2 * z + 1/2 * ~z
    >>> variable_to_complex(z_im)
    -1/2 * I * z + 1/2 * I * ~z
    """
    name = str(var)  # TODO: implement .name
    if name.endswith('_re'):
        return VV[name[:-3]].real_part()
    elif name.endswith('_im'):
        return VV[name[:-3]].imaginary_part()
    else:
        raise ValueError(f"Unknown variable {name} in RCF formula")

def term_to_complex(term: RCF.Term) -> Term:
    """Convert a RCF term to a complex term. The RCF term must be a
    polynomial in variables of the form :code:`*_re` and :code:`*_im`.

    >>> z_re, z_im = RCF.VV.get('z_re', 'z_im')
    >>> term = z_im**2 + z_re**2
    >>> term_to_complex(term)
    z * ~z
    """
    result: Term = Term(0)
    for vars, coeff in term.summands():
        power_product = Term(1)
        for var, exp in vars.items():
            power_product = power_product * variable_to_complex(var) ** exp
        result = result + coeff * power_product
    return result

def formula_to_complex(formula: RCF.Formula) -> Formula:
    """Convert a formula in the theory of real closed fields to an equivalent
    formula in the theory of complex numbers. Raise a :class:`ValueError` if the
    RCF formula is quantified and or contains variables not of the form
    :code:`*_re` and :code:`*_im`.

    >>> z_re, z_im = RCF.VV.get('z_re', 'z_im')
    >>> phi = And(z_im**2 + z_re**2 == 0, 2*z_im*z_re == 0)
    >>> formula_to_complex(phi)
    And(z * ~z == 0, -1/2 * I * z**2 + 1/2 * I * (~z)**2 == 0)
    """
    OPS: dict[type[RCF.AtomicFormula], type[AtomicFormula]] = {
        RCF.Eq: Eq, RCF.Ne: Ne, RCF.Ge: Ge, RCF.Gt: Gt, RCF.Le: Le, RCF.Lt: Lt}
    if isinstance(formula, RCF.AtomicFormula):
        lhs = term_to_complex(formula.lhs)
        rhs = term_to_complex(formula.rhs)
        return OPS[type(formula)](lhs, rhs)
    if isinstance(formula, (All, Ex)):
        raise ValueError("Cannot convert quantified RCF formula back to complex formula")
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, _T, _F)):
        return formula.op(*(formula_to_complex(arg) for arg in formula.args))
    assert False, type(formula)

def qe(formula: Formula, assume: Iterable[AtomicFormula] = [], use_redlog: bool = False, **options) -> Formula:
    """Return a quantifier-free formula equivalent to the input formula.

    :param formula:
        The input formula to which quantifier elimination will be applied.

    :param assume:
        A list of atomic formulas that are assumed to hold. The return value
        is equivalent modulo those assumptions.

    :param use_redlog:
        If :obj:`True`, use :func:`RCF.redlog.qe` internallyto perform
        real quantifier elimination. By default, :func:`RCF.qe` is used.

    :param options:
        Additional keyword arguments to be passed to :func:`RCF.qe` or
        :func:`RCF.redlog.qe`.

    >>> z = VV['z']
    >>> phi = Ex(z, z**2 == -1)
    >>> qe(phi)
    T
    """
    rcf_assume = assume_to_rcf(assume)
    rcf_formula = formula_to_rcf(formula).to_pnf()
    topmost_op = type(rcf_formula)
    if use_redlog:
        rcf_qe_formula: Optional[RCF.Formula] = RCF.redlog.qe(rcf_formula, assume=rcf_assume)
    else:
        rcf_qe_formula = RCF.qe(rcf_formula, assume=rcf_assume, **options)
    if rcf_qe_formula is None:
        raise ValueError("Quantifier elimination failed")
    # TS: I had suggested to look at rcf_qe_formula.op, but now I think it is
    # better to stick with checking the topmost_op. When in doubt, DNF should be
    # used. The double calls of CNF/DNF are probably the best solution at the
    # moment. We can discuss this.
    if topmost_op == All:
        rcf_qe_formula = RCF.cnf(RCF.cnf(rcf_qe_formula))
    else:
        rcf_qe_formula = RCF.dnf(RCF.dnf(rcf_qe_formula))
    result = formula_to_complex(rcf_qe_formula)
    result = simplify(result)
    return result


from logic1.theories.Complex.simplify import simplify