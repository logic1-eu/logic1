from typing import Iterable, Optional

from logic1.firstorder.boolean import _F, _T, And, Equivalent, Implies, Not, Or
from logic1.firstorder.quantified import All, Ex
from logic1.theories import RCF
from logic1.theories.Complex.ast import _I, Conj, Im, Rat, Re, Var
from logic1.theories.Complex.simplify import simplify
from logic1.theories.Complex.term import VV, Term
from logic1.theories.Complex.atomic import AtomicFormula, Eq, Ne, Ge, Gt, Le, Lt
from logic1.theories.Complex.normalize import ArithmeticEvaluator, RealNormalizer, ComplexNormalizer, cartesian_normal_form, conjugate_normal_form
from logic1.theories.Complex.types import Formula


class RCF_Evaluator(ArithmeticEvaluator[RCF.Term]):
    """A term visitor that evaluates a complex term to a term in the
    theory of real closed fields. Raises a ValueError if the term
    contains any complex-specific operations that cannot be evaluated
    in RCF, such as the imaginary unit or complex conjugation.
    Implements the abstract class :class:`.Complex.ArithmeticEvaluator`.
    """

    def _add(self, a: RCF.Term, b: RCF.Term) -> RCF.Term:
        """Returns the sum of the RCF terms a and b. Implements
        abstract method :meth:`.Complex.ArithmeticEvaluator._add`.
        """
        return a + b

    def _neg(self, a: RCF.Term) -> RCF.Term:
        """Returns the negation of the RCF term a. Implements the
        abstract method :meth:`.Complex.ArithmeticEvaluator._neg`.
        """
        return -a

    def _mul(self, a: RCF.Term, b: RCF.Term) -> RCF.Term:
        """Returns the product of the RCF terms a and b. Implements the
        abstract method :meth:`.Complex.ArithmeticEvaluator._mul`.
        """
        return a * b

    def visit_rat(self, num: Rat) -> RCF.Term:
        """Returns the RCF term corresponding to the rational term num.
        Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_rat`.
        """
        return RCF.Term(num.value)

    def visit_i(self, _: _I) -> RCF.Term:
        """Cannot evaluate the imaginary unit in RCF. Implements the
        abstract method :meth:`.Complex.TermVisitor.visit_i`.
        """
        raise ValueError("Cannot evaluate imaginary unit in RCF")

    def visit_var(self, var: Var) -> RCF.Term:
        """Cannot evaluate complex variables in RCF. Implements the
        abstract method :meth:`.Complex.TermVisitor.visit_var`.
        """
        raise ValueError(f"Cannot evaluate complex variable {var} in RCF")

    def visit_conj(self, conj: Conj) -> RCF.Term:
        """Cannot evaluate complex conjugations in RCF. Implements the
        abstract method :meth:`.Complex.TermVisitor.visit_conj`.
        """
        raise ValueError(f"Cannot evaluate complex conjunction {conj} in RCF")

    def visit_re(self, re: Re) -> RCF.Term:
        if isinstance(re.arg, Var):
            return RCF.VV[f"{re.arg.name}_re"]
        else:
            raise ValueError(f"Cannot evaluate real part of non-variable term {re.arg} in RCF")

    def visit_im(self, im: Im) -> RCF.Term:
        if isinstance(im.arg, Var):
            return RCF.VV[f"{im.arg.name}_im"]
        else:
            raise ValueError(f"Cannot evaluate imaginary part of non-variable term {im.arg} in RCF")

def real_atom_to_rcf(atom: AtomicFormula) -> RCF.AtomicFormula:
    """Converts an atomic formula in the theory of complex numbers that is
    already known to be a real formula to an equivalent atomic formula
    in the theory of real closed fields.
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

def formula_to_rcf(formula: Formula) -> RCF.Formula:
    """Converts a formula in the theory of complex numbers to an
    equivalent formula in the theory of real closed fields. Raises a
    ValueError if the formula contains any complex-specific operations
    that cannot be evaluated in RCF, such as the imaginary unit or
    complex conjugation.
    """
    if isinstance(formula, AtomicFormula):
       if formula.is_real():
            return real_atom_to_rcf(formula)
       else:
            formula = formula.as_real_formula()
            return formula_to_rcf(formula)
    if isinstance(formula, (All, Ex)):
        var = conjugate_normal_form(formula.var.normal_ast)
        assert isinstance(var, Var)
        var_re = RCF.VV[f"{var.name}_re"]
        var_im = RCF.VV[f"{var.name}_im"]
        arg = formula_to_rcf(formula.arg)
        return formula.op([var_re, var_im], arg)
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, _T, _F)):
        return formula.op(*(formula_to_rcf(arg) for arg in formula.args))
    assert False, type(formula)


def variable_to_complex(var: RCF.Variable) -> Term:
    """Converts an RCF variable to a complex variable. The RCF variable
    must be of the form x_re or x_im for some variable name x, where
    x_re corresponds to the real part of the complex variable and x_im
    corresponds to the imaginary part.
    """
    name = str(var)  # TODO: implement .name
    if name.endswith('_re'):
        return VV[name[:-3]].real_part()
    elif name.endswith('_im'):
        return VV[name[:-3]].imaginary_part()
    else:
        raise ValueError(f"Unknown variable {name} in RCF formula")


def term_to_complex(term: RCF.Term) -> Term:
    """Converts an RCF term to a complex term. The RCF term must be a
    polynomial in variables of the form x_re and x_im for some variable
    name x, where x_re corresponds to the real part of the complex
    variable and x_im corresponds to the imaginary part.
    """
    result: Term = Term(0)
    for vars, coeff in term.summands():
        power_product = Term(1)
        for var, exp in vars.items():
            power_product = power_product * variable_to_complex(var) ** exp
        result = result + coeff * power_product
    return result


def formula_to_complex(formula: RCF.Formula) -> Formula:
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
    """Quantifier elimination for the theory of complex numbers. Returns a
    quantifier-free formula equivalent to the input formula.
    """
    assumption = formula_to_rcf(And(*assume))
    if isinstance(assumption, RCF.AtomicFormula):
        rcf_assume = [assumption]
    elif isinstance(assumption, And):
        rcf_assume = [arg for arg in assumption.args
                      if isinstance(arg, RCF.AtomicFormula)]
    else:
        rcf_assume = []
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