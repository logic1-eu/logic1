
from typing import Any

from logic1.firstorder.boolean import _F, _T, And, Equivalent, Implies, Not, Or
from logic1.firstorder.quantified import All, Ex
from logic1.theories import RCF
from logic1.theories.Complex.simplify import simplify
from logic1.theories.RCF.atomic import AtomicFormula as RCF_AtomicFormula  # TODO: export in __init__
from logic1.theories.RCF.typing import Formula as RCF_Formula  # TODO: export in __init__
from logic1.theories.Complex.atomic import AtomicFormula, AtomicFormulaVisitor, Eq, Ne, Ge, Gt, Le, Lt
from logic1.theories.Complex.normalize import ArithmeticEvaluator
from logic1.theories.Complex.term import Add, Mul, Neg, Rational, Term, Variable, _I, Conj, Re, Im
from logic1.theories.Complex.types import Formula
from gmpy2 import mpq


class RCF_Evaluator(ArithmeticEvaluator[RCF.Term], AtomicFormulaVisitor[RCF_Formula]):
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
    
    def visit_rational(self, num: Rational) -> RCF.Term:
        """Returns the RCF term corresponding to the rational term num.
        Implements the abstract method
        :meth:`.Complex.TermVisitor.visit_rational`.
        """
        return RCF.Term(num.value)

    def visit_i(self, _: _I) -> RCF.Term:
        """Cannot evaluate the imaginary unit in RCF. Implements the
        abstract method :meth:`.Complex.TermVisitor.visit_i`.
        """
        raise ValueError("Cannot evaluate imaginary unit in RCF")
    
    def visit_variable(self, var: Variable) -> RCF.Term:
        """Cannot evaluate complex variables in RCF. Implements the
        abstract method :meth:`.Complex.TermVisitor.visit_variable`.
        """
        raise ValueError(f"Cannot evaluate complex variable {var} in RCF")
    
    def visit_conj(self, conj: Conj) -> RCF.Term:
        """Cannot evaluate complex conjugations in RCF. Implements the
        abstract method :meth:`.Complex.TermVisitor.visit_conj`.
        """
        raise ValueError(f"Cannot evaluate complex conjunction {conj} in RCF")
    
    def visit_re(self, re: Re) -> RCF.Term:
        if isinstance(re.arg, Variable):
            return RCF.VV[f"{re.arg.name}_re"]
        else:
            raise ValueError(f"Cannot evaluate real part of non-variable term {re.arg} in RCF")
        
    def visit_im(self, im: Im) -> RCF.Term:
        if isinstance(im.arg, Variable):
            return RCF.VV[f"{im.arg.name}_im"]
        else:
            raise ValueError(f"Cannot evaluate imaginary part of non-variable term {im.arg} in RCF")
        
    def visit_eq(self, eq: Eq) -> RCF.Eq:
        """Returns the RCF formula corresponding to the complex equality
        eq. Implements the abstract method
        :meth:`.Complex.AtomicFormulaVisitor.visit_eq`.
        """
        return RCF.Eq(eq.lhs.accept(self), eq.rhs.accept(self))
    
    def visit_ne(self, ne: Ne) -> RCF.Ne:
        """Returns the RCF formula corresponding to the complex
        inequality ne. Implements the abstract method
        :meth:`.Complex.AtomicFormulaVisitor.visit_ne`.
        """
        return RCF.Ne(ne.lhs.accept(self), ne.rhs.accept(self))

    def visit_ge(self, ge: Ge) -> RCF.Ge:
        """Returns the RCF formula corresponding to the complex
        greater-than-or-equal ge. Implements the abstract method
        :meth:`.Complex.AtomicFormulaVisitor.visit_ge`.
        """
        return RCF.Ge(ge.lhs.accept(self), ge.rhs.accept(self))

    def visit_le(self, le: Le) -> RCF.Le:
        """Returns the RCF formula corresponding to the complex
        less-than-or-equal le. Implements the abstract method
        :meth:`.Complex.AtomicFormulaVisitor.visit_le`.
        """
        return RCF.Le(le.lhs.accept(self), le.rhs.accept(self))
    
    def visit_gt(self, gt: Gt) -> RCF.Gt:
        """Returns the RCF formula corresponding to the complex
        greater-than gt. Implements the abstract method
        :meth:`.Complex.AtomicFormulaVisitor.visit_gt`.
        """
        return RCF.Gt(gt.lhs.accept(self), gt.rhs.accept(self))
    
    def visit_lt(self, lt: Lt) -> RCF.Lt:
        """Returns the RCF formula corresponding to the complex
        less-than lt. Implements the abstract method
        :meth:`.Complex.AtomicFormulaVisitor.visit_lt`.
        """
        return RCF.Lt(lt.lhs.accept(self), lt.rhs.accept(self))
        

def formula_to_rcf(formula: Formula) -> RCF_Formula:
    """Converts a formula in the theory of complex numbers to an
    equivalent formula in the theory of real closed fields. Raises a
    ValueError if the formula contains any complex-specific operations
    that cannot be evaluated in RCF, such as the imaginary unit or
    complex conjugation.
    """
    if isinstance(formula, AtomicFormula): 
       if formula.is_real():
            formula = formula.op(Re(formula.lhs), Re(formula.rhs)).normalize()
            return formula.accept(RCF_Evaluator())
       else:
            formula = formula.as_real_formula()
            return formula_to_rcf(formula)
    if isinstance(formula, (All, Ex)):
        var_re = RCF.VV[f"{formula.var.name}_re"]
        var_im = RCF.VV[f"{formula.var.name}_im"]
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
        return Re(Variable(name[:-3]))
    elif name.endswith('_im'):
        return Im(Variable(name[:-3]))
    else:
        raise ValueError(f"Unknown variable {name} in RCF formula")


def term_to_complex(term: RCF.Term) -> Term:
    """Converts an RCF term to a complex term. The RCF term must be a
    polynomial in variables of the form x_re and x_im for some variable
    name x, where x_re corresponds to the real part of the complex
    variable and x_im corresponds to the imaginary part.
    """
    result: Term = Rational(mpq(0))
    for vars, coeff in term.summands():
        power_product = Mul(*[variable_to_complex(var) ** exp for var, exp in vars.items()])
        result = result + Rational(coeff) * power_product
    return result.normalize_weak()


def formula_to_complex(formula: RCF_Formula) -> Formula:
    OPS: dict[type[RCF_AtomicFormula], type[AtomicFormula]] = {RCF.Eq: Eq, RCF.Ne: Ne, RCF.Ge: Ge, RCF.Gt: Gt, RCF.Le: Le, RCF.Lt: Lt}
    if isinstance(formula, RCF_AtomicFormula):
        lhs = term_to_complex(formula.lhs)
        rhs = term_to_complex(formula.rhs)
        return OPS[type(formula)](lhs, rhs)
    if isinstance(formula, (All, Ex)):
        raise ValueError("Cannot convert quantified RCF formula back to complex formula")
    if isinstance(formula, (And, Or, Not, Implies, Equivalent, _T, _F)):
        return formula.op(*(formula_to_complex(arg) for arg in formula.args))
    assert False, type(formula)
    

def qe(formula: Formula, final_simplify: bool = True) -> Formula:
    """Quantifier elimination for the theory of complex numbers. Returns a
    quantifier-free formula equivalent to the input formula.
    """
    rcf_formula = formula_to_rcf(formula)
    rcf_qe_formula = RCF.qe(rcf_formula)
    if rcf_qe_formula is None:
        raise ValueError("Quantifier elimination failed")
    result = formula_to_complex(rcf_qe_formula)
    if final_simplify:
        result = simplify(result)
    return result