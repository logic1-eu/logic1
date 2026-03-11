
from typing import Any

from logic1.firstorder.boolean import And, Implies, Implies, Not, Or, Equivalent
from logic1.firstorder.quantified import All, Ex
from logic1.theories import RCF
from logic1.theories.RCF.atomic import AtomicFormula as RCF_AtomicFormula  # TODO: export
from logic1.theories.RCF.typing import Formula as RCF_Formula  # TODO: export
from logic1.theories.Complex.atomic import AtomicFormula, Eq, Ne, Ge, Gt, Le, Lt
from logic1.theories.Complex.simplify import ArithmeticEvaluator
from logic1.theories.Complex.term import Add, Mul, Neg, Rational, Term, Variable, _I, Conj, Re, Im
from logic1.theories.Complex.typing import Formula
from gmpy2 import mpq

class RCF_Evaluator(ArithmeticEvaluator[RCF.Term]):
    
    def _add(self, a: RCF.Term, b: RCF.Term) -> RCF.Term:
        return a + b
    
    def _neg(self, a: RCF.Term) -> RCF.Term:
        return -a
    
    def _mul(self, a: RCF.Term, b: RCF.Term) -> RCF.Term:
        return a * b
    
    def visit_rational(self, num: Rational) -> RCF.Term:
        return RCF.Term(num.value)

    def visit_i(self, _: _I) -> RCF.Term:
        raise ValueError("Cannot evaluate imaginary unit in RCF")
    
    def visit_variable(self, var: Variable) -> RCF.Term:
        raise ValueError(f"Cannot evaluate complex variable {var} in RCF")
    
    def visit_conj(self, conj: Conj) -> RCF.Term:
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
        

def formula_complex_to_rcf(formula: Formula) -> Any:  # TODO: export RCF.Formula, this could be a FormulaVisitor
    if isinstance(formula, AtomicFormula):
        a = Re(formula.lhs - formula.rhs).normalize().accept(RCF_Evaluator())  # type: ignore
        b = Im(formula.lhs - formula.rhs).normalize().accept(RCF_Evaluator())  # type: ignore
        if isinstance(formula, Eq):
            return And(a == 0, b == 0)
        if isinstance(formula, Ne):
            return Or(a != 0, b != 0)
        if isinstance(formula, Ge):
            return And(a >= 0, b == 0)
        if isinstance(formula, Le):
            return And(a <= 0, b == 0)
        if isinstance(formula, Gt):
            return And(a > 0, b == 0)
        if isinstance(formula, Lt):
            return And(a < 0, b == 0)
        raise ValueError(f"Unknown atomic formula type: {type(formula)}")
    if isinstance(formula, (All, Ex)):
        var_re = RCF.VV[f"{formula.var.name}_re"]
        var_im = RCF.VV[f"{formula.var.name}_im"]
        arg = formula_complex_to_rcf(formula.arg)
        return formula.op([var_re, var_im], arg)
    return formula.op(*(formula_complex_to_rcf(arg) for arg in formula.args))

def variable_rcf_to_complex(var: RCF.Variable) -> Term:
    name = str(var)  # TODO: implement .name
    if name.endswith('_re'):
        return Re(Variable(name[:-3]))
    elif name.endswith('_im'):
        return Im(Variable(name[:-3]))
    else:
        raise ValueError(f"Unknown variable {name} in RCF formula")

def term_rcf_to_complex(term: RCF.Term) -> Term:
    result: Term = Rational(mpq(0))
    for vars, coeff in term.summands():
        power_product = Mul(*[variable_rcf_to_complex(var) ** exp for var, exp in vars.items()])
        result = result + Rational(coeff) * power_product
    return result 

def formula_rcf_to_complex(formula: RCF_Formula) -> Formula:
    OPS = {RCF.Eq: Eq, RCF.Ne: Ne, RCF.Ge: Ge, RCF.Gt: Gt, RCF.Le: Le, RCF.Lt: Lt}
    if isinstance(formula, RCF_AtomicFormula):
        return OPS[type(formula)](term_rcf_to_complex(formula.lhs), term_rcf_to_complex(formula.rhs))
    if isinstance(formula, (All, Ex)):
        raise ValueError("Cannot convert quantified RCF formula back to complex formula")
    return formula.op(*(formula_rcf_to_complex(arg) for arg in formula.args))  # type: ignore
    

def qe(formula: Formula) -> Formula:
    """Quantifier elimination for the theory of complex numbers. Returns a
    quantifier-free formula equivalent to the input formula.
    """
    rcf_formula = formula_complex_to_rcf(formula)
    rcf_qe_formula = RCF.qe(rcf_formula)
    return formula_rcf_to_complex(rcf_qe_formula).simplify()  # type: ignore