"""Selected QF_NRA inputs from the SMT-LIB 2025 non-incremental release.

Source: https://doi.org/10.5281/zenodo.16740866 (SMT-LIB Initiative).
The original files are in ``inputs``; ``NOTES.md`` records their individual
contributors, licenses, and selection criteria. Each worker builds only its
own Logic1 formula before the timed benchmark call.

This reader supports polynomial expressions, Boolean connectives, lexical
``let`` bindings, and nullary definitions. Division requires a nonzero
constant denominator. Unsupported constructs are rejected.
"""

from fractions import Fraction
from pathlib import Path
import re
from typing import cast

from logic1.firstorder import And, F, Not, Or, T
from logic1.theories.RCF import VV
from logic1.theories.RCF.atomic import Eq, Ge, Gt, Le, Lt
from logic1.theories.RCF.term import Term
from logic1.theories.RCF.types import Formula


type SExpr = str | list[SExpr]
type Value = Term | Formula

_TOKEN = re.compile(r'\s+|;[^\n]*|"(?:[^"]|"")*"|\|[^|]*\||[()]|[^\s();"|]+')
_RELATIONS = {"=": Eq, "<": Lt, "<=": Le, ">": Gt, ">=": Ge}


def _sexpressions(source: str) -> list[SExpr]:
    """Read SMT-LIB S-expressions, including quoted metadata fields.
    """
    forms: list[SExpr] = []
    stack = [forms]
    position = 0
    for match in _TOKEN.finditer(source):
        if match.start() != position:
            raise ValueError(f"Unexpected SMT-LIB text at offset {position}")
        position = match.end()
        token = match.group()
        if token.isspace() or token.startswith(";"):
            continue
        if token == "(":
            child: list[SExpr] = []
            stack[-1].append(child)
            stack.append(child)
        elif token == ")":
            if len(stack) == 1:
                raise ValueError("Unmatched closing parenthesis")
            stack.pop()
        else:
            stack[-1].append(token)
    if position != len(source) or len(stack) != 1:
        raise ValueError("Incomplete SMT-LIB input")
    return forms


def _term(expr: SExpr, symbols: dict[str, Value]) -> Term:
    value = _value(expr, symbols)
    if not isinstance(value, Term):
        raise ValueError(f"Expected a real polynomial, got {expr!r}")
    return value


def _formula(expr: SExpr, symbols: dict[str, Value]) -> Formula:
    value = _value(expr, symbols)
    if isinstance(value, Term):
        raise ValueError(f"Expected a Boolean formula, got {expr!r}")
    return value


def _value(expr: SExpr, symbols: dict[str, Value]) -> Value:
    """Translate a well-sorted polynomial or Boolean expression.

    Let bindings are evaluated in the outer environment before any new
    binding becomes visible. Nested lets therefore implement shadowing
    without altering that outer environment.
    """
    if isinstance(expr, str):
        if expr in symbols:
            return symbols[expr]
        if expr == "true":
            return cast(Formula, T)
        if expr == "false":
            return cast(Formula, F)
        try:
            return Term(Fraction(expr))
        except ValueError as exc:
            raise ValueError(f"Unknown SMT-LIB symbol {expr!r}") from exc
    if not expr or not isinstance(expr[0], str):
        raise ValueError(f"Invalid SMT-LIB expression {expr!r}")
    operator, *operands = expr
    if operator == "let" and len(operands) == 2:
        bindings, body = operands
        if not isinstance(bindings, list) or not bindings:
            raise ValueError("Expected nonempty let bindings")
        values: dict[str, Value] = {}
        for binding in bindings:
            if not isinstance(binding, list) or len(binding) != 2:
                raise ValueError(f"Invalid let binding {binding!r}")
            name, expression = binding
            if not isinstance(name, str) or name in values:
                raise ValueError(f"Invalid or repeated let name {name!r}")
            values[name] = _value(expression, symbols)
        return _value(body, symbols | values)
    if operator in {"+", "-", "*"} and operands:
        terms = [_term(operand, symbols) for operand in operands]
        if operator == "+":
            result = Term(0)
            for term in terms:
                result += term
            return result
        if operator == "-":
            result = -terms[0] if len(terms) == 1 else terms[0]
            for term in terms[1:]:
                result -= term
            return result
        result = Term(1)
        for term in terms:
            result *= term
        return result
    if operator == "/" and len(operands) >= 2:
        result = _term(operands[0], symbols)
        for operand in operands[1:]:
            denominator = _term(operand, symbols)
            if not denominator.is_constant() or not denominator.as_constant():
                raise ValueError("Only division by nonzero constants is supported")
            result /= denominator
        return result
    if operator == "and":
        return And(*(_formula(operand, symbols) for operand in operands))
    if operator == "or":
        return Or(*(_formula(operand, symbols) for operand in operands))
    if operator == "not" and len(operands) == 1:
        return Not(_formula(operands[0], symbols))
    if operator in _RELATIONS and len(operands) == 2:
        left, right = (_term(operand, symbols) for operand in operands)
        return _RELATIONS[operator](left - right, 0)
    raise ValueError(f"Unsupported SMT-LIB operator {operator!r}")


def _load(path: Path) -> Formula:
    """Read assertions, evaluating definitions in their declaration order.
    """
    expressions = _sexpressions(path.read_text(encoding="utf-8"))
    names: list[str] = []
    reserved_names: set[str] = set()
    declarations: dict[int, str] = {}
    for index, expression in enumerate(expressions):
        if not isinstance(expression, list) or not expression:
            raise ValueError(f"Invalid SMT-LIB command in {path.name}")
        command, *arguments = expression
        if command == "declare-fun" and len(arguments) == 3:
            name, parameters, sort = arguments
            if not isinstance(name, str) or parameters != [] or sort != "Real":
                raise ValueError(f"Unsupported declaration in {path.name}")
            names.append(name)
            declarations[index] = name
        elif command == "declare-const" and len(arguments) == 2:
            name, sort = arguments
            if not isinstance(name, str) or sort != "Real":
                raise ValueError(f"Unsupported declaration in {path.name}")
            names.append(name)
            declarations[index] = name
        elif command == "define-fun" and len(arguments) == 4:
            name, parameters, sort, _ = arguments
            if not isinstance(name, str) or parameters != [] or sort not in ("Real", "Bool"):
                raise ValueError(f"Unsupported definition in {path.name}")
        else:
            continue
        if name in reserved_names:
            raise ValueError(f"Duplicate SMT-LIB name {name!r}")
        reserved_names.add(name)

    # Sage requires alphanumeric variable names. Reserve source declarations
    # and definitions before choosing names for unsupported symbols.
    used_names = set(reserved_names)
    variables: dict[str, Term] = {}
    for index, name in enumerate(names):
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            safe_name = name
        else:
            safe_name = f"smtlib_var_{index}"
            while safe_name in used_names:
                safe_name += "_"
        used_names.add(safe_name)
        variables[name] = VV[safe_name]

    symbols: dict[str, Value] = {}
    assertions: list[Formula] = []
    logic: str | None = None
    for index, expression in enumerate(expressions):
        assert isinstance(expression, list)
        command, *arguments = expression
        if index in declarations:
            name = declarations[index]
            symbols[name] = variables[name]
        elif command == "define-fun" and len(arguments) == 4:
            name, _, sort, body = arguments
            assert isinstance(name, str)
            value = _value(body, symbols)
            if (sort == "Real") != isinstance(value, Term):
                raise ValueError(f"Incorrect sort in definition {name!r}")
            symbols[name] = value
        elif command == "set-logic" and len(arguments) == 1:
            logic = str(arguments[0])
        elif command == "assert" and len(arguments) == 1:
            assertions.append(_formula(arguments[0], symbols))
        elif command in {"set-info", "check-sat", "exit"}:
            continue
        else:
            raise ValueError(f"Unsupported SMT-LIB command {command!r} in {path.name}")
    if logic != "QF_NRA" or not assertions:
        raise ValueError(f"Expected asserted QF_NRA formula in {path.name}")
    return And(*assertions)


_UNCU = Path("20220314-Uncu/"
             "Rational_Function_Proof_Calls_With_Disjunctive_Denominator_Simplifications")
_HEIZMANN = Path("20170501-Heizmann-UltimateInvariantSynthesis")
_ULTIMATE = Path("UltimateAutomizer")
_POLYPAVER = Path("meti-tarski/polypaver/bench-sqrt-3d")
_PINE = Path("20200911-Pine")
_GEOGEBRA = Path("20211101-Geogebra")
_ECONOMICS = Path("20180501-Economics-Mulligan")
_KISSING = Path("kissing")
_PPDA = Path("20240407-pPDA-Chiari-Pontiggia-Winkler/certificate_no_hints")
_MBO = Path("20161105-Sturm-MBO")
_MGC = Path("20161105-Sturm-MGC")

# (case ID, archive path below QF_NRA, expected input atom count)
SMTLIB_CASES: tuple[tuple[str, Path, int], ...] = (
    ("mitrinovic-4-14", _UNCU / "DDC_ProveIneq_Mitrinovic_4_14.smt2", 16),
    ("mitrinovic-3-29", _UNCU / "DDC_ProveIneq_Mitrinovic_3_29.smt2", 20),
    ("knopp-schur-b", _UNCU / "DDC_ProveIneq_ISSAC05_KnoppSchur_b.smt2", 29),
    ("half-2", _HEIZMANN / "half_2.i_3_6_2.bpl_3.smt2", 81),
    ("issac10-ex4h", _UNCU / "DDC_ProveIneq_ISSAC10_ex4h.smt2", 39),
    ("levin-rhsc", _UNCU / "DDC_ProveIneq_ISSAC05_LevinRHSc.smt2", 46),
    ("sumcracker-14d", _UNCU / "DDC_ProveIneq_JSC_SUMCracker_14d.smt2", 44),
    ("sumcracker-14e", _UNCU / "DDC_ProveIneq_JSC_SUMCracker_14e.smt2", 54),
    ("lex-index", _ULTIMATE / "InVarSynth_LexIndexValue-Pointer-Reduced.bpl_Iteration1_0.smt2",
     46),
    ("polypaver-0326", _POLYPAVER / "polypaver-bench-sqrt-3d-chunk-0326.smt2", 19),
    ("pine-quadratic", _PINE / "1599121886379408000.smt2", 12),
    ("pine-pendulum", _PINE / "1599121909151965000.smt2", 16),
    ("pentagon-diagonal", _GEOGEBRA / "RegPentagon-Diagonal_Side.smt2", 12),
    ("triangle-sym321", _GEOGEBRA / "Triangle-SymmHomPoly_3-2-1.smt2", 9),
    ("economics-0016c", _ECONOMICS / "MulliganEconomicsModel0016c.smt2", 16),
    ("economics-0063c", _ECONOMICS / "MulliganEconomicsModel0063c.smt2", 11),
    ("issac10-ex3e", _UNCU / "DDC_ProveIneq_ISSAC10_ex3e.smt2", 27),
    ("issac10-ex3f", _UNCU / "DDC_ProveIneq_ISSAC10_ex3f.smt2", 32),
    ("sumcracker-14b", _UNCU / "DDC_ProveIneq_JSC_SUMCracker_14b.smt2", 24),
    ("sumcracker-14c", _UNCU / "DDC_ProveIneq_JSC_SUMCracker_14c.smt2", 34),
    ("max-in-array", _HEIZMANN / "standard_maxInArray_ground.i_3_2_2.bpl_5.smt2", 111),
    ("partial-init", _HEIZMANN / "standard_partial_init_ground.i_3_2_2.bpl_5.smt2", 125),
    ("hhk2008", _HEIZMANN / "hhk2008.c.i_3_3_2.bpl_5.smt2", 138),
    ("two-index", _HEIZMANN / "standard_two_index_01.i_3_2_2.bpl_5.smt2", 126),
    ("economics-1001c", _ECONOMICS / "MulliganEconomicsModel1001c.smt2", 40),
    ("kissing-4-14", _KISSING / "kissing_4_14.smt2", 105),
    ("kissing-4-25", _KISSING / "kissing_4_25.smt2", 325),
    ("ppda-mod5", _PPDA / "mod5_POMC_no_hints.smt2", 243),
    ("ppda-mod7", _PPDA / "mod7_POMC_no_hints.smt2", 655),
    ("mbo-e7e10", _MBO / "mbo_E7E10.smt2", 8),
    ("kochanski", _GEOGEBRA / "KochanskiPiApproximation-RegDodecagon.smt2", 29),
    ("mgc-10", _MGC / "mgc_10.smt2", 13),
    # Further inputs with individually timed full DNF attempts.
    ("mbo-e11e14", _MBO / "mbo_E11E14.smt2", 8),
    ("mbo-e14", _MBO / "mbo_E14.smt2", 7),
    ("mbo-e4e11", _MBO / "mbo_E4E11.smt2", 8),
    ("cggmp2005", _HEIZMANN / "cggmp2005_variant.c.i_3_3_2.bpl_5.smt2", 133),
    ("const1", _HEIZMANN / "const1.i_3_6_2.bpl_5.smt2", 117),
    ("ddlm2013", _HEIZMANN / "ddlm2013.i_3_4_2.bpl_3.smt2", 336),
    ("gj2007", _HEIZMANN / "gj2007.c.i_4_3_3.bpl_5.smt2", 253),
    ("gr2006", _HEIZMANN / "gr2006.c.i_4_4_3.bpl_5.smt2", 219),
    ("nested-c", _HEIZMANN / "nested.c_4_2_2.bpl_3.smt2", 206),
    ("nested-i", _HEIZMANN / "nested.i_4_5_3.bpl_3.smt2", 245),
    ("simple1", _HEIZMANN / "simple1.i_3_6_2.bpl_9.smt2", 172),
    ("all-diff2", _HEIZMANN / "standard_allDiff2_ground.i_4_4_3.bpl_3.smt2", 594),
    ("compare-modified", _HEIZMANN / "standard_compareModified_ground.i_4_2_2.bpl_5.smt2", 189),
    ("password", _HEIZMANN / "standard_password_ground.i_4_2_2.bpl_5.smt2", 189),
    ("running", _HEIZMANN / "standard_running.i_4_2_2.bpl_5.smt2", 211),
    ("economics-0052c", _ECONOMICS / "MulliganEconomicsModel0052c.smt2", 23),
    ("economics-0086c", _ECONOMICS / "MulliganEconomicsModel0086c.smt2", 39),
    ("right-triangle", _GEOGEBRA / "IsoRightTriangle-Bottema1_4a.smt2", 9),
    ("hendecagon", _GEOGEBRA / "RegHendecagon-Diagonal_Side.smt2", 24),
    ("triangle-bottema", _GEOGEBRA / "Triangle-Bottema5_3.smt2", 13),
    ("ppda-escape25", _PPDA / "escape25_POMC_no_hints.smt2", 545),
    ("ppda-schelling", _PPDA / "schelling_POMC_no_hints.smt2", 708),
    ("beyer-figure4", _ULTIMATE
     / "InVarSynth_BeyerHenzingerMajumdarRybalchenko-PLDI2007-Figure4.bpl_Iteration2_0.smt2", 88),
    ("bug-hoare", _ULTIMATE / "InVarSynth_BugHoareAnnotationWithMinimiation.bpl_Iteration1_0.smt2",
     100),
    ("impulse", _ULTIMATE / "InVarSynth_ImpulseTest.bpl_Iteration2_0.smt2", 108),
    ("monadic-invariant", _ULTIMATE / "InVarSynth_MonadicInvariant01.bpl_Iteration1_0.smt2", 46),
    ("nutz-transform05", _ULTIMATE / "InVarSynth_NutzTransformation05.c_Iteration1_0.smt2", 87),
    ("bugtest1", _ULTIMATE / "InVarSynth_bugtest1.bpl_Iteration2_0.smt2", 89),
    ("threadpooling", _ULTIMATE / "InVarSynth_threadpooling_out2.mover.bpl_Iteration2_0.smt2",
     147),
    ("kissing-3-9", _KISSING / "kissing_3_9.smt2", 45),
)


def load_input(relative_path: Path) -> Formula:
    """Translate one selected original file into a Logic1 formula.
    """
    return _load(Path(__file__).with_name("inputs") / relative_path)
