"""Selected QF_NRA inputs from the SMT-LIB 2025 non-incremental release.

Source: https://doi.org/10.5281/zenodo.16740866 (SMT-LIB Initiative).
The original files are in ``inputs``; ``NOTES.md`` records their individual
contributors, licenses, and selection criteria. Each worker builds only its
own Logic1 formula before the timed benchmark call.

This reader implements only the SMT-LIB constructs used by the selected
files. It rejects other constructs rather than changing their meaning.
"""

from fractions import Fraction
from pathlib import Path
import re

from logic1.firstorder import And, Or
from logic1.theories.RCF import VV
from logic1.theories.RCF.atomic import Eq, Ge, Gt, Le, Lt
from logic1.theories.RCF.term import Term, Variable
from logic1.theories.RCF.types import Formula


type SExpr = str | list[SExpr]

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


def _term(expr: SExpr, symbols: dict[str, Variable]) -> Term:
    if isinstance(expr, str):
        if expr in symbols:
            return symbols[expr]
        try:
            return Term(Fraction(expr))
        except ValueError as exc:
            raise ValueError(f"Unknown SMT-LIB term {expr!r}") from exc

    if not expr or not isinstance(expr[0], str):
        raise ValueError(f"Invalid SMT-LIB term {expr!r}")
    operator, *operands = expr
    terms = [_term(operand, symbols) for operand in operands]
    if operator == "+" and terms:
        result = Term(0)
        for term in terms:
            result += term
        return result
    if operator == "-" and terms:
        result = -terms[0] if len(terms) == 1 else terms[0]
        for term in terms[1:]:
            result -= term
        return result
    if operator == "*" and terms:
        result = Term(1)
        for term in terms:
            result *= term
        return result
    raise ValueError(f"Unsupported SMT-LIB term operator {operator!r}")


def _formula(expr: SExpr, symbols: dict[str, Variable]) -> Formula:
    if isinstance(expr, str):
        raise ValueError(f"Unsupported SMT-LIB formula {expr!r}")
    if not expr or not isinstance(expr[0], str):
        raise ValueError(f"Invalid SMT-LIB formula {expr!r}")
    operator, *operands = expr
    if operator == "and":
        return And(*(_formula(operand, symbols) for operand in operands))
    if operator == "or":
        return Or(*(_formula(operand, symbols) for operand in operands))
    if operator in _RELATIONS and len(operands) == 2:
        left, right = (_term(operand, symbols) for operand in operands)
        return _RELATIONS[operator](left - right, 0)
    raise ValueError(f"Unsupported SMT-LIB formula operator {operator!r}")


def _load(path: Path) -> Formula:
    names: list[str] = []
    assertions: list[SExpr] = []
    logic: str | None = None
    for expression in _sexpressions(path.read_text(encoding="utf-8")):
        if not isinstance(expression, list) or not expression:
            raise ValueError(f"Invalid SMT-LIB command in {path.name}")
        command, *arguments = expression
        if command == "set-logic" and len(arguments) == 1:
            logic = str(arguments[0])
        elif command == "declare-fun" and len(arguments) == 3:
            name, parameters, sort = arguments
            if not isinstance(name, str) or parameters != [] or sort != "Real":
                raise ValueError(f"Unsupported declaration in {path.name}")
            names.append(name)
        elif command == "assert" and len(arguments) == 1:
            assertions.append(arguments[0])
        elif command in {"set-info", "check-sat", "exit"}:
            continue
        else:
            raise ValueError(f"Unsupported SMT-LIB command {command!r} in {path.name}")
    if logic != "QF_NRA" or not assertions:
        raise ValueError(f"Expected asserted QF_NRA formula in {path.name}")
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate SMT-LIB declaration in {path.name}")

    # Sage requires alphanumeric variable names. Rename only unsupported
    # symbols, choosing names that cannot collide with source declarations.
    used_names = set(names)
    symbols: dict[str, Variable] = {}
    for index, name in enumerate(names):
        if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
            safe_name = name
        else:
            safe_name = f"smtlib_var_{index}"
            while safe_name in used_names:
                safe_name += "_"
        used_names.add(safe_name)
        symbols[name] = VV[safe_name]
    return And(*(_formula(assertion, symbols) for assertion in assertions))


_UNCU = Path("20220314-Uncu/"
             "Rational_Function_Proof_Calls_With_Disjunctive_Denominator_Simplifications")
_HEIZMANN = Path("20170501-Heizmann-UltimateInvariantSynthesis")

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
)


def load_input(relative_path: Path) -> Formula:
    """Translate one selected original file into a Logic1 formula.
    """
    return _load(Path(__file__).with_name("inputs") / relative_path)
