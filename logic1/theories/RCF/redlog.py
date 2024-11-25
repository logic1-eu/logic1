import subprocess
from typing import Final, Iterable

import logic1.firstorder as firstorder
from logic1.support.excepthook import NoTraceException
from logic1.theories.RCF.atomic import AtomicFormula, Variable
from logic1.theories.RCF.qe import GENERIC
from logic1.theories.RCF.typing import Formula

_START: Final = '889e0d7343405c079195e7b8903c8c9e'
_END: Final = 'b0061974914468de549a2af8ced10316'

_PRG: Final = """module redlog;

off1 'output;
off1 'nat;

linelength(2**24);

global '(start!* end!*);

start!* := "889e0d7343405c079195e7b8903c8c9e";
end!* := "b0061974914468de549a2af8ced10316";

rl_set '(r);

operator wrap;
procedure wrap(f); <<
    terpri();
    prin2 start!*;
    prin2 f;
    prin2 end!*;
    nil
>>;

operator r2py_qea;
procedure r2py_qea(l);
    % Convert the Lisp Prefix Form of an AM result of rlqea to a Python string.
    begin scalar res, pair, guard, ans;
        pop l;
        if l then <<
            pair := car l;
            pop pair;
            guard := r2py_formula car pair;
            ans := r2py_equation_list cadr pair;
            res := lto_sconcat {"[(", guard, ", ", ans, ")"};
            for each pair in cdr l do <<
                pop pair;
                guard := r2py_formula car pair;
                ans := r2py_equation_list cadr pair;
                res := lto_sconcat {res, ", (", guard, ", ", ans, ")"}
            >>
        >> else <<
            res := "[]"
        >>;
        res := lto_sconcat {res, "]"};
        return res
    end;

procedure r2py_equation_list(l);
    begin scalar equation, ans;
        pop l;
        if l then <<
            equation := car l;
            ans := lto_sconcat {"['", ioto_smaprin equation, "'"};
            for each equation in cdr l do
                ans := lto_sconcat {ans, ", '", ioto_smaprin equation, "'"};
            ans:= lto_sconcat {ans, "]"}
        >> else <<
            ans := "[]"
        >>;
        return ans
    end;

operator r2py_formula;
procedure r2py_formula(f);
    % Convert the Lisp Prefix Form a formula to a Python string.
    begin scalar map1, map2, map3, map4, op, lhs, rhs, argl, result, v, m, w;
        map1 := '((equal . "==") (neq . "!=") (lessp . "<")
                  (greaterp . ">") (leq . "<=")  (geq . ">="));
        map2 := '((true . "T") (false . "F"));
        map3 := '((and . "And") (or . "Or") (impl . "Implies")
                  (equiv . "Equivalent") (not . "Not"));
        map4 := '((ex . "Ex") (all . "All"));
        if rl_op f eq 'repl then
            f := rl_mk2('impl, rl_arg2r f, rl_arg2l f);
        op := rl_op f;
        if w := atsoc(op, map1) then <<
            lhs := ioto_smaprin rl_arg2l f;
            rhs := ioto_smaprin rl_arg2r f;
            return lto_sconcat {lhs, " ", cdr w, " ", rhs}
        >>;
        if w := atsoc(op, map2) then
            return cdr w;
        if w := atsoc(op, map3) then <<
            argl := for each arg in rl_argn f collect r2py_formula arg;
            result := lto_sconcat {cdr w, "(", pop argl};
            for each arg in argl do
                result := lto_sconcat {result, ", ", arg};
            result := lto_sconcat {result, ")"};
            return result
        >>;
        if w := atsoc(op, map4) then <<
            v := rl_var f;
            m := r2py_formula rl_mat f;
            return lto_sconcat {cdr w, "(", v, ", ", m, ")"}
        >>
    end;

operator r2py_gqe;
procedure r2py_gqe(l);
    % Convert the Lisp Prefix Form of an AM result of rlgqe to a Python string.
    begin scalar atoms, formula, th, res;
        pop l;
        atoms := cdr pop l;
        formula := pop l;
        if atoms then <<
            th := lto_sconcat {"[", r2py_formula car atoms};
            for each at in cdr atoms do
                th := lto_sconcat {th, ", ", r2py_formula at};
            th := lto_sconcat {th, "]"}
        >> else <<
            th := "[]"
        >>;
        res := r2py_formula formula;
        return lto_sconcat {"(", th, ", ", res, ")"}
    end;

endmodule;
"""


def _call_redlog(command: str) -> str:
    echo_string = _PRG.replace('"', r'\"') + '\n' + f'wrap({command});\n' + '\n' + 'quit;\n'
    cp = subprocess.run(f'echo "{echo_string}" | redcsl -w', shell=True, capture_output=True)
    return _unwrap(cp.stdout.decode())


def _eval(s: str, variables: set[Variable]) -> object:
    from logic1.firstorder import Ex, All, Equivalent, Implies, And, Or, Not, T, F  # noqa
    return eval(s, locals() | {str(v): v for v in variables})


def gqe(f: Formula, generic: GENERIC = GENERIC.FULL) -> tuple[list[AtomicFormula], Formula]:
    """Generic real quantifier elimination using the Redlog function `rlgqe
    <https://www.redlog.eu/documentation/service.php?key=rlgqe>`_.

    :param f:
      The input formula to which quantifier elimination will be applied.

    :returns:
      A pair `(assumptions, f')`. The formula `f'` is a quantifier-free
      equivalent of `f` modulo the `assumptions`. All assumptions are
      instances of :class:`.Ne`; if `generic=GENERIC.MONOMIAL`, then all left
      hand sides of assumptions are monomial .

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
    >>> redlog.gqe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=GENERIC.MONOMIAL)
    ([b != 0], Or(a + 1 == 0, 4*a*c - b^2 + 4*c <= 0))
    >>> redlog.gqe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=GENERIC.FULL)
    ([a + 1 != 0], 4*a*c - b^2 + 4*c <= 0)

    .. seealso::
      :meth:`.qe` with `generic` in :attr:`.GENERIC.FULL`, :attr:`.GENERIC.MONOMIAL`.
    """
    match generic:
        case GENERIC.NONE:
            raise NoTraceException('GENERIC.NONE is not supported - use redlog.qe instead')
        case GENERIC.MONOMIAL:
            rl_switches = 'off rlqegenct;'
        case GENERIC.FULL:
            rl_switches = 'on rlqegenct;'
        case _:
            assert False, generic
    rl_f = f.as_redlog()
    output = _call_redlog(f'<< {rl_switches} r2py_gqe rlgqe({rl_f}) >>')
    result = _eval(output, _variables(f))
    assert isinstance(result, tuple), result
    assert len(result) == 2, result
    assert isinstance(result[0], list), result
    assert all(isinstance(at, AtomicFormula) for at in result[0]), result
    assert isinstance(result[1], firstorder.Formula), result
    return result


def _map_option(logic1_setting: bool, redlog_switch: str) -> str:
    return f'{"on" if logic1_setting else "off"} {redlog_switch};'


def qe(f: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
    """Real quantifier elimination using the Redlog function `rlqe
    <https://www.redlog.eu/documentation/service.php?key=rlqe>`_.

    :param f:
      The input formula to which quantifier elimination will be applied.

    :param assume:
      A list of atomic formulas that are assumed to hold. The return value
      is equivalent modulo those assumptions.

    :returns:
      A quantifier-free equivalent of `f` modulo `assume`.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x, y = VV.get('a', 'b', 'c', 'x', 'y')
    >>> redlog.qe(All(x, Ex(y, x**2 + x*y + b > 0 and x + a*y**2 + b <= 0)));
    a < 0
    >>> redlog.qe(Ex(x, (a + 1) * x**2 + b * x + c == 0), [b != 0])
    Or(a + 1 == 0, 4*a*c - b^2 + 4*c <= 0)
    >>> redlog.qe(All(x, Ex(y, And(b + x**2 + x*y > 0, a*y**2 + b + x <= 0))))
    And(b > 0, a < 0)
    """
    rl_f = f.as_redlog()
    rl_assume = '{' + ', '.join(atom.as_redlog() for atom in assume) + '}'
    output = _call_redlog(f'r2py_formula rlqe({rl_f}, {rl_assume})')
    result = _eval(output, _variables(f))
    assert isinstance(result, firstorder.Formula), result
    return result


def qea(f: Formula) -> list[tuple[Formula, list[str]]]:
    input = f.as_redlog()
    output = _call_redlog(f'r2py_qea rlqea {input}')
    result = _eval(output, _variables(f))
    assert isinstance(result, list), result
    assert all(isinstance(guard, firstorder.Formula) for guard, _ in result), result
    assert all(isinstance(ans, list) for _, ans in result), result
    assert all(isinstance(s, str) for _, ans in result for s in ans), result
    return result


def simplify(f: Formula, assume: Iterable[AtomicFormula] = [],
             explode_always: bool = True, prefer_order: bool = True, prefer_weak: bool = False) \
        -> Formula:
    """Simplification using the Redlog function `rlsimpl
    <https://www.redlog.eu/documentation/service.php?key=simpl>`_.

    :param f:
      The input formula to which quantifier elimination will be applied.

    :param assume:
      A list of atomic formulas that are assumed to hold. The return value
      is equivalent modulo those assumptions.

    :returns:
      A simplified equivalent of `f` modulo `assume`.
    """
    rl_switches = (f'{_map_option(explode_always, "rlsiexpla")} '
                   f'{_map_option(prefer_order, "rlsipo")} '
                   f'{_map_option(prefer_weak, "rlsipw")}')
    rl_f = f.as_redlog()
    rl_assume = '{' + ', '.join(atom.as_redlog() for atom in assume) + '}'
    output = _call_redlog(f'<< {rl_switches} r2py_formula rlsimpl({rl_f}, {rl_assume}) >>')
    result = _eval(output, _variables(f))
    assert isinstance(result, firstorder.Formula), result
    return result


def _unwrap(s: str) -> str:
    start = s.find(_START) + len(_START)
    s = s[start:]
    end = s.find(_END)
    return s[:end]


def _variables(f: Formula) -> set[Variable]:
    return set(f.fvars()).union(f.qvars())
