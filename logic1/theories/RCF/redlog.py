import subprocess
from typing import Final, Iterable, Optional

import logic1.firstorder as firstorder
from logic1.support.excepthook import NoTraceException
from logic1.theories.RCF.term import Variable
from logic1.theories.RCF.atomic import AtomicFormula
from logic1.theories.RCF.qe import Generic
from logic1.theories.RCF.types import Formula

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

algebraic operator vv;

procedure privv(v); <<
    prin2!* "VV['";
    prin2!* cadr v;
    prin2!* "']";
    nil
>>;

put('vv, 'prifn, 'privv);

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
    % Convert the Lisp Prefix Form of a formula to a Python string.
    begin scalar map1, map2, map3, map4, op, lhs, rhs, argl, nargl, result, v, m, w;
        map1 := '((true . "T") (false . "F"));
        map2 := '((equal . "==") (neq . "!=") (lessp . "<")
                  (greaterp . ">") (leq . "<=")  (geq . ">="));
        map3 := '((and . "And") (or . "Or") (impl . "Implies")
                  (equiv . "Equivalent") (not . "Not"));
        map4 := '((ex . "Ex") (all . "All"));
        op := if atom f then f else car f;
        if w := atsoc(op, map1) then
            return cdr w;
        argl := cdr f;
        if op eq 'repl then <<
            op := 'impl;
            argl := reverse argl;
        >>;
        if w := atsoc(op, map2) then <<
            lhs := ioto_smaprin subsvv car argl;
            rhs := ioto_smaprin subsvv cadr argl;
            return lto_sconcat {lhs, " ", cdr w, " ", rhs}
        >>;
        if w := atsoc(op, map3) then <<
            nargl := for each arg in argl collect r2py_formula arg;
            result := lto_sconcat {cdr w, "(", pop nargl};
            for each arg in nargl do
                result := lto_sconcat {result, ", ", arg};
            result := lto_sconcat {result, ")"};
            return result
        >>;
        if w := atsoc(op, map4) then <<
            v := ioto_smaprin subsvv car argl;
            m := r2py_formula cadr argl;
            return lto_sconcat {cdr w, "(", v, ", ", m, ")"}
        >>
    end;

procedure subsvv(f); <<
    f := numr simp f;
    for each v in kernels f do
        f := numr subf(f, {v . {'vv, v}});
    prepf f
>>;

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


def _call_help(command: str) -> str:
    echo_string = _PRG.replace('"', r'\"')
    echo_string += '\nlinelength 80;'
    echo_string += f'\nlisp prin2 start!*; {command}; lisp prin2 end!*; quit;\n'
    cp = subprocess.run(f'echo "{echo_string}" | redcsl -w', shell=True, capture_output=True)
    return _unwrap(cp.stdout.decode())


def _call_redlog(command: str) -> str:
    echo_string = _PRG.replace('"', r'\"') + '\n' + f'wrap({command});\n' + '\n' + 'quit;\n'
    cp = subprocess.run(f'echo "{echo_string}" | redcsl -w', shell=True, capture_output=True)
    return _unwrap(cp.stdout.decode())


def _eval(s: str) -> object:
    from logic1.firstorder import Ex, All, Equivalent, Implies, And, Or, Not, T, F
    from logic1.theories.RCF.term import VV
    return eval(s, locals())


def _map_option(logic1_setting: bool, redlog_switch: str) -> str:
    return f'{"on" if logic1_setting else "off"} {redlog_switch};'


def _unwrap(s: str) -> str:
    start = s.find(_START) + len(_START)
    s = s[start:]
    end = s.find(_END)
    return s[:end]


# Using Redlog as a parser for Redlog input:

def to_logic1(s: str) -> Formula:
    """Return the Redlog formula in `s` as a Logic1 Formula. `s` must be valid
    Redlog input. `vars` is a superset of the variables in `s`.


    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, x, y = VV.get('a', 'b', 'x', 'y')
    >>> s = 'all(x, ex(y, x**2 + x * y + b > 0 and (x + a * y**2 + b < 0 or x + a * y**2 + b = 0)))'
    >>> redlog.to_logic1(s)
    All(x, Ex(y, And(x**2 + x*y + b > 0, Or(a*y**2 + b + x < 0, a*y**2 + b + x == 0))))
    """
    output = _call_redlog(f'<< r2py_formula ({s}) >>')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result


# Wrapped Redlog functions in alphabetical order:

def cnf(f: Formula, bnfsm: bool = False, bnfsac: bool = True) -> Formula:
    """Conjunctive normal form using the Redlog function `rlcnf
    <https://www.redlog.eu/documentation/service.php?key=cnf>`_.

    :param f:
      The input formula.

    :returns:
      A conjunctive normal form of `f`.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b = VV.get('a', 'b')
    >>> f = Or(a < 0, And(b != 0, a == 0), And(b > 0, a == 0), And(a > 0, b**2 - 4*a >= 0))
    >>> redlog.cnf(f)
    And(Or(b != 0, a <= 0),
        Or(b != 0, a < 0),
        Or(b > 0, a <= 0, b**2 - 4*a >= 0),
        Or(a <= 0, b**2 - 4*a >= 0))
    >>> redlog.cnf(f, bnfsm=True)
    And(Or(b != 0, a < 0), Or(a <= 0, b**2 - 4*a >= 0))
    """
    rl_switches = (f'{_map_option(bnfsac, "rlbnfsac")} '
                   f'{_map_option(bnfsm, "rlbnfsm")}')
    rl_f = f.as_redlog()
    output = _call_redlog(f'<< {rl_switches} r2py_formula rlcnf({rl_f}) >>')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result


def dnf(f: Formula, bnfsm: bool = False, bnfsac: bool = True) -> Formula:
    """Disjunctive normal form using the Redlog function `rldnf
    <https://www.redlog.eu/documentation/service.php?key=dnf>`_.

    :param f:
      The input formula.

    :returns:
      A conjunctive normal form of `f`.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b = VV.get('a', 'b')
    >>> f = And(a >= 0, Or(b == 0, a != 0), Or(b <= 0, a != 0), Or(a <= 0, b**2 - 4*a < 0))
    >>> redlog.dnf(f)
    Or(And(b == 0, a >= 0), And(b <= 0, a > 0, b**2 - 4*a < 0), And(a > 0, b**2 - 4*a < 0))
    >>> redlog.dnf(f, bnfsm=True)
    Or(And(b == 0, a >= 0), And(a > 0, b**2 - 4*a < 0))
    """
    rl_switches = (f'{_map_option(bnfsac, "rlbnfsac")} '
                   f'{_map_option(bnfsm, "rlbnfsm")}')
    rl_f = f.as_redlog()
    output = _call_redlog(f'<< {rl_switches} r2py_formula rldnf({rl_f}) >>')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result


def gqe(f: Formula, generic: Generic = Generic.FULL) -> tuple[list[AtomicFormula], Formula]:
    """Generic real quantifier elimination using the Redlog function `rlgqe
    <https://www.redlog.eu/documentation/service.php?key=rlgqe>`_.

    :param f:
      The input formula to which quantifier elimination will be applied.

    :returns:
      A pair `(assumptions, f')`. The formula `f'` is a quantifier-free
      equivalent of `f` modulo the `assumptions`. All assumptions are
      instances of :class:`.Ne`; if `generic=Generic.MONOMIAL`, then all left
      hand sides of assumptions are monomial .

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
    >>> redlog.gqe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=Generic.MONOMIAL)
    ([b != 0], Or(a + 1 == 0, 4*a*c - b**2 + 4*c <= 0))
    >>> redlog.gqe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=Generic.FULL)
    ([a + 1 != 0], 4*a*c - b**2 + 4*c <= 0)

    .. seealso::
      :meth:`.qe` with `generic` in :attr:`.Generic.FULL`, :attr:`.Generic.MONOMIAL`.
    """
    match generic:
        case Generic.NONE:
            raise NoTraceException('Generic.NONE is not supported - use redlog.qe instead')
        case Generic.MONOMIAL:
            rl_switches = 'off rlqegenct;'
        case Generic.FULL:
            rl_switches = 'on rlqegenct;'
        case _:
            assert False, generic
    rl_f = f.as_redlog()
    output = _call_redlog(f'<< {rl_switches} r2py_gqe rlgqe({rl_f}) >>')
    result = _eval(output)
    assert isinstance(result, tuple), result
    assert len(result) == 2, result
    assert isinstance(result[0], list), result
    assert all(isinstance(at, AtomicFormula) for at in result[0]), result
    assert isinstance(result[1], firstorder.Formula), result
    return result


def gsn(f: Formula, assume: Iterable[AtomicFormula] = [], form: str = 'auto',
        bnfsm=False, bnfsac: bool = True) -> Formula:
    """Real quantifier elimination using the Redlog function `rlgsn
    <https://www.redlog.eu/documentation/service.php?key=rlgsn>`_.

    :param f:
      The input formula to which quantifier elimination will be applied.

    :param assume:
      A list of atomic formulas that are assumed to hold. The return value
      is equivalent modulo those assumptions.

    :param form:
      Explicitly choose the normal form of the output. Possible arguments are
      the strings 'auto' (default), 'cnf', 'dnf'.

    :param bnfsm:
    :param bnfsac:
      Are passed on to CNF/DNF computation.

    :returns:
      A simplified equivalent of `f` modulo `assume`.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> x, y, z = VV.get('x', 'y', 'z')
    >>> redlog.gsn(Implies(And(x * y + 1 == 0, y * z + 1 == 0), x == z))
    T
    >>> a, b = VV.get('a', 'b')
    >>> f = Or(a < 0, And(b != 0, a == 0), And(b > 0, a == 0), And(a > 0, b**2 - 4*a >= 0))
    >>> redlog.gsn(f)
    Or(a < 0, And(b != 0, a == 0), And(a > 0, b**2 - 4*a >= 0))
    >>> redlog.gsn(f, form='cnf')
    And(Or(b != 0, a <= 0),
        Or(b != 0, a < 0),
        Or(b > 0, a <= 0, b**2 - 4*a >= 0),
        Or(a <= 0, b**2 - 4*a >= 0))
    """
    rl_switches = (f'{_map_option(bnfsac, "rlbnfsac")} '
                   f'{_map_option(bnfsm, "rlbnfsm")}')
    rl_f = f.as_redlog()
    rl_assume = '{' + ', '.join(atom.as_redlog() for atom in assume) + '}'
    output = _call_redlog(f'<< {rl_switches} r2py_formula rlgsn({rl_f}, {rl_assume}, {form}) >>')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result

def help(key: Optional[str] = None, developer: bool = False) -> None:
    """Raw access to the redlog help system. A call of this function corresponds
    to the question mark within Redlog. Call `rl_help()` without arguments to
    get started. Then, e.g., `?services` becomes `rl_help("services")`, and
    `?rlqe` becomes `rl_help("rlqe")` which displays information on `:meth:qe`.
    Note that all services and switches are prefixed with `rl` in Redlog and in
    the help system, but not in Logic1.
    """
    if key is None:
        key = "nil"
    else:
        key = "'" + key
    if developer is False:
        devp = "nil"
    else:
        devp = "t"
    output = _call_help(f"rl_help({key}, {devp})")
    print(output)

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
    >>> redlog.qe(All(x, Ex(y, x**2 + x*y + b > 0 and x + a*y**2 + b <= 0)))
    a < 0
    >>> redlog.qe(Ex(x, (a + 1) * x**2 + b * x + c == 0), [b != 0])
    Or(a + 1 == 0, 4*a*c - b**2 + 4*c <= 0)
    >>> redlog.qe(All(x, Ex(y, And(b + x**2 + x*y > 0, a*y**2 + b + x <= 0))))
    And(b > 0, a < 0)
    """
    rl_f = f.as_redlog()
    rl_assume = '{' + ', '.join(atom.as_redlog() for atom in assume) + '}'
    output = _call_redlog(f'r2py_formula rlqe({rl_f}, {rl_assume})')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result


def qea(f: Formula) -> list[tuple[Formula, list[str]]]:
    input = f.as_redlog()
    output = _call_redlog(f'r2py_qea rlqea {input}')
    result = _eval(output)
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
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result
