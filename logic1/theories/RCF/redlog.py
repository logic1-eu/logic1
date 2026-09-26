"""The use of the Redlog interface requires the computer algebra system Reduce.
Binary distributions are available on `SourceForge
<https://sourceforge.net/projects/reduce-algebra/>`_. The executable
:file:`redcsl` must be in the system path. Test the following in your shell:

.. code-block::

    $ redcsl
    Reduce (CSL, rev 6864), 24-Aug-2024 ...

    1: rlset reals;
    Redlog Revision 6618 of 2023-10-06, 06:18:51Z
    (c) 1992-2023 T. Sturm and A. Dolzmann (www.redlog.eu)
    type ?; for help

    {}

    2: rlqe ex(x, a*x + b = 0);

    b = 0 or a <> 0

    3: quit;

This module allows you to perform the same quantifier elimination in Redlog
from within Python:

>>> from logic1.interactive.RCF import *
>>> result = redlog.qe(Ex(x, a * x + b == 0))
>>> result
Or(b == 0, a != 0)

Both, the argument of :func:`.redlog.qe` and the result are instances of
:class:`Formula <.RCF.types.Formula>`.
"""

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
    if cp.returncode == 127:
        raise RuntimeError('redcsl not found. Install Reduce and make sure redcsl is in your PATH.')
    if cp.returncode != 0:
        raise RuntimeError(f'redcsl failed with exit code {cp.returncode}:\n'
                           f'{cp.stderr.decode()}')
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


# Wrapped Redlog functions in alphabetical order:

def cnf(f: Formula, bnfsm: bool = False, bnfsac: bool = True) -> Formula:
    """Return a conjunctive normal form of ``f``, using the Redlog function
    :redlog:`rlcnf`.

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
    """Return a disjunctive normal form of ``f``, using the Redlog function
    :redlog:`rldnf`.

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
    """Apply generic real quantifier elimination to ``f``, using the Redlog
    function :redlog:`rlgqe`.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
    >>> redlog.gqe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=Generic.MONOMIAL)
    ([b != 0], Or(a + 1 == 0, 4*a*c - b**2 + 4*c <= 0))
    >>> redlog.gqe(Ex(x, (a + 1) * x**2 + b * x + c == 0), generic=Generic.FULL)
    ([a + 1 != 0], 4*a*c - b**2 + 4*c <= 0)

    .. seealso::
      The Logic1 function :func:`qe() <.RCF.qe.qe>` with the option
      ``generic=Generic.FULL`` or ``generic=Generic.MONOMIAL``.
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
    """Apply Gröbner simplification to a Boolean normal form of ``f``, using the
    Redlog function :redlog:`rlgsn`.

    The argument ``assume`` is a list of atomic formulas that are assumed to
    hold. The argument ``form`` allows to choose the normal form of the output,
    where possible arguments are the strings ``'auto'`` (default), ``'cnf'``,
    ``'dnf'``. The options ``bnfsm`` and ``bnfsac`` set the Redlog switches
    ``rlbnfsm`` and ``rlbnfsac``, respectively, which control the behavior of
    the Boolean normal form computation. Returns a simplified equivalent of
    ``f`` modulo ``assume``.

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

    .. seealso::
      The Logic1 function :func:`gsimplify() <.RCF.gsimplify.gsimplify>`.
    """
    rl_switches = (f'{_map_option(bnfsac, "rlbnfsac")} '
                   f'{_map_option(bnfsm, "rlbnfsm")}')
    rl_f = f.as_redlog()
    rl_assume = '{' + ', '.join(atom.as_redlog() for atom in assume) + '}'
    output = _call_redlog(f'<< {rl_switches} r2py_formula rlgsn({rl_f}, {rl_assume}, {form}) >>')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result

def qe(f: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
    """Apply real quantifier elimination to ``f``, using the Redlog function
    :redlog:`rlqe`.

    The argument ``assume`` is a list of atomic formulas that are assumed to
    hold. The return value is equivalent to ``f`` modulo the ``assumptions``.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x, y = VV.get('a', 'b', 'c', 'x', 'y')
    >>> redlog.qe(Ex(x, (a + 1) * x**2 + b * x + c == 0), [b != 0])
    Or(a + 1 == 0, 4*a*c - b**2 + 4*c <= 0)
    >>> redlog.qe(All(x, Ex(y, And(x**2 + x*y + b > 0, x + a*y**2 + b <= 0))))
    And(b > 0, a < 0)

    .. seealso::
      The Logic1 function :func:`qe <.RCF.qe.qe>` with the default option
      ``generic=Generic.NONE``.
    """
    rl_f = f.as_redlog()
    rl_assume = '{' + ', '.join(atom.as_redlog() for atom in assume) + '}'
    output = _call_redlog(f'r2py_formula rlqe({rl_f}, {rl_assume})')
    result = _eval(output)
    assert isinstance(result, firstorder.Formula), result
    return result


def qea(f: Formula) -> list[tuple[Formula, list[str]]]:
    """Apply extended real quantifier elimination to ``f``, using the Redlog
    function :redlog:`rlqea`.

    The result is a list of pairs ``(f', answer)``, where each ``answer`` is the
    unparsed string containing the corresponding generalized term returned by
    Redlog.

    >>> from logic1 import *
    >>> from logic1.theories.RCF import *
    >>> a, b, c, x = VV.get('a', 'b', 'c', 'x')
    >>> redlog.qea(Ex(x, a * x**2 + b * x + c == 0))
    [(And(c == 0, b == 0, a == 0), ['x = infinity1']),
     (And(a != 0, 4*a*c - b**2 <= 0), ['x = ( - sqrt( - 4*a*c + b**2) - b)/(2*a)']),
     (And(a != 0, 4*a*c - b**2 <= 0), ['x = (sqrt( - 4*a*c + b**2) - b)/(2*a)']),
     (And(b != 0, a == 0), ['x = ( - c)/b'])]
    """
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
    """Simplify ``f``, using the Redlog function `rlsimpl`.

    The argument ``assume`` is a list of atomic formulas that are assumed to
    hold. The result is equivalent to ``f`` modulo ``assume``.

    .. seealso::
      The Logic1 function :func:`simplify() <.RCF.simplify.simplify>`.
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


# Using Redlog as a parser for Redlog input:

def to_logic1(s: str) -> Formula:
    """Parse the string ``s`` as a Redlog formula and return it as a Logic1
    :class:`Formula <.RCF.types.Formula>`.

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


# Accessing Redlog Help

def help(key: Optional[str] = None, developer: bool = False) -> None:
    """Raw access to the redlog help system.

    A call of this function corresponds to the question mark within Redlog.
    Note that all Redlog functions and switches are prefixed with ``rl`` inside
    Redlog and in the help system, but not in Logic1.

    >>> from logic1.theories.RCF import redlog
    >>> redlog.help()  # doctest: +ELLIPSIS
    <BLANKLINE>
    REDLOG BUILTINS
        ?, all, and, ball, bex, equiv, ex, false, impl, mkand, mkor, not, or, repl,
        rlabout, rlset, true
    <BLANKLINE>
    REDLOG SERVICES
        rl1equation, rlall, rlatl, rlatml, rlatnum, rlbvarl, rlcad, rlcadporder,
        rlcadproj, rlcnf, rldecdeg, rldecdeg1, rldepth, rldima, rldnf, rldpep,
        rldump, rlenf, rlex, rlex2, rlexpand, rlexpanda, rlexplats, rlfvarl,
        ...
        rlsimplbasic, rlslfq, rlsmt2read, rlsmtqe, rlstex, rlstruct, rlsymbolify,
        rltab, rltan2, rlterml, rltermml, rlthsimpl, rltnf, rltropsat, rlvarl,
        rlvcreduce, rlvsl, rlwqe, rlwqea, rlxqe, rlxqea
    <BLANKLINE>
    REDLOG TYPES
        Any, Assignment/1, Atom, Enum/n, Flag, Formula, Integer, List/1, List5/5,
        LPolyQ, MList/1, Pair/2, Rational, String, Switch, Term, Triplet/3,
        TruthValue, Variable, Void
    <BLANKLINE>
    REDLOG KEYWORDS
        activity, auto, cnf, dfg, dlcs, dnf, mathematica, qepcad, sat, slfq, smt2,
        unknown, unsat, zmom
    <BLANKLINE>
    SEE ALSO
        ?builtins   more information on builtins
        ?services   more information on services
        ?types      more information on types
        ?X          for a specific service, type, or switch X
    <BLANKLINE>

    >>> redlog.help('services')  # doctest: +ELLIPSIS
    <BLANKLINE>
    REDLOG SERVICES
        rl1equation     equivalent DNF with one relevant equation in each branch (DCFSF)
        rlall           universal closure
        rlatl           set of contained atomic formulas
        ...
        rlwqea          weak quantifier elimination with answer
        rlxqe           weakly parametric linear quantifier elimination
        rlxqea          weakly parametric linear quantifier elimination with answer
    <BLANKLINE>
    SEE ALSO
        ?X              for a specific service X
    <BLANKLINE>

    >>> redlog.help('rlqe')  # doctest: +ELLIPSIS
    <BLANKLINE>
    SYNOPSIS
        rlqe(formula: Formula, assume = {}: List(Atom))
    <BLANKLINE>
    DESCRIPTION
        quantifier elimination
    <BLANKLINE>
    RETURNS
        Formula
    <BLANKLINE>
    ARGUMENTS
        formula    first-order input formula
        assume     atomic input assumptions
    <BLANKLINE>
    SEE ALSO
        rlcad      cylindrical algebraic decomposition
        rlgcad     generic cylindrical algebraic decomposition
        rlghqe     generic Hermitian quantifier elimination
        ...
        rlqea      quantifier elimination with answer
        rlqeipo    quantifier elimination in position
        rlqews     quantifier elimination with selection
    <BLANKLINE>
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
