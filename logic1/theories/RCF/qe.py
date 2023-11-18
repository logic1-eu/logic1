# mypy: strict_optional = False

"""Real quantifier elimination by virtual substitution.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import auto, Enum
import logging  # noqa
from sage.rings.fraction_field import FractionField  # type: ignore
from sage.rings.integer_ring import ZZ  # type: ignore
from time import time
from typing import Optional, TypeAlias

from ...firstorder import (
    All, And, AtomicFormula, F, _F, Formula, Not, Or, QuantifiedFormula, T)
from .rcf import (
    RcfAtomicFormula, RcfAtomicFormulas, ring, Term, Variable, Eq, Ne, Ge, Le,
    Gt, Lt)
from .pnf import pnf as _pnf
from .simplify import simplify as _simplify

from ...support.tracing import trace  # noqa


Quantifier: TypeAlias = type[QuantifiedFormula]
QuantifierBlock: TypeAlias = tuple[Quantifier, list[Variable]]


class DegreeViolation(Exception):
    pass


class Failed(Exception):
    pass


class FoundT(Exception):
    pass


class NSP(Enum):
    NONE = auto()
    PLUS_EPSILON = auto()
    MINUS_EPSILON = auto()
    PLUS_INFINITY = auto()
    MINUS_INFINITY = auto()


@dataclass
class TestPoint:

    guard: Optional[Formula] = None
    num: Term = 0
    den: Term = 1
    nsp: NSP = NSP.NONE


@dataclass
class EliminationSet:

    variable: Variable
    test_points: list[TestPoint]


@dataclass
class Node:

    variables: list[Variable]
    formula: Formula
    answer: list

    def __str__(self):
        return f'Node({self.variables}, {self.formula}, {self.answer})'

    def eset(self) -> EliminationSet:
        return self.gauss_eset() or self.regular_eset()

    def gauss_eset(self) -> Optional[EliminationSet]:
        return None

    def regular_eset(self) -> EliminationSet:

        def guard():
            return Ne(a, 0) if not a.is_constant() else None

        x = self.variables.pop()
        test_points = [TestPoint(nsp=NSP.MINUS_INFINITY)]
        for atom in self.formula.atoms():
            assert isinstance(atom, RcfAtomicFormulas)
            match atom.lhs.degree(x):
                case -1 | 0:
                    continue
                case 1:
                    a = atom.lhs.coefficient({x: 1})
                    b = atom.lhs.coefficient({x: 0})
                    match atom:
                        case Eq():
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Ne():
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case Le():
                            if a.is_constant() and a > 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Ge():
                            if a.is_constant() and a < 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a)
                        case Lt():
                            if a.is_constant() and a > 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case Gt():
                            if a.is_constant() and a < 0:
                                continue
                            tp = TestPoint(guard=guard(), num=-b, den=a, nsp=NSP.PLUS_EPSILON)
                        case _:
                            assert False, atom
                    test_points.append(tp)
                case _:
                    raise DegreeViolation(atom, x, atom.lhs.degree(x))
        return EliminationSet(variable=x, test_points=test_points)

    def vsubs(self, eset: EliminationSet) -> list[Node]:
        variables = self.variables
        x = eset.variable
        new_nodes = []
        for tp in eset.test_points:
            new_formula = self.formula.transform_atoms(lambda atom: self.vsubs_atom(atom, x, tp))
            if tp.guard is not None:
                new_formula = And(tp.guard, new_formula)
            new_nodes.append(Node(variables.copy(), new_formula, []))
        return new_nodes

    def vsubs_atom(self, atom: RcfAtomicFormula, x: Variable, tp: TestPoint) -> Formula:

        def mu() -> Formula:
            """Substitute ±oo into ordering constraint.
            """
            c = lhs.coefficient({x: 0})
            mu: Formula = atom.func(c, 0)
            for e in range(1, lhs.degree(x) + 1):
                c = lhs.coefficient({x: e})
                if tp.nsp == NSP.MINUS_INFINITY and e % 2 == 1:
                    c = - c
                mu = Or(Gt(c, 0), And(Eq(c, 0), mu))
            return mu

        def nu() -> Formula:
            """Substitute ±€ into any constraint.
            """
            ...

        def sigma() -> Formula:
            """Substitute quotient into any constraint.
            """
            fraction_field = FractionField(ring.sage_ring)
            lhs = atom.lhs
            lhs = lhs.subs(**{str(x): fraction_field(tp.num, tp.den)})
            match atom:
                case Eq() | Ne():
                    lhs = lhs.numerator()
                case Ge() | Le() | Gt() | Lt():
                    lhs = (lhs * lhs.denominator() ** 2).numerator()
                case _:
                    assert False, atom.func
            assert lhs.parent() in (ring.sage_ring, ZZ), lhs.parent()
            return atom.func(lhs, 0)

        def tau():
            """Substitute transcendental element in to equality.
            """
            args = []
            for e in range(lhs.degree(x) + 1):
                c = lhs.coefficient({x: e})
                if c.is_zero():
                    continue
                if c.is_constant():
                    return F
                args.append(Eq(c, 0))
            return And(*args)

        if tp.nsp is NSP.NONE:
            # Substitute quotient into any constraint.
            return sigma()
        if atom.func in (Le, Lt):
            atom = atom.converse_func(- atom.lhs, atom.rhs)
        match atom:
            case Eq(lhs=lhs) | Ne(lhs=lhs):
                # Substitute transcendental element in to equality.
                return tau()
            case Ge(lhs=lhs):
                match tp.nsp:
                    case NSP.PLUS_EPSILON:
                        ...
                    case NSP.MINUS_EPSILON:
                        ...
                    case NSP.PLUS_INFINITY | NSP.MINUS_INFINITY:
                        # Substitute ±oo into ordering constraint.
                        return mu()
            case Gt(lhs=lhs):
                match tp.nsp:
                    case NSP.PLUS_EPSILON:
                        ...
                    case NSP.MINUS_EPSILON:
                        ...
                    case NSP.PLUS_INFINITY | NSP.MINUS_INFINITY:
                        # Substitute ±oo into ordering constraint.
                        return mu()
        assert False, (atom, tp)


class Pool(list[Node]):

    def __init__(self, nodes: list[Node]) -> None:
        self.push(nodes)

    def push(self, nodes: list[Node]) -> None:
        for node in nodes:
            match node.formula:
                case _F():
                    continue
                case Or(args=args):
                    for arg in args:
                        self.append(Node(node.variables, arg, node.answer))
                case And() | AtomicFormula():
                    self.append(node)
                case _:
                    assert False, node


@dataclass
class VirtualSubstitution:
    """Quantifier elimination by virtual substitution.
    """

    blocks: Optional[list[QuantifierBlock]] = None
    matrix: Optional[Formula] = None
    negated: Optional[bool] = None
    pool: Optional[Pool] = None
    success_nodes: Optional[list[Node]] = None
    failure_nodes: Optional[list[Node]] = None

    def __call__(self, f: Formula, show_progress: bool = False) -> Formula:
        if show_progress:
            save_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
        result = self.virtual_substitution(f)
        if show_progress:
            logging.getLogger().setLevel(save_level)
        return result

    def __str__(self):
        if self.blocks is not None:
            _h = [q.__qualname__ + ' ' + str(v) for q, v in self.blocks]
            _h = '  '.join(_h)
            blocks = f'[{_h}]'
        else:
            blocks = None
        if self.negated is None:
            read_as = ''
        elif self.negated is False:
            read_as = '  # read as Ex'
        else:
            assert self.negated is True
            read_as = '  # read as Not All'
        if self.pool is not None:
            _h = [f'{node}' for node in self.pool]
            _h = ',\n                '.join(_h)
            pool = f'[{_h}]'
        else:
            pool = None
        if self.success_nodes is not None:
            _h = [f'{str(f)}' for f in self.success_nodes]
            _h = ',\n                '.join(_h)
            success_nodes = f'[{_h}]'
        else:
            success_nodes = None
        return (f'{self.__class__} [\n'
                f'    blocks   = {blocks},\n'
                f'    matrix   = {self.matrix},\n'
                f'    negated  = {self.negated},{read_as}\n'
                f'    pool     = {str(pool)},\n'
                f'    success_nodes = {success_nodes}\n'
                f']')

    def collect_success_nodes(self) -> None:
        if self.failure_nodes:
            raise DegreeViolation()
        self.matrix = Or(*(node.formula for node in self.success_nodes))
        if self.negated:
            self.matrix = Not(self.matrix)
        self.negated = None
        self.pool = None
        self.success_nodes = None
        logging.info(f'{self.collect_success_nodes.__qualname__}: {self}')

    def pnf(self, f: Formula) -> Formula:
        return _pnf(f)

    def pop_block(self) -> None:
        assert self.matrix, "no matrix"
        quantifier, vars_ = self.blocks.pop()
        matrix = self.matrix
        self.matrix = None
        if quantifier is All:
            self.negated = True
            matrix = Not(matrix)
        else:
            self.negated = False
        self.pool = Pool([Node(vars_, self.simplify(matrix), [])])
        self.success_nodes = []
        logging.info(f'{self.pop_block.__qualname__}: {self}')

    def process_pool(self) -> None:
        while self.pool:
            node = self.pool.pop()
            try:
                eset = node.eset()
            except DegreeViolation:
                self.failure_nodes.append(node)
                continue
            nodes = node.vsubs(eset)
            if nodes[0].variables:
                self.pool.push(nodes)
            else:
                self.success_nodes.extend(nodes)

    def setup(self, f: Formula) -> None:
        f = self.pnf(f)
        blocks = []
        vars_ = []
        while isinstance(f, QuantifiedFormula):
            Q = type(f)
            while isinstance(f, Q):
                vars_.append(f.var)
                f = f.arg
            blocks.append((Q, vars_))
            vars_ = []
        self.blocks = blocks
        self.matrix = f
        logging.info(f'{self.setup.__qualname__}: {self}')

    def simplify(self, *args, **kwargs) -> Formula:
        return _simplify(*args, **kwargs)

    def virtual_substitution(self, f):
        # The following manipulation of a private property is dirty. There
        # seems to be no supported way to reset reference time.
        logging._startTime = time()  # type: ignore
        self.setup(f)
        while self.blocks:
            try:
                self.pop_block()
                self.process_pool()
            except FoundT:
                self.pool = None
                self.success_nodes = [Node(variables=[], formula=T, answer=[])]
            self.collect_success_nodes()
        return self.simplify(self.matrix)


qe = virtual_substitution = VirtualSubstitution()
