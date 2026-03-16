from collections.abc import Iterable, Set
from dataclasses import dataclass, field
from typing import Optional, Self
from unittest import result

import networkx as nx

from logic1 import abc
from logic1.firstorder.boolean import And, Or
from logic1.theories.Complex.atomic import AtomicFormula, Eq, Ne
from logic1.theories.Complex.term import _I, I, Conj, Im, Neg, Rational, Re, Term, TermVisitor, Variable
from logic1.theories.Complex.types import Formula, Number

from gmpy2 import mpz


@dataclass(frozen=True)
class Options(abc.simplify.Options):
    pass


class TermComplexityVisitor(TermVisitor[float]):

    def visit_rational(self, num: Rational) -> float:
        if num.value.denominator == mpz(1):
            return 1.0
        else:
            return 2.0
        
    def visit_i(self, _: _I) -> float:
        return 1.0
    
    def visit_variable(self, var: Variable) -> float:
        return 1.0

    def visit_add(self, add) -> float:
        return 1.0 + sum((arg.arg if isinstance(arg, Neg) else arg).accept(self) for arg in add.args)
    
    def visit_mul(self, mul) -> float:   
        return 1.0 + sum(arg.accept(self) for arg in mul.args)

    def visit_pow(self, pow) -> float:
        return 1.0 + pow.base.accept(self)

    def visit_neg(self, neg: Neg) -> float:
        return 1.0 + neg.arg.accept(self)
    
    def visit_conj(self, conj: Conj) -> float:
        return conj.arg.accept(self)
    
    def visit_re(self, re: Re) -> float:
        return 1.0 + re.arg.accept(self)
        
    def visit_im(self, im: Im) -> float:
        return 1.0 + im.arg.accept(self)


@dataclass
class InternalRepresentation(
        abc.simplify.InternalRepresentation[AtomicFormula, Term, Variable, Number]):

    _atoms: set[AtomicFormula] = field(default_factory=set)
    _options: Options = field(default_factory=Options)

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> abc.simplify.RESTART:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.add`.
        """
        if gand is Or:
            atoms = (atom.to_complement() for atom in atoms)
        for atom in atoms:
            atom = atom.normalize_complex()
            assert atom.rhs.is_zero()
            try:
                if atom._eval_constant():
                    continue
                else:
                    raise InternalRepresentation.Inconsistent()
            except ValueError: 
                pass
            if atom.to_complement() in self._atoms:
                raise InternalRepresentation.Inconsistent()
            self._atoms.add(atom)
        return abc.simplify.RESTART.NONE

    def extract(self, gand: type[And | Or], ref: Self) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.extract`.
        """
        INF = float('inf')
        complexity_visitor = TermComplexityVisitor()

        # print("self:", self._atoms)
        # print("ref:", ref._atoms)

        def merge(atom1: AtomicFormula, atom2: AtomicFormula) -> Optional[AtomicFormula]:
            # print("atom1:", atom1, atom1.is_real(), atom1.is_imaginary())
            # print("atom2:", atom2, atom2.is_real(), atom2.is_imaginary())
            if isinstance(atom1, Ne) or not isinstance(atom2, Eq):
                return None
            if atom1.is_imaginary():
                assert isinstance(atom1, Eq)
                atom1 = atom1.op(atom1.lhs / I, atom1.rhs / I)
            if atom2.is_real():
                atom2 = atom2.op(atom2.lhs * I, atom2.rhs * I)
            if not atom1.is_real() or not atom2.is_imaginary():
                return None
            result = atom1.op(atom1.lhs + atom2.lhs, atom1.rhs + atom2.rhs).normalize_complex()
            # print("result:", result, result.is_real(), result.is_imaginary())
            return result

        def complexity(atom: AtomicFormula) -> float:
            # return len(str(atom))
            visitor = TermComplexityVisitor()
            return 1.0 + atom.lhs.accept(visitor) + atom.rhs.accept(visitor)

        nodes: set[AtomicFormula] = set()
        penalties: dict[AtomicFormula, float] = dict()

        for atom in self._atoms:
            if atom in ref._atoms:
                continue
            if atom.to_complement() in ref._atoms:
                raise InternalRepresentation.Inconsistent()
            nodes.add(atom)
            penalties[atom] = complexity(atom)
        for atom in ref._atoms:
            nodes.add(atom)
            penalties[atom] = 0.0

        # print("nodes:", nodes)

        edge_weight = dict()
        edge_direction = dict()
        for node in nodes:
            for other in nodes:
                new_node = merge(node, other)
                if new_node is None:
                    continue
                old_weight = min(edge_weight.get((node, other), INF), edge_weight.get((other, node), INF))
                new_weight = complexity(new_node)
                if new_weight < old_weight:
                    edge_weight[(node, other)] = edge_weight[(other, node)] = new_weight
                    edge_direction[(node, other)] = edge_direction[(other, node)] = (node, other)

        min_edge_weight = dict()
        min_edge_node = dict()
        for node in nodes:
            min_weight = penalties[node]
            min_node = None
            for other in nodes:
                weight = edge_weight.get((node, other), INF)
                if weight < min_weight:
                    min_weight = weight
                    min_node = other
            min_edge_weight[node] = min_weight
            min_edge_node[node] = min_node

        node_cost = {n: min(penalties[n], min_edge_weight[n]) for n in nodes}

        G: nx.Graph = nx.Graph()
        G.add_nodes_from(nodes)
        for u, v in edge_weight:    
            savings = node_cost[u] + node_cost[v] - edge_weight[(u, v)]
            if savings > 0:
                G.add_edge(u, v, weight=savings)
            
        matching = nx.max_weight_matching(G)
        atoms = nodes.intersection(self._atoms)
        result: set[AtomicFormula] = set()
 
        for u, v in matching:
            u, v = edge_direction[(u, v)]
            atom = merge(u, v)
            assert atom is not None
            if atom in ref._atoms:
                continue
            if atom.to_complement() in ref._atoms:
                raise InternalRepresentation.Inconsistent()
            result.add(atom)
            atoms.remove(u)
            atoms.remove(v)

        for u in atoms:
            v = min_edge_node[u]
            if v is None:
                atom = u
            else:
                u, v = edge_direction[(u, v)]
                atom = merge(u, v)
                assert atom is not None
            if atom in ref._atoms:
                continue
            if atom.to_complement() in ref._atoms:
                raise InternalRepresentation.Inconsistent()
            result.add(atom)
            
        if gand is Or:
            return [atom.to_complement() for atom in result]
        else:
            return list(result)

    def next_(self, remove: Optional[Variable] = None) -> Self:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.next_`.
        """
        return self.__class__(_atoms=set(self._atoms), _options=self._options)
    

@dataclass(frozen=True)
class Simplify(abc.simplify.Simplify[
        AtomicFormula, Term, Variable, Number, InternalRepresentation, Options]):

    _options: Options = field(default_factory=Options)

    def create_initial_representation(self, assume: Iterable[AtomicFormula]) \
            -> InternalRepresentation:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.create_initial_representation`.
        """
        rep = InternalRepresentation(_options=self._options)
        rep.add(And, assume)
        return rep

    def simpl_at(self, atom: AtomicFormula, context: Optional[type[And] | type[Or]]) -> Formula:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.simpl_at`.
        """
        return atom.simplify()


def simplify(f: Formula, assume: Iterable[AtomicFormula] = [], **options) -> Formula:
    return Simplify(Options(**options)).simplify(f, assume)


def is_valid(f: Formula, assume: Iterable[AtomicFormula] = [], **options) -> Optional[bool]:
    return Simplify(Options(**options)).is_valid(f, assume)
