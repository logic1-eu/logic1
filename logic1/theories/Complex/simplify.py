"""Simplification for complex formulas.
"""

from collections.abc import Iterable, Set
from dataclasses import dataclass, field
from typing import Optional, Self, TypeVar

from gmpy2 import mpz
import networkx as nx

from logic1 import abc
from logic1.firstorder.boolean import _F, F, _T, T, And, Or
from logic1.theories import RCF

from logic1.theories.Complex.types import α, Formula, Number
from logic1.theories.Complex.ast import _I, ASTVisitor, Add, Conj, Im, Mul, Neg, Pow, Rat, Re, Var
from logic1.theories.Complex.term import I, Term, Variable
from logic1.theories.Complex.atomic import AtomicFormula, Eq


def min_weight_partial_edge_cover(nodes_costs: dict[α, float], edge_weights: dict[tuple[α, α], float]) -> Set[tuple[α, α]]:
    """Return a partial edge cover minimizing edge and uncovered-node costs.

    >>> nodes_costs = {'a': 1, 'b': 2, 'c': 4}
    >>> edge_weights = {('a', 'b'): 1, ('b', 'c'): 2}
    >>> cover = min_weight_partial_edge_cover(nodes_costs, edge_weights)
    >>> [(min(u, v), max(u, v)) for u, v in sorted(cover)]
    [('b', 'c')]
    """
    nodes = set(nodes_costs.keys())

    min_costs: dict[α, float] = dict()
    min_others: dict[α, Optional[α]] = dict()
    for node in nodes:
        min_costs[node] = nodes_costs[node]
        min_others[node] = None
        for other in nodes:
            weight = edge_weights.get((node, other), float('inf'))
            if weight < min_costs[node]:
                min_costs[node] = weight
                min_others[node] = other

    G: nx.Graph = nx.Graph()
    G.add_nodes_from(nodes)
    for node, other in edge_weights:
        savings = min_costs[node] + min_costs[other] - edge_weights[(node, other)]
        if savings > 0:
            G.add_edge(node, other, weight=savings)

    matching = nx.max_weight_matching(G)
    for node, other in matching:
        nodes.remove(node)
        nodes.remove(other)
    for node in nodes:
        maybe_other = min_others[node]
        if maybe_other is not None:
            matching.add((node, maybe_other))
    return matching



@dataclass(frozen=True)
class Options(abc.simplify.Options):
    """Options for the simplification process. Currently empty, but can
    be extended in the future.
    """
    pass


class ComplexityVisitor(ASTVisitor[float]):
    """Visitor that computes a measure of the complexity of a :class:`.ast.AST`,
    used to guide the simplification process. The complexity is defined as
    :code:`1` plus the sum of the complexities of the children.
    """

    def visit_rat(self, num: Rat) -> float:
        """Return the complexity of a rational number. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_rat`.

        >>> ComplexityVisitor().visit_rat(Rat(2))
        1.0
        >>> ComplexityVisitor().visit_rat(Rat(0.5))
        2.0
        """
        if num.value.denominator == mpz(1):
            return 1.0
        else:
            return 2.0

    def visit_i(self, _: _I) -> float:
        """Return the complexity of the imaginary unit. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_i`.

        >>> ComplexityVisitor().visit_i(I)
        1.0
        """
        return 1.0

    def visit_var(self, var: Var) -> float:
        """Return the complexity of a variable. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_var`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_var(z)
        1.0
        """
        return 1.0

    def visit_add(self, add: Add) -> float:
        """Return the complexity of an addition. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_add`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_add(z + 2)
        3.0
        """
        return 1.0 + sum((arg.arg if isinstance(arg, Neg) else arg).accept(self) for arg in add.args)

    def visit_mul(self, mul: Mul) -> float:
        """Return the complexity of a multiplication. Implements
        the abstract method :meth:`.Complex.ASTVisitor.visit_mul`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_mul(z * 2)
        3.0
        """
        return 1.0 + sum(arg.accept(self) for arg in mul.args)

    def visit_pow(self, pow: Pow) -> float:
        """Return the complexity of a power. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_pow`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_pow(z ** 5)
        2.0
        """
        return 1.0 + pow.base.accept(self)

    def visit_neg(self, neg: Neg) -> float:
        """Return the complexity of a negation. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_neg`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_neg(-z)
        2.0
        """
        return 1.0 + neg.arg.accept(self)

    def visit_conj(self, conj: Conj) -> float:
        """Return the complexity of a conjugation. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_conj`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_conj(~z)
        2.0
        """
        return 1.0 + conj.arg.accept(self)

    def visit_re(self, re: Re) -> float:
        """Return the complexity of a real part. Implements the
        abstract method :meth:`.Complex.ASTVisitor.visit_re`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_re(Re(z))
        2.0
        """
        return 1.0 + re.arg.accept(self)

    def visit_im(self, im: Im) -> float:
        """Return the complexity of an imaginary part. Implements
        the abstract method :meth:`.Complex.ASTVisitor.visit_im`.

        >>> z = Var('z')
        >>> ComplexityVisitor().visit_im(Im(z))
        2.0
        """
        return 1.0 + im.arg.accept(self)


@dataclass
class InternalRepresentation(
        abc.simplify.InternalRepresentation[AtomicFormula, Term, Variable, Number]):
    """Internal representation of a set of atomic formulas used for
    simplification. Implements the abstract class
    :class:`.abc.simplify.InternalRepresentation` for the theory of
    complex numbers.
    """

    _atoms: set[AtomicFormula] = field(default_factory=set)
    _options: Options = field(default_factory=Options)

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> abc.simplify.RESTART:
        """Implements the abstract method
        :meth:`.abc.simplify.InternalRepresentation.add`.
        """
        if gand is Or:
            atoms = [atom.to_complement() for atom in atoms]
        for atom in atoms:
            simple_atom = atom.simplify()
            if simple_atom is T:
                continue
            if simple_atom is F:
                raise InternalRepresentation.Inconsistent()
            assert isinstance(simple_atom, AtomicFormula) and simple_atom.rhs.is_zero()
            if simple_atom.to_complement() in self._atoms:
                raise InternalRepresentation.Inconsistent()
            self._atoms.add(simple_atom)
        return abc.simplify.RESTART.NONE

    @staticmethod
    def _complexity(atom: AtomicFormula) -> float:
        """Return a measure of the complexity of the given atomic
        formula, used to guide the simplification process.
        """
        visitor = ComplexityVisitor()
        return 1.0 + atom.lhs.normal_ast.accept(visitor) + atom.rhs.normal_ast.accept(visitor)

    def extract(self, gand: type[And | Or], ref: Self) -> list[AtomicFormula]:
        """Implements the abstract method
        :meth:`.abc.simplify.InternalRepresentation.extract`.
        """

        assert all(atom in self._atoms for atom in ref._atoms)

        refs = set(ref._atoms)

        node_costs: dict[AtomicFormula, float] = dict()
        edge_weights: dict[tuple[AtomicFormula, AtomicFormula], float] = dict()
        edge_results: dict[tuple[AtomicFormula, AtomicFormula], AtomicFormula | _T | _F] = dict()

        for atom in self._atoms:
            node_costs[atom] = self._complexity(atom)
        for atom in refs:
            node_costs[atom] = 0.0

        for node in node_costs:
            for other in node_costs:
                new = self._merge_atoms(node, other)
                if new is None:
                    continue
                if new is F:
                    raise InternalRepresentation.Inconsistent()
                if new is T:
                    new_weight = 0.0
                else:
                    assert isinstance(new, AtomicFormula)
                    new_weight = self._complexity(new)
                old_weight = edge_weights.get((node, other), float('inf'))
                if new_weight < old_weight:
                    edge_weights[(node, other)] = edge_weights[(other, node)] = new_weight
                    edge_results[(node, other)] = edge_results[(other, node)] = new

        cover = min_weight_partial_edge_cover(node_costs, edge_weights)

        result: set[AtomicFormula] = set()
        remaining: set[AtomicFormula] = set(self._atoms)
        for node, other in cover:
            res = edge_results[(node, other)]
            if res is T:
                continue
            assert isinstance(res, AtomicFormula)
            if res in refs:
                continue
            if res.to_complement() in refs:
                raise InternalRepresentation.Inconsistent()
            result.add(res)
            if node in remaining:
                remaining.remove(node)
            if other in remaining:
                remaining.remove(other)
        result.update(remaining)

        result = set(atom for atom in result if not atom in refs)
        for atom in result:
            if atom.to_complement() in refs:
                raise InternalRepresentation.Inconsistent()
        if gand is Or:
            return [atom.to_complement() for atom in result]
        else:
            return list(result)

    @staticmethod
    def _merge_atoms(atom1: AtomicFormula, atom2: AtomicFormula) -> Optional[AtomicFormula | _T | _F]:
        """Given two atomic formulas, returns a
        single atomic formula that is equivalent to both of them, or
        None if no such formula can be found.
        """
        if not isinstance(atom1, Eq) or not isinstance(atom2, Eq):
            return None
        if atom1.is_imaginary():
            atom1 = atom1.op(atom1.lhs / I, atom1.rhs / I)
        if atom2.is_real():
            atom2 = atom2.op(atom2.lhs * I, atom2.rhs * I)
        if not atom1.is_real() or not atom2.is_imaginary():
            return None
        result = atom1.op(atom1.lhs + atom2.lhs, atom1.rhs + atom2.rhs)
        return result.simplify()

    def next_(self, remove: Optional[Variable] = None) -> Self:
        """Implements the abstract method
        :meth:`.abc.simplify.InternalRepresentation.next_`.
        """
        atoms = set()
        for atom in self._atoms:
            if remove is None or remove not in atom.fvars():
                atoms.add(atom)
        return self.__class__(_atoms=atoms, _options=self._options)


@dataclass(frozen=True)
class Simplify(abc.simplify.Simplify[
        AtomicFormula, Term, Variable, Number, InternalRepresentation, Options]):
    """Simplification procedure for the theory of complex numbers.
    Implements the abstract class :class:`.abc.simplify.Simplify`.
    """

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
    """Return a simplified formula that is equivalent to the given
    formula f, using the given assumptions and options.
    """
    rcf_assume = assume_to_rcf(assume)
    rcf_formula = formula_to_rcf(f)
    rcf_formula = RCF.simplify(rcf_formula, assume=rcf_assume)
    formula = formula_to_complex(rcf_formula)
    return Simplify(Options(**options)).simplify(formula, assume)


def is_valid(f: Formula, assume: Iterable[AtomicFormula] = [], **options) -> Optional[bool]:
    """Return :obj:`True` if the formula is valid under the given
    assumptions and options, :obj:`False` if it is not valid, and :obj:`None`
    if the validity cannot be determined.
    """
    rcf_assume = assume_to_rcf(assume)
    rcf_formula = formula_to_rcf(f)
    return RCF.is_valid(rcf_formula, assume=rcf_assume, **options)


from logic1.theories.Complex.qe import assume_to_rcf, formula_to_complex, formula_to_rcf
