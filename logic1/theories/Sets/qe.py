from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations
import logging
from typing import Iterable, Never

from logic1 import abc
from logic1 import depricated
from logic1.firstorder import And, Or, _T
from logic1.support.tracing import trace  # noqa
from logic1.theories.Sets.atomic import AtomicFormula, C, Eq, Ne, Variable
from logic1.theories.Sets.bnf import dnf
from logic1.theories.Sets.simplify import simplify
from logic1.theories.Sets.typing import Formula


class Pool(depricated.qe.PoolOnePrimitive):

    def dnf(self, f: Formula) -> Formula:
        return dnf(f)


class DepricatedQuantifierElimination(depricated.qe.QuantifierElimination):
    """Quantifier elimination for the theory of Sets.

    >>> from logic1.firstorder import *
    >>> from logic1.theories.Sets import *
    >>> a, u, v, w, x, y, z = VV.get('a', 'u', 'v', 'w', 'x', 'y', 'z')
    >>> f = All(u, Ex(w, All(x, Ex([y, v], And(Or(u == v, v != w),
    ...                                        ~ Equivalent(u == x, u != w),
    ...                                        y == a)))))
    >>> qe(f)
    C_(2)
    >>> g = Ex([x, y, z],
    ...        And(x != y, x != z, y != z, All(u, Or(u == x, u == y, u == z))))
    >>> qe(g)
    And(C(3), C_(4))
    >>> h = Implies(Ex([w, x], w != x),
    ...             Ex([w, x, y, z],
    ...                And(w != x, w != y, w != z, x != y, x != z, y != z)))
    >>> qe(h)
    Or(C_(2), C(4))
    """

    def __call__(self, f, sism: bool = True, show_progress: bool = False) -> Formula:
        if show_progress:
            save_level = logging.getLogger().getEffectiveLevel()
            logging.getLogger().setLevel(logging.INFO)
        self.sism = sism
        result = self.qe(f)
        if show_progress:
            logging.getLogger().setLevel(save_level)
        return result

    def select_and_pop(self, vars_: list, f: Formula) -> Variable:
        # revise: use a counter
        d = {v: 0 for v in vars_}
        args = f.args if f.op is And else (f,)
        for atom in args:
            for v in set(atom.fvars()):
                if v in vars_:
                    d[v] += 1
        vals = list(d.values())
        keys = list(d.keys())
        v = keys[vals.index(min(vals))]
        vars_.remove(v)
        # print(d, v)
        return v

    def _Pool(self, vars_: list[Variable], f: Formula) -> Pool:
        return Pool(vars_, f)

    def qe1(self, v: Variable, f: Formula) -> Formula:
        def eta(Z: set, k: int) -> Formula:
            args = []
            for k_choice in combinations(Z, k):
                args1: list[Formula] = []
                for z in Z:
                    args_inner = []
                    for chosen_z in k_choice:
                        args_inner.append(Eq(z, chosen_z))
                    args1.append(Or(*args_inner))
                f1 = And(*args1)
                args2 = []
                for i in range(k):
                    for j in range(i + 1, k):
                        args2.append(Ne(k_choice[i], k_choice[j]))
                f2 = And(*args2)
                args.append(And(f1, f2))
            return Or(*args)

        logging.info(f'{self.qe1.__qualname__}: eliminating Ex {v} ({f})')
        eqs, nes = self._split_by_relation(f)
        if eqs:
            solution = eqs[0].rhs if eqs[0].rhs != v else eqs[0].lhs
            return f.subs({v: solution})
        Z = set(And(*nes).fvars()) - {v}
        m = len(Z)
        args = (And(eta(Z, k), C(k + 1)) for k in range(1, m + 1))
        phi_prime: Formula = Or(*args)
        logging.info(f'{self.qe1.__qualname__}: result is {phi_prime}')
        return phi_prime

    def simplify(self, f: Formula) -> Formula:
        return simplify(f) if self.sism else f.simplify()

    @staticmethod
    def _split_by_relation(f) -> tuple[list[Eq], list[Ne]]:
        eqs = []
        nes = []
        args = f.args if f.op is And else (f,)
        for atom in args:
            if atom.op is Eq:
                eqs.append(atom)
            else:
                assert atom.op is Ne, f'bad atom {atom} - {atom.op}'
                nes.append(atom)
        return eqs, nes


old_qe = DepricatedQuantifierElimination()


class Theory(abc.qe.Theory[AtomicFormula, Variable, Variable, Never]):
    """Implements the abstract method :meth:`simplify()
    <.abc.qe.Theory.simplify>` of its super class :class:`.abc.qe.Theory`.
    Required by :class:`.Node` and :class:`.QuantifierElimination`.
    """

    def simplify(self, f: Formula) -> Formula:
        """Implements the abstract method :meth:`.abc.qe.Theory.simplify`.
        """
        return simplify(f)


@dataclass
class Node(abc.qe.Node[Formula, Variable, Theory]):
    """Implements the abstract methods :meth:`copy() <.abc.qe.Node.copy>` and
    :meth:`process() <.abc.qe.Node.process>` of its super class
    :class:`.abc.qe.Node`. Required by :class:`.QuantifierElimination`.
    """

    options: abc.qe.Options

    def copy(self) -> Node:
        """Implements the abstract method :meth:`.abc.qe.Node.copy`.
        """
        return Node(variables=self.variables, formula=self.formula, options=self.options)

    def process(self, theory: Theory) -> list[Node]:
        """Implements the abstract method :meth:`.abc.qe.Node.process`.
        """
        # We assume that that atoms of the form v == v or v != v have been
        # removed by simplification. This is similar to the assumption in RCF
        # that right hand sides are zero. Both are not asserted anywhere at the
        # moment.
        counter = Counter(self.formula.bvars(frozenset(self.variables)))
        if not counter:
            return [Node(variables=[], formula=self.formula, options=self.options)]
        best_variable, _ = counter.most_common()[-1]
        self.variables.remove(best_variable)
        phi_prime = self.qe1(best_variable, self.formula)
        phi_prime = simplify(phi_prime, assume=theory.atoms)
        phi_prime = dnf(phi_prime)
        match phi_prime:
            case Or():
                nodes = []
                for arg in phi_prime.args:
                    formula = simplify(arg, assume=theory.atoms)
                    if formula is _T():
                        raise abc.qe.FoundT()
                    nodes.append(Node(variables=self.variables.copy(),
                                      formula=formula,
                                      options=self.options))
                return nodes
            case _:
                formula = simplify(phi_prime, assume=theory.atoms)
                if formula is _T():
                    raise abc.qe.FoundT()
                return [Node(variables=self.variables.copy(),
                             formula=formula,
                             options=self.options)]

    def qe1(self, v: Variable, f: Formula) -> Formula:
        def eta(Z: set[Variable], k: int) -> Formula:
            args = []
            for k_choice in combinations(Z, k):
                args1: list[Formula] = []
                for z in Z:
                    args_inner = (Eq(z, chosen_z) for chosen_z in k_choice)
                    args1.append(Or(*args_inner))
                for i in range(k):
                    for j in range(i + 1, k):
                        args1.append(Ne(k_choice[i], k_choice[j]))
                args.append(And(*args1))
            return Or(*args)

        def split(v: Variable, f: Formula) -> tuple[list[Eq], list[Ne], list[AtomicFormula]]:
            eqs = []
            nes = []
            others = []
            args = f.args if isinstance(f, And) else (f,)
            for atom in args:
                match atom:
                    case Eq() if v in atom.fvars():
                        eqs.append(atom)
                    case Ne() if v in atom.fvars():
                        nes.append(atom)
                    case AtomicFormula():
                        others.append(atom)
                    case _:
                        assert False, (v, f)
            return eqs, nes, others

        logging.info(f'{self.qe1.__qualname__}: eliminating Ex {v} ({f})')
        eqs, nes, others = split(v, f)
        if eqs:
            if eqs[0].lhs == v:
                return And(f.subs({v: eqs[0].rhs}), *others)
            else:
                assert eqs[0].rhs == v
                return And(f.subs({v: eqs[0].lhs}), *others)
        Z = {z for ne in nes for z in ne.fvars(frozenset({v}))}
        m = len(Z)
        args = (And(eta(Z, k), C(k + 1)) for k in range(1, m + 1))
        phi_prime = Or(*args)
        result = And(phi_prime, *others)
        logging.info(f'{self.qe1.__qualname__}: result is {phi_prime}')
        return result


@dataclass
class QuantifierElimination(abc.qe.QuantifierElimination[
        Node, Theory, None, abc.qe.Options, AtomicFormula, Variable, Variable, Never]):
    """Implements the abstract methods
    :meth:`create_options() <.abc.qe.QuantifierElimination.create_options>`,
    :meth:`create_root_nodes() <.abc.qe.QuantifierElimination.create_root_nodes>`,
    :meth:`create_theory() <.abc.qe.QuantifierElimination.create_theory>`,
    :meth:`create_true_node() <.abc.qe.QuantifierElimination.create_true_node>`,
    :meth:`final_simplify() <.abc.qe.QuantifierElimination.final_simplify>`,
    :meth:`init_env() <.abc.qe.QuantifierElimination.init_env>`,
    :meth:`init_env_arg() <.abc.qe.QuantifierElimination.init_env_arg>` of its
    super class :class:`.abc.qe.QuantifierElimination`.
    """

    def create_options(self, **kwargs) -> abc.qe.Options:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.create_options`.
        """
        return abc.qe.Options(**kwargs)

    def create_root_nodes(self, variables: Iterable[Variable], matrix: Formula) -> list[Node]:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.create_root_nodes`.
        """
        assert self.options is not None
        assert self.theory is not None
        formula = simplify(matrix, assume=self.theory.atoms)
        formula = dnf(formula)
        match formula:
            case Or():
                root_nodes = []
                for arg in formula.args:
                    node = Node(variables=list(variables),
                                formula=simplify(arg, assume=self.theory.atoms),
                                options=self.options)
                    if node.formula is _T():
                        raise abc.qe.FoundT()
                    root_nodes.append(node)
                return root_nodes
            case _:
                node = Node(variables=list(variables),
                            formula=simplify(formula, assume=self.theory.atoms),
                            options=self.options)
                if node.formula is _T():
                    raise abc.qe.FoundT()
                return [node]

    def create_theory(self, assume: Iterable[AtomicFormula]) -> Theory:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.create_theory`.
        """
        return Theory(assume)

    def create_true_node(self) -> Node:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.create_true_node`.
        """
        assert self.options is not None
        return Node(variables=[], formula=_T(), options=self.options)

    def final_simplify(self, formula: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.final_simplify`.
        """
        return simplify(formula, assume)

    @classmethod
    def init_env(cls, none: None) -> None:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.init_env`.
        """
        pass

    def init_env_arg(self) -> None:
        """Implements the abstract method :meth:`.abc.qe.QuantifierElimination.init_env_arg`.
        """
        return None


qe = quantifier_elimination = QuantifierElimination()
