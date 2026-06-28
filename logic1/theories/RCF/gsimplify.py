from dataclasses import dataclass, field
import logging
from math import prod
from typing import cast, Final, Iterable, Self

from logic1.firstorder import And, _F, Not, Or, _T
from logic1.theories.RCF import redlog
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.RCF.bnf import cnf
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.term import Term, VV
from logic1.theories.RCF.types import Formula


class NextClause(Exception):
    pass


@dataclass
class Clause:

    _data: dict[type[AtomicFormula], set[AtomicFormula]] = field(
        default_factory=lambda: {Eq: set(), Ne: set(), Gt: set(), Lt: set()})

    def __getitem__(self, key: type[AtomicFormula]) -> set[AtomicFormula]:
        return self._data[key]

    def __setitem__(self, key: type[AtomicFormula], entry: set[AtomicFormula]) -> None:
        self._data[key] = entry

    def __repr__(self) -> str:
        parts = [f'{rel.__name__}: {self[rel]}'
                 for rel in (Eq, Ne, Gt, Lt)
                 if self[rel]]
        return f'Clause({", ".join(parts)})'

    def _add(self, atom: AtomicFormula) -> None:
        rel = type(atom)
        if rel in (Ge, Le):
            strict_rel = rel.strict_part()
            self[strict_rel].add(strict_rel(atom.lhs, 0))
            self[Eq].add(Eq(atom.lhs, 0))
        else:
            self[rel].add(atom)

    def as_product(self, rel: type[AtomicFormula]) -> Term:
        return prod((atom.lhs for atom in self[rel]), start=Term(1))

    @classmethod
    def from_disjunction(cls, f: Or | AtomicFormula) -> Self:
        """Create new clause from a possibly degenerate disjunction.
        """
        clause = cls()
        if isinstance(f, Or):
            args = f.args
        else:
            args = (f,)
        for arg in args:
            assert isinstance(arg, AtomicFormula)
            clause._add(arg)
        return clause

    def is_equational(self) -> bool:
        return not any(self[rel] for rel in (Ne, Lt, Gt))

    def to_formula(self) -> Formula:
        return Or(*set().union(*self._data.values()))


@dataclass(frozen=True)
class Options:
    use_redlog_cnf: bool = True
    bnfsac: bool = True
    bnfsm: bool = True


@dataclass
class GSimplify:

    _options: Options = field(default_factory=Options)

    def cnf(self, f: Formula) -> Formula:
        if self._options.use_redlog_cnf:
            logging.info('computing cnf using Redlog with '
                         f'{self._options.bnfsac=}, '
                         f'{self._options.bnfsm=} ...')
            return redlog.cnf(f, bnfsac=self._options.bnfsac,
                              bnfsm=self._options.bnfsm)
        else:
            logging.info('computing cnf using PyEda ...')
            return cnf(f)


    def formula_to_clauses(self, f: Formula) -> tuple[list[Clause], bool]:
        assert isinstance(f, (And, Or))

        # Special treatment of flat formulas. Here conjunction goes via negation!
        if all(isinstance(arg, AtomicFormula) for arg in f.args):
            logging.info('skipping cnf computation on flat formula')
            if isinstance(f, And):
                f = Not(f).to_nnf()
                negated = True
            else:
                negated = False
            return [Clause.from_disjunction(cast(Or, f))], negated

        # The general case using CNF. Disjunction goes via negation.
        if isinstance(f, Or):
            f = Not(f).to_nnf()
            negated = True
        else:
            negated = False
        f = self.cnf(f)
        logging.info(f'cnf has {len(list(f.atoms()))} atoms')
        assert isinstance(f, And) and not all(isinstance(arg, AtomicFormula) for arg in f.args)
        clauses = []
        for arg in f.args:
            clauses.append(Clause.from_disjunction(arg))
        return clauses, negated

    def gsimplify(self, f: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
        f = simplify(f, assume=assume)
        if isinstance(f, (AtomicFormula, _T, _F)):
            return f
        clauses, negate = self.formula_to_clauses(f)
        clauses = self.gsimplify_clauses(clauses, assume=assume)
        f = And(*(clause.to_formula() for clause in clauses))
        # Although f is a CNF, we recompute a CNF in order to take advantage of
        # subsumption and cut.
        f = self.cnf(f)
        if negate:
            f = Not(f)
        # Now f is essenially in either CNF or DNF. The final simplification
        # will not preserve this in general.
        f = simplify(f, assume=assume)
        return f

    def gsimplify_clauses(self, clauses: list[Clause], assume: Iterable[AtomicFormula] = []) -> list[Clause]:
        # RL: Simplify assume
        # RL: GSimplify assume - check for inconsistent assumptions
        count = len(clauses)
        new_clauses = []

        # Remove equational clauses entailed by the assumed equations. Add all
        # other equational clauses to the assumed equations.
        logging.info(f'processing equational clauses ({count} clauses left)')
        assumed_eq_gb = Term.gbasis(atom.lhs for atom in assume if isinstance(atom, Eq))
        for clause in clauses:
            if clause.is_equational():
                logging.debug(f'{count} clauses left')
                count -= 1
                product = clause.as_product(Eq)
                if self.in_radical(product, assumed_eq_gb):
                    continue
                h = product.reduce(assumed_eq_gb)
                simplified_equation = simplify(Eq(h, 0), assume=assume)
                if isinstance(simplified_equation, _T):
                    continue
                if isinstance(simplified_equation, _F):
                    return [Clause()]
                new_clauses.append(clause)
                assumed_eq_gb.append(product)
                assumed_eq_gb = Term.gbasis(assumed_eq_gb)

        # Now simplify the other clauses.
        logging.info(f'processing non-equational clauses ({count} clauses left)')
        for clause in clauses:
            try:
                if not clause.is_equational():
                    logging.debug(f'{count} clauses left')
                    count -= 1
                    this_assumed_eq_gb = Term.gbasis(assumed_eq_gb + [atom.lhs for atom in clause[Ne]])
                    # Process the product of all equations together with
                    # disequality and strict inequality assumptions
                    F = {atom.lhs for atom in assume if isinstance(atom, (Ne, Gt, Lt))}
                    product = prod(F, start=clause.as_product(Eq))
                    if self.in_radical(product, this_assumed_eq_gb):
                        continue
                    h = product.reduce(this_assumed_eq_gb)
                    simplified_equation = simplify(Eq(h, 0), assume=assume)
                    if isinstance(simplified_equation, _T):
                        continue
                    new_clause = Clause()
                    if not isinstance(simplified_equation, _F):
                        for atom in clause[Eq]:  # !*rlgsred=T
                            h = atom.lhs.reduce(this_assumed_eq_gb)
                            simplified_atom = simplify(Eq(h, 0), assume=assume)
                            if isinstance(simplified_atom, _T):
                                raise NextClause()
                            if isinstance(simplified_atom, _F):
                                continue
                            new_clause[Eq].add(Eq(h, 0))

                    # Equations are fine now. Process all inequalities. We test radical membership for
                    # the strict ones but not for the weak ones anymore. [AD: but we are missing sth here]
                    for rel in (Gt, Lt):
                        for atom in clause[rel]:
                            if self.in_radical(atom.lhs, this_assumed_eq_gb):
                                continue
                            h = atom.lhs.reduce(this_assumed_eq_gb)
                            simplified_atom = simplify(rel(h, 0), assume=assume)
                            if isinstance(simplified_atom, _T):
                                raise NextClause()
                                # AD: We lose sth due to splitting, might not matter
                            if isinstance(simplified_atom, _F):
                                continue
                            new_clause[rel].add(rel(h, 0))  # !*rlgsred=T

                    H = set()
                    for atom in clause[Ne]:  # !*rlgsred=T
                        h = atom.lhs.reduce(assumed_eq_gb)
                        simplified_atom = simplify(Ne(h, 0), assume=assume)
                        if isinstance(simplified_atom, _T):
                            assert False
                            continue
                        if isinstance(simplified_atom, _F):
                            assert False
                            raise NextClause()
                        H.add(h)
                    new_clause[Ne] = {Ne(g, 0) for g in Term.gbasis(H)}  # !*rlgssub=T

                    new_clauses.append(new_clause)

            except NextClause:
                pass

        return new_clauses

    def in_radical(self, f: Term, G: list[Term]) -> bool:
        # Y = VV.fresh()
        Y = VV['SecretRabinovichVariable']
        return Term(1) in Term.gbasis(G + [1 - Y * f])


def gsimplify(f: Formula, assume: Iterable[AtomicFormula] =[], **options) -> Formula:
    return GSimplify(Options(**options)).gsimplify(f, assume)
