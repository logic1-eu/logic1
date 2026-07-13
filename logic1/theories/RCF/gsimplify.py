from dataclasses import dataclass, field
import logging
from math import prod
from typing import cast, final, Iterable, Optional

from logic1.firstorder import And, _F, Not, Or, _T
from logic1.theories.RCF import redlog
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.RCF.bnf import cnf
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.term import Term, VV
from logic1.theories.RCF.types import Formula


@dataclass(frozen=True)
class Options:
    use_redlog_cnf: bool = True
    bnfsac: bool = True
    bnfsm: bool = True
    radical: bool = True

@final
@dataclass(init=False)
class Clause:

    _atoms: dict[type[AtomicFormula], set[AtomicFormula]]

    def __getitem__(self, key: type[AtomicFormula]) -> set[AtomicFormula]:
        return self._atoms[key]

    def __init__(self, f: Optional[Or | AtomicFormula] = None) -> None:
        self._atoms = {Eq: set(), Ne: set(), Gt: set(), Lt: set()}
        if isinstance(f, Or):
            args = f.args
        elif isinstance(f, AtomicFormula):
            args = (f,)
        else:
            args = ()
        for arg in args:
            assert isinstance(arg, AtomicFormula)
            self.add(arg)

    def __len__(self) -> int:
        return sum(len(value) for value in self._atoms.values())

    def __setitem__(self, key: type[AtomicFormula], entry: set[AtomicFormula]) -> None:
        self._atoms[key] = entry

    def __repr__(self) -> str:
        parts = [f'{rel.__name__}: {self[rel]}'
                 for rel in (Eq, Ne, Gt, Lt)
                 if self[rel]]
        return f'Clause({", ".join(parts)})'

    def add(self, atom: AtomicFormula) -> None:
        rel = type(atom)
        if rel in (Ge, Le):
            strict_part = rel.strict_part()
            self[strict_part].add(strict_part(atom.lhs, 0))
            self[Eq].add(Eq(atom.lhs, 0))
        else:
            self[rel].add(atom)

    def as_atom(self) -> Optional[AtomicFormula]:
        if len(self) == 0:
            return None
        elif len(self) == 1:
            for rel in (Eq, Ne, Gt, Lt):
                if len(self[rel]) == 1:
                    return next(iter(self[rel]))
            assert False, f'Unexpected clause: {self}'
        elif len(self) == 2 and len(self[Eq]) == 1 and len(self[Gt]) == 1:
            eq_lhs = next(iter(self[Eq])).lhs
            gt_lhs = next(iter(self[Gt])).lhs
            if eq_lhs == gt_lhs:
                return Ge(eq_lhs, 0)
            else:
                return None
        elif len(self) == 2 and len(self[Eq]) == 1 and len(self[Lt]) == 1:
            eq_lhs = next(iter(self[Eq])).lhs
            lt_lhs = next(iter(self[Lt])).lhs
            if eq_lhs == lt_lhs:
                return Le(eq_lhs, 0)
            else:
                return None
        elif all(len(self[rel]) == 0 for rel in (Ne, Gt, Lt)):
            return Eq(self.product_of(Eq), 0)
        else:
            return None

    def copy(self) -> Clause:
        new_clause = Clause()
        for rel in (Eq, Ne, Gt, Lt):
            new_clause[rel] = self[rel].copy()
        return new_clause

    def get(self, rel: type[AtomicFormula]) -> set[AtomicFormula]:
        return self._atoms.get(rel, set())

    def product_of(self, rel: type[AtomicFormula]) -> Term:
        return prod((atom.lhs for atom in self[rel]), start=Term(1))

    def term_list_of(self, rel: type[AtomicFormula]) -> list[Term]:
        return [atom.lhs for atom in self[rel]]

    def to_formula(self) -> Formula:
        return Or(*set().union(*self._atoms.values()))


@dataclass(init=False)
class GlobalPremise:

    _atoms: dict[type[AtomicFormula], set[AtomicFormula]]
    _basis: set[Term]
    _have_gbasis: bool
    _options: Options

    @property
    def assume(self) -> list[AtomicFormula]:
        return sorted(set().union(*self._atoms.values()))

    @property
    def gbasis(self) -> list[Term]:
        if not self._have_gbasis:
            self._gbasis = Term.gbasis(self._basis, radical=self._options.radical)
            self._have_gbasis = True
        return self._gbasis

    def __getitem__(self, key: type[AtomicFormula]) -> set[AtomicFormula]:
        return self._atoms[key]

    def __init__(self, assume: Iterable[AtomicFormula], options: Options) -> None:
        self._atoms = {Eq: set(), Ne: set(), Ge: set(), Gt: set(), Le: set(), Lt: set()}
        self._basis = set()
        for atom in assume:
            key = type(atom)
            self._atoms[key].add(atom)
            if key is Eq:
                self._basis.add(atom.lhs)
        self._have_gbasis = False
        self._options = options

    def add(self, atom: AtomicFormula) -> None:
        key = type(atom)
        self._atoms[key].add(atom)
        if key is Eq:
            self._basis.add(atom.lhs)
            self._have_gbasis = False

    def product_of(self, rels: type[AtomicFormula] | tuple[type[AtomicFormula], ...]) -> Term:
        if not isinstance(rels, tuple):
            rels = (rels,)
        return prod((atom.lhs for rel in rels for atom in self[rel]), start=Term(1))

    def term_list_of(self, rels: type[AtomicFormula] | tuple[type[AtomicFormula], ...]) -> list[Term]:
        if not isinstance(rels, tuple):
            rels = (rels,)
        return [atom.lhs for rel in rels for atom in self[rel]]


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
            return [Clause(cast(Or, f))], negated

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
            clauses.append(Clause(arg))
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

        # Pass 1: Identify equational clauses and add them to the global premise,
        # unless already entailed.
        logging.info(f'processing equational clauses ({count} clauses left)')
        global_premise = GlobalPremise(assume, self._options)
        for clause in clauses:
            atom = clause.as_atom()
            if atom is None:
                continue
            logging.debug(f'{count} clauses left')
            count -= 1
            if isinstance(atom, Eq):
                h = atom.lhs.reduce(global_premise.gbasis)
                simplified_equation = simplify(Eq(h, 0), assume=global_premise.assume)
                if isinstance(simplified_equation, _T):
                    continue
                if isinstance(simplified_equation, _F):
                    return [Clause()]
            global_premise.add(atom)
            new_clauses.append(clause)

        # Pass 2: Simplify the other clauses
        logging.info(f'processing non-equational clauses ({count} clauses left)')
        for clause in clauses:
            if clause.as_atom() is not None:
                continue
            logging.debug(f'{count} clauses left')
            count -= 1
            # We decide against filtering out disequalities via a
            # radical membership test at this point. The idea is that
            # adding their left hand sides to the ideal brings it closer
            # to the radical, which is good for our purposes.
            clause_gbasis = Term.gbasis(global_premise.gbasis + clause.term_list_of(Ne), radical=self._options.radical)

            new_clause = Clause()

            # 1. Build the left hand side atoms of the new clause:
            H = set()
            for atom in clause[Ne]:  # !*rlgsred=T
                h = atom.lhs.reduce(global_premise.gbasis)
                H.add(h)
            G = Term.gbasis(H, radical=self._options.radical)
            new_clause[Ne] = {Ne(g, 0) for g in G}  # !*rlgssub=T

            # 2. Determine the relevant right hand side relations of the clause.
            # Eq may be dropped via a test on the product of left hand sides of Eq taken from
            # clause and Ne, Gt, Lt taken from the global premise. Recall that weak inequalities
            # are split and do not occur explicitly.
            product = clause.product_of(Eq) * global_premise.product_of((Ne, Gt, Lt))
            product = product.reduce(global_premise.gbasis)
            test = simplify(Eq(product, 0), assume=global_premise.assume)
            if isinstance(test, _T):
                continue
            elif isinstance(test, _F):
                rhs_rels = [Gt, Lt]
            else:
                rhs_rels = [Eq, Gt, Lt]

            # 3. Build the right hand side atoms of the new clause.
            for rel in rhs_rels:
                for atom in clause[rel]:
                    h = atom.lhs.reduce(clause_gbasis)
                    new_clause[rel].add(rel(h, 0))  # !*rlgsred=T

            # Starting with `new clause`, we compute a `test_clause`. If
            # `test_clause` is recoginizably equivalent to T, then `new_clause`
            # is redundant modulo `global_premise.assume` and thus dropped.
            test_clause = new_clause.copy()

            # Patch the disequalities
            test_clause[Ne] = set()
            for f in clause_gbasis:
                test_clause.add(Ne(f, 0))

            # Add `product == 0`
            test_clause.add(Eq(product, 0))

            # Add further redundant information from `global_premise`
            for rel in (Ne, Ge, Gt, Le, Lt):
                for f in global_premise.term_list_of(rel):
                    h = f.reduce(clause_gbasis)
                    test_clause.add(rel.complement()(h, 0))

            # Test `test_clause`
            test_formula = simplify(test_clause.to_formula())
            assert test_formula is not _F
            if isinstance(test_formula, _T):
                continue

            new_clauses.append(new_clause)

        return new_clauses

    def _in_radical(self, f: Term, G: list[Term]) -> bool:
        # Currently unsused. With `Y = VV.fresh()` there would be efficiency
        # problems with a growing SAGE polynomial ring.
        Y = VV['SecretRabinovichVariable']
        return Term(1) in Term.gbasis(G + [1 - Y * f])


def gsimplify(f: Formula, assume: Iterable[AtomicFormula] =[], **options) -> Formula:
    return GSimplify(Options(**options)).gsimplify(f, assume)
