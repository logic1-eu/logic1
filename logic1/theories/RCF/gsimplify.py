"""Gröbner simplification of formulas in the theory of real closed fields (RCF).
The implementation follows but is not limited to ideas discussed in
[DolzmannSturm-1997]_.
"""

from dataclasses import dataclass, field
import logging
from itertools import chain
from math import prod
from typing import cast, final, Iterable, Optional, Sequence

from logic1.firstorder import And, _F, Not, Or, _T
from logic1.abc.simplify import InternalRepresentation
from logic1.theories.RCF import redlog
from logic1.theories.RCF.atomic import AtomicFormula, Eq, Ge, Gt, Le, Lt, Ne
from logic1.theories.RCF.bnf import cnf
from logic1.theories.RCF.simplify import simplify
from logic1.theories.RCF.term import Term
from logic1.theories.RCF.types import Formula


@dataclass(frozen=True)
class Options:
    use_redlog_cnf: bool = True
    """If :obj:`True`, use :func:`.redlog.cnf` for CNF computation. If
    :obj:`False`, use :func:`.RCF.bnf.cnf`, which is based on PyEda.
    """

    bnfsac: bool = True
    """The value to be used for the option ``bnfsac`` (Boolean normal form with
    subsumption and cut) when calling :func:`.redlog.cnf`.
    """

    bnfsm: bool = True
    """The value to be used for the option ``bnfsm`` (Boolean normal form smart)
    when calling :func:`.redlog.cnf`.
    """

    radical: bool = True
    r"""This option applies whenever a Gröbner basis is computed for a given set
    of generators :math:`f_1, \dots, f_n`. If :obj:`False`, compute a Gröbner
    basis of the ideal :math:`\langle f_1, \dots, f_n \rangle`. If :obj:`True`,
    compute a Gröbner basis of the radical ideal :math:`\sqrt{\langle f_1,
    \dots, f_n \rangle}`. The latter leads to stronger simplification at the
    cost of more expensive Gröbner basis computations.
    """


type ClauseAtom = Eq | Ne | Gt | Lt
"""An :class:`AtomicFormula` that can occur in a :class:`.Clause`. The types
:class:`Ge` and :class:`Le` are not included, because they are represented by
their strict parts :class:`Gt` and :class:`Lt`, respectively, together with the
corresponding :class:`Eq` atoms.
"""


@final
@dataclass(init=False)
class Clause:
    """A clause is a disjunction of atomic formulas.

    It is represented as a dictionary mapping the types :class:`Eq`,
    :class:`Ne`, :class:`Gt`, and :class:`Lt` of atomic formulas to sets of
    corresponding atomic formulas. The types :class:`Ge` and :class:`Le` are
    not stored explicitly, but are represented by their strict parts
    :class:`Gt` and :class:`Lt`, respectively, together with the corresponding
    :class:`Eq` atoms.
    """

    _atoms: dict[type[ClauseAtom], set[ClauseAtom]]

    def __getitem__(self, key: type[ClauseAtom]) -> set[ClauseAtom]:
        """Return the set of atoms of type ``key`` in the clause.
        """
        return self._atoms[key]

    def __init__(self, f: Optional[Or | AtomicFormula | _F] = None) -> None:
        """Initialize a clause from a formula ``f``.

        ``f`` must be either a disjunction of atomic formulas, a single atomic
        formula, the truth value :obj:`.F`, or :obj:`None`. Both :obj:`.F` and
        :obj:`None` create an empty clause.

        Raises :exc:`AssertionError` if ``f`` does not meet this specification.
        """
        self._atoms = {Eq: set(), Ne: set(), Gt: set(), Lt: set()}
        if isinstance(f, Or):
            assert all(isinstance(arg, AtomicFormula) for arg in f.args)
            args = f.args
        elif isinstance(f, AtomicFormula):
            args = (f,)
        elif isinstance(f, _F) or f is None:
            args = ()
        else:
            assert False, f
        for arg in args:
            self.add(arg)

    def __iter__(self):
        """Iterate over all atoms in the clause.
        """
        for rel in (Eq, Ne, Gt, Lt):
            yield from self[rel]

    def __len__(self) -> int:
        """Return the number of atoms in the clause.
        """
        return sum(len(value) for value in self._atoms.values())

    def __setitem__(self, key: type[ClauseAtom], entry: set[ClauseAtom]) -> None:
        """Set the set of atoms of type ``key`` in the clause to ``entry``.
        """
        self._atoms[key] = entry

    def __repr__(self) -> str:
        """Return a string representation of the clause.

        The output contains valid code for creating the clause.
        """
        parts = [f'{rel.__name__}: {self[rel]}'
                 for rel in (Eq, Ne, Gt, Lt)
                 if self[rel]]
        return f'Clause({", ".join(parts)})'

    def add(self, atom: AtomicFormula) -> None:
        """Add an atomic formula ``atom`` to the clause.
        """
        rel = type(atom)
        if rel in (Ge, Le):
            strict_part = rel.strict_part()
            self[strict_part].add(strict_part(atom.lhs, 0))
            self[Eq].add(Eq(atom.lhs, 0))
        else:
            self[cast(type[ClauseAtom], rel)].add(cast(ClauseAtom, atom))

    def as_atom(self) -> AtomicFormula:
        """Return the clause as a single atomic formula.

        Raise :exc:`ValueError` if this is not possible.
        """
        if not self.is_atomic():
            raise ValueError(f'Clause {self} is not atomic')
        if len(self) == 1:
            for rel in (Eq, Ne, Gt, Lt):
                if len(self[rel]) == 1:
                    return next(iter(self[rel]))
            assert False
        if self.is_equational():
            return Eq(self.product_of(Eq), 0)
        eq_lhs = next(iter(self[Eq])).lhs
        xt = Gt if len(self[Gt]) == 1 else Lt
        return xt(eq_lhs, 0)

    def copy(self) -> Clause:
        """Return a copy of the clause, copying the sets but not their elements.
        """
        new_clause = Clause()
        for rel in (Eq, Ne, Gt, Lt):
            new_clause[rel] = self[rel].copy()
        return new_clause

    def is_atomic(self) -> bool:
        """Return :obj:`True` if the clause can be represented as a single
        atomic formula, :obj:`False` otherwise.
        """
        if len(self) == 0:
            return False
        if len(self) == 1:
            return True
        if self.is_equational():
            return True
        if len(self) > 2:
            return False
        assert len(self) == 2
        if len(self[Eq]) != 1:
            return False
        for xt in (Gt, Lt):
            if len(self[xt]) == 1:
                eq_lhs = next(iter(self[Eq])).lhs
                xt_lhs = next(iter(self[xt])).lhs
                if eq_lhs == xt_lhs:
                    return True
        return False

    def is_empty(self) -> bool:
        """Return :obj:`True` if the clause is empty, :obj:`False` otherwise.
        """
        return len(self) == 0

    def is_equational(self) -> bool:
        """Return :obj:`True` if the clause is equational, :obj:`False`
        otherwise.

        An equational clause contains at least two equations and no atoms of
        any other type. It can be represented by a single equation using the
        product of the left hand sides of its equations.
        """
        return len(self) > 1 and all(len(self[rel]) == 0 for rel in (Ne, Gt, Lt))

    def product_of(self, rel: type[ClauseAtom]) -> Term:
        """Return the product of the left hand sides of all atoms of type
        ``rel`` in the clause.
        """
        return prod((atom.lhs for atom in self[rel]), start=Term(1))

    def term_list_of(self, rel: type[ClauseAtom]) -> list[Term]:
        """Return a list of the left hand sides of all atoms of type ``rel`` in
        the clause.
        """
        return [atom.lhs for atom in self[rel]]

    def to_formula(self) -> Formula:
        """Return the clause as a formula.
        """
        return Or(*set().union(*self._atoms.values()))


@dataclass(init=False)
class GlobalPremise:
    """A global premise represents a set of atomic formulas used to simplify
    other formulas. As a set of atomic formulas, it can be used as the
    ``assume`` argument of :func:`.RCF.simplify.simplify`. It furthermore
    provides a Gröbner basis of the ideal or radical ideal (depending on the
    option :attr:`.Options.radical`) generated by the left hand sides of its
    equations, which can be used to simplify other formulas by Gröbner reduction
    of the left hand sides of their atomic formulas.
    """

    _atoms: dict[type[AtomicFormula], set[AtomicFormula]]
    _basis: set[Term]
    _have_gbasis: bool
    _options: Options

    def assume(self, gbasis: Sequence[Term]) -> list[AtomicFormula]:
        """Return a list of atomic formulas comprising

          1. the atomic formulas of the global premise,
          2. the atomic formulas obtained by reducing the left hand sides of
             the global premise's atomic formulas reduced modulo the Gröbner
             basis ``gbasis``,
          3. the equations corresponding to ``gbasis``.
        """
        assumption = set(chain.from_iterable(self._atoms.values()))
        base = assumption.copy()
        assumption.update(atom.op(atom.lhs.reduce(gbasis), 0) for atom in base)
        assumption.update(Eq(f, 0) for f in gbasis)
        return sorted(assumption)

    @property
    def gbasis(self) -> list[Term]:
        """Return a Gröbner basis of the ideal or radical ideal generated by the
        left hand sides of the global premise's equations.
        """
        if not self._have_gbasis:
            self._gbasis = Term.gbasis(self._basis, radical=self._options.radical)
            self._have_gbasis = True
        return self._gbasis

    def __getitem__(self, key: type[AtomicFormula]) -> set[AtomicFormula]:
        """Return the set of atoms of type ``key`` in the global premise.
        """
        return self._atoms[key]

    def __init__(self, assume: Iterable[AtomicFormula], options: Options) -> None:
        """Initialize a global premise from a set of atomic formulas ``assume``.
        """
        self._atoms = {
            Eq: set(),
            Ne: set(),
            Ge: set(),
            Gt: set(),
            Le: set(),
            Lt: set(),
        }
        self._basis = set()
        for atom in assume:
            key = type(atom)
            self._atoms[key].add(atom)
            if key is Eq:
                self._basis.add(atom.lhs)
        self._have_gbasis = False
        self._options = options

    def add(self, atom: AtomicFormula) -> None:
        """Add an atomic formula ``atom`` to the global premise.
        """
        key = type(atom)
        self._atoms[key].add(atom)
        if key is Eq:
            self._basis.add(atom.lhs)
            self._have_gbasis = False

    def product_of(self, rels: type[AtomicFormula] | tuple[type[AtomicFormula], ...]) -> Term:
        """Return the product of the left hand sides of all atoms of types in
        ``rels`` in the global premise.
        """
        if not isinstance(rels, tuple):
            rels = (rels,)
        return prod((atom.lhs for rel in rels for atom in self[rel]), start=Term(1))

    def term_list_of(self, rels: type[AtomicFormula] | tuple[type[AtomicFormula], ...]) -> list[Term]:
        """Return the list of the left hand sides of all atoms of types in
        ``rels`` in the global premise.
        """
        if not isinstance(rels, tuple):
            rels = (rels,)
        return [atom.lhs for rel in rels for atom in self[rel]]


@dataclass
class GSimplify:

    _options: Options = field(default_factory=Options)

    class Inconsistent(Exception):
        pass

    def __call__(self, f: Formula, assume: Iterable[AtomicFormula] = []) -> Formula:
        """Gröbner-simplify ``f`` modulo ``assume``. Raise :exc:`Inconsistent`
        if ``assume`` is detected to be inconsistent.
        """
        assume_as_clauses_neg = [Clause(Or(*(atom.to_complement() for atom in assume)))]
        assume_as_clauses_neg = self.gsimplify_clauses(assume_as_clauses_neg, assume=[])
        if len(assume_as_clauses_neg) == 0:
            raise self.Inconsistent()

        f = simplify(f, assume=assume)
        if isinstance(f, (AtomicFormula, _T, _F)):
            return f

        # Negation is used as a heuristic when the top-level operator is Or.
        if isinstance(f, Or):
            f = Not(f).to_nnf()
            negated = True
        else:
            negated = False

        clauses = self.formula_to_clauses(f)
        clauses = self.gsimplify_clauses(clauses, assume=assume)
        f = And(*(clause.to_formula() for clause in clauses))
        # Although f is a CNF, we recompute a CNF in order to take advantage of
        # subsumption and cut.
        f = self.cnf(f)

        if negated:
            f = Not(f)

        # Now f is essentially in either CNF or DNF. The final simplification
        # will not preserve this in general.
        f = simplify(f, assume=assume)
        return f

    def cnf(self, f: Formula) -> Formula:
        """Return a conjunctive normal form of ``f``. Depending on the option
        :attr:`.Options.use_redlog_cnf`, the CNF is computed using either
        :func:`.redlog.cnf` or :func:`.RCF.bnf.cnf`, which is based on
        [PyEDA](https://pyeda.readthedocs.io).
        """
        if self._options.use_redlog_cnf:
            logging.info('computing cnf using Redlog with '
                         f'{self._options.bnfsac=}, '
                         f'{self._options.bnfsm=} ...')
            return redlog.cnf(f, bnfsac=self._options.bnfsac,
                                 bnfsm=self._options.bnfsm)
        else:
            logging.info('computing cnf using PyEda ...')
            return cnf(f)

    def formula_to_clauses(self, f: Formula) -> list[Clause]:
        """Convert a formula to a list of clauses.
        """
        assert isinstance(f, (And, Or))

        f = self.cnf(f)
        logging.info(f'cnf has {len(list(f.atoms()))} atoms')
        assert isinstance(f, And) and not all(isinstance(arg, AtomicFormula) for arg in f.args)
        clauses = []
        for arg in f.args:
            clauses.append(Clause(arg))
        return clauses

    def gsimplify_clauses(self, clauses: list[Clause], assume: Iterable[AtomicFormula]) -> list[Clause]:
        """Gröbner-simplify a list of clauses modulo ``assume``.
        """

        count = len(clauses)

        strong_global_premise = GlobalPremise(assume, self._options)
        weak_global_premise = GlobalPremise(assume, self._options)

        # Step 1: Split the clauses into atoms and proper clauses
        logging.info(f'splitting clauses ({count} input clauses left)')
        proper_clauses = []
        atoms = []
        for clause in clauses:
            if clause.is_equational():
                proper_clauses.append(clause)
                atom = clause.as_atom()
                strong_global_premise.add(atom)
            elif clause.is_atomic():
                count -= 1
                logging.debug(f'{count} input clauses left')
                atom = clause.as_atom()
                atoms.append(atom)
                weak_global_premise.add(atom)
                strong_global_premise.add(atom)
            else:
                proper_clauses.append(clause)

        new_clauses = []

        # Step 2: Simplify the atoms, if present
        if len(atoms) > 0:
            logging.info(f'processing atoms ({count} input clauses left)')
            atoms_as_clauses_neg = [Clause(Or(*(atom.to_complement() for atom in atoms)))]
            atoms_as_clauses_neg = self.gsimplify_clauses(atoms_as_clauses_neg, assume=assume)
            if len(atoms_as_clauses_neg) == 0:
                return [Clause()]
            assert len(atoms_as_clauses_neg) == 1
            for atom_neg in atoms_as_clauses_neg[0]:
                atom = atom_neg.to_complement()
                if isinstance(atom, Eq):
                    _, factors = atom.lhs.factor()
                    new_clauses.append(Clause(Or(*(Eq(factor, 0) for factor in factors))))
                else:
                    new_clauses.append(Clause(atom))

        # Step 3: Simplify the proper clauses
        logging.info(f'processing proper clauses ({count} input clauses left)')
        for clause in proper_clauses:
            count -= 1
            logging.debug(f'{count} input clauses left')

            if clause.is_equational():
                global_premise = weak_global_premise
            else:
                global_premise = strong_global_premise

            new_clause = Clause()

            # 1. Build the atoms of /\ Eq for the new clause /\ Eq -> \/ (Gt, Lt, Eq)
            F = set()
            for atom in clause[Ne]:
                f = atom.lhs.reduce(global_premise.gbasis)
                F.add(f)
            G = Term.gbasis(F, radical=self._options.radical)
            H = set()
            for g in G:
                h = g.reduce(global_premise.gbasis)
                H.add(h)
            new_clause[Ne] = {Ne(h, 0) for h in H}

            # 2. For the simplification of \/ (Gt, Lt, Eq) we use a Gröbner basis of /\ Eq together
            # with the equations of `global_premise`.
            clause_gbasis = Term.gbasis(global_premise.gbasis + clause.term_list_of(Ne),
                                        radical=self._options.radical)
            clause_assume = global_premise.assume(gbasis=clause_gbasis)

            # 2.1. Simplify \/ Eq by considering the product of its left hand sides plus certain
            # left hand sides of the global premise. We might discover T or redundancy of \/ Eq.
            product = clause.product_of(Eq) * global_premise.product_of((Ne, Gt, Lt))
            product = product.reduce(clause_gbasis)
            try:
                test_formula = simplify(Eq(product, 0), assume=clause_assume)
            except InternalRepresentation.Inconsistent:
                # The general idea of theory simplification is the equivalence
                #
                #     /\ Θ -> (φ <-> /\ Θ /\ φ).
                #
                # More generally, θ can be conjunctively added to any subformula
                # of φ. In our situation, we have
                #
                #     /\ `global_premise` -> ( /\ Eq -> \/ (Gt, Lt, Eq) <->
                #                              /\ (`global_premise`, Eq) -> \/ (Gt, Lt, Eq) )
                #
                # and we have heuristically detected that
                #
                #     /\ (`global_premise`, Eq)
                #
                # is inconsistent. Note, however, that /\ Eq is not explicitly
                # present but enters the computation via Gröbner reductions
                # during the computation of
                #
                #     `assume=global_premise.assume(gbasis=clause_gbasis)`.
                #
                # We conclude that `clause` is redundant modulo `global_premise`.
                continue
            if isinstance(test_formula, _T):
                continue
            # Recall that weak inequalities have been split and do not occur explicitly.
            if isinstance(test_formula, _F):
                rhs_rels: list[type[ClauseAtom]] = [Gt, Lt]
            else:
                rhs_rels = [Gt, Lt, Eq]

            # 2.2. Simplify \/ (Gt, Lt) or \/ (Gt, Lt, Eq), depending on the previous if
            for rel in rhs_rels:
                for atom in clause[rel]:
                    h = atom.lhs.reduce(clause_gbasis)
                    new_clause[rel].add(rel(h, 0))  # !*rlgsred=T

            # This concludes the computation of `new_clause`; simplify it
            and_eq = new_clause[Ne]
            new_clause[Ne] = set()
            new_clause_formula = new_clause.to_formula()
            try:
                # The following simplification drops \/ new_clause[Ne]
                new_clause_formula = simplify(new_clause_formula, assume=clause_assume,
                                                                  explode_always=False)
            except InternalRepresentation.Inconsistent:
                # With the same reasoning as in the previous exception handling,
                # we conclude that `clause` is redundant modulo
                # `global_premise`.
                continue
            assert isinstance(new_clause_formula, (AtomicFormula, Or, _T, _F)) and new_clause_formula.depth() <= 1
            if isinstance(new_clause_formula, _T):
                continue
            # if isinstance(new_clause_formula, _F), then we must not return
            # [Clause()], because `and_eq` would be dropped.
            new_clause = Clause(new_clause_formula)
            # Restore \/ new_clause[Ne]
            new_clause[Ne] = and_eq
            new_clauses.append(new_clause)

        return new_clauses

def gsimplify(f: Formula, assume: Iterable[AtomicFormula] =[], **options) -> Formula:
    """Gröbner-simplify ``f`` modulo ``assume``. Raise
    :exc:`GSimplify.Inconsistent` if ``assume`` is detected to be inconsistent.
    """
    return GSimplify(Options(**options))(f, assume)
