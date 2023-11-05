import logging

from functools import lru_cache
from operator import xor
from sage.all import oo, Rational  # type: ignore
from sage.rings.infinity import MinusInfinity, PlusInfinity  # type: ignore
from typing import Iterable, Optional, Self, TypeAlias

from . import rcf  # need qualified names of relations for pattern matching
from ... import abc

from ...firstorder.formula import Formula
from ...firstorder.boolean import And, Or
from ...firstorder.atomic import AtomicFormula
from ...firstorder.truth import T, F
from .rcf import (BinaryAtomicFormula, RcfAtomicFormula, RcfAtomicFormulas,
                  Term, Variable, Ring, Eq, Ne, Ge, Le, Gt, Lt)
from .pnf import pnf

from ...support.tracing import trace  # noqa


logging.basicConfig(
    format='%(levelname)s[%(filename)s:%(lineno)d]: %(message)s',
    level=logging.CRITICAL)


class Theory(abc.simplify.Theory):

    class _Interval:
        # Non-empty real intervals. Raises Inconsistent when an empty interval
        # is created.

        def __contains__(self, q: Rational) -> bool:
            assert isinstance(q, Rational)
            if q < self.start or (self.lopen and q == self.start):
                return False
            if self.end < q or (self.ropen and q == self.end):
                return False
            return True

        def __init__(self, lopen: bool = False, start: Rational | MinusInfinity = -oo,
                     end: Rational | PlusInfinity = oo, ropen: bool = False) -> None:
            assert lopen or start is not -oo
            assert ropen or end is not oo
            if start > end:
                raise Theory.Inconsistent
            if (lopen or ropen) and start == end:
                raise Theory.Inconsistent
            self.lopen = lopen
            self.start = start
            self.end = end
            self.ropen = ropen

        def intersection(self, other: Self) -> Self:
            if self.start < other.start:
                lopen = other.lopen
                start = other.start
            elif other.start < self.start:
                lopen = self.lopen
                start = self.start
            else:
                assert self.start == other.start
                start = self.start
                lopen = self.lopen or other.lopen
            if self.end < other.end:
                end = self.end
                ropen = self.ropen
            elif other.end < self.end:
                end = other.end
                ropen = other.ropen
            else:
                assert self.end == other.end
                end = self.end
                ropen = self.ropen or other.ropen
            return self.__class__(lopen, start, end, ropen)

        def is_point(self) -> bool:
            return self.start == self.end

        def __repr__(self):
            left = '(' if self.lopen else '['
            right = ')' if self.ropen else ']'
            return f'{left}{self.start}, {self.end}{right}'

    _reference: dict[Term, tuple[_Interval, set]]
    _current: dict[Term, tuple[_Interval, set]]

    def __repr__(self):
        return f'Theory({self._reference}, {self._current})'

    def add(self, gand: type[And] | type[Or], atoms: Iterable[AtomicFormula]) -> None:
        for atom in atoms:
            # rel is the relation of atom, p is the parametric part, and q is
            # the negative of the Rational absolute summand.
            rel, p, q = self._decompose_atom(atom)
            if gand is Or:
                rel = rel.complement_func
            match rel:
                # We model p in ivl \ exc.
                #
                # Compare https://stackoverflow.com/q/71441761/ which suggests
                # the use of __qualname__ here.
                case rcf.Eq:
                    ivl = Theory._Interval(False, q, q, False)
                    exc = set()
                case rcf.Ne:
                    ivl = Theory._Interval(True, -oo, oo, True)
                    exc = {q}
                case rcf.Ge:
                    ivl = Theory._Interval(False, q, oo, True)
                    exc = set()
                case rcf.Le:
                    ivl = Theory._Interval(True, -oo, q, False)
                    exc = set()
                case rcf.Gt:
                    ivl = Theory._Interval(True, q, oo, True)
                    exc = set()
                case rcf.Lt:
                    ivl = Theory._Interval(True, -oo, q, True)
                    exc = set()
                case _:
                    assert False
            if p in self._current:
                cur_ivl, cur_exc = self._current[p]
                ivl = ivl.intersection(cur_ivl)
                exc = exc.union(cur_exc)
                # Restrict exc to the new ivl.
                exc = {x for x in exc if x in ivl}
                # Note that exc is a subset of ivl now. Fix the case that ivl
                # is closed on either side and the corresponding endpoint is in
                # exc. We are going to use inf and sup in contrast to start and
                # end, because ivl can be a FiniteSet.
                if ivl.start in exc:
                    # It follows that ivl is left-closed. Theory._Interval raises
                    # Inconsonsitent if ivl gets empty.
                    ivl = Theory._Interval(True, ivl.start, ivl.end, ivl.ropen)
                    exc = exc.difference({ivl.start})
                if ivl.end in exc:
                    # It follows that ivl is right-closed. ivl cannot get emty
                    # here.
                    ivl = Theory._Interval(ivl.lopen, ivl.start, ivl.end, True)
                    exc = exc.difference(ivl.end)
            self._current[p] = (ivl, exc)

    def __init__(self, prefer_weak: bool, prefer_order: bool, _develop: int) -> None:
        self.prefer_weak = prefer_weak
        self.prefer_order = prefer_order
        self._develop = _develop
        self._reference = dict()
        self._current = dict()

    @staticmethod
    @lru_cache(maxsize=None)
    def _compose_atom(rel: type[BinaryAtomicFormula], p: Term, q: Rational) -> AtomicFormula:
        num = q.numerator()
        den = q.denominator()
        return rel(den * p - num, Ring(0), chk=False)

    @staticmethod
    @lru_cache(maxsize=None)
    def _decompose_atom(f: BinaryAtomicFormula) -> tuple[type[AtomicFormula], Term, Rational]:
        """Decompose into relation :math:`\rho`, term :math:`p` without
        absolute summand, and rational :math:`q` such that :data:`f` is
        equivalent to :math:`p \rho q`.

        We assume that :data:`f` has gone through :meth:`_simpl_at` so that its
        right hand side is zero and its left hand side polynomial has gone
        through SymPy's :meth:`expand`.

        >>> from .rcf import var
        >>> a, b = var.set('a', 'b')
        >>> f = Le(6*a**2 + 12*a*b + 6*b**2 + 3, 0)
        >>> rel, p, q = Theory._decompose_atom(f); rel, p, q
        (<class 'logic1.theories.RCF.rcf.Le'>, a^2 + 2*a*b + b^2, -1/2)
        >>> g = Theory._compose_atom(rel, p, q); g
        Le(2*a^2 + 4*a*b + 2*b^2 + 1, 0)
        >>> (f.lhs / g.lhs)
        3
        """
        q = - f.lhs.constant_coefficient()
        p = f.lhs + q
        c = p.content()
        p, _ = p.quo_rem(c)
        # Given that _simpl_at has procuces a monic polynomial, q != 0 will not
        # be divisible by c. This is relevant for the reconstruction in
        # _compose_atom to work.
        assert c == 1 or not c.divides(q), f'{c} divides {q}'
        q = q / c
        return f.func, p, q

    def extract(self, gand: type[And] | type[Or]) -> list[AtomicFormula]:
        L: list[AtomicFormula] = []
        for p in self._current:
            if p in self._reference:
                ref_ivl, ref_exc = self._reference[p]
            else:
                ref_ivl, ref_exc = Theory._Interval(True, -oo, oo, True), set()
            ivl, exc = self._current[p]
            # ivl cannot be empty because the construction of an empty interval
            # raises an Exception in `add`.
            if ivl.is_point():
                if ref_ivl.is_point():
                    assert ref_ivl.start == ivl.start, f'{ref_ivl} != {ivl}'
                else:
                    # Pick the one point of ivl.
                    q = ivl.start
                    # When gand is And, the equation q = 0 is generally
                    # preferable. Otherwise, q = 0 would become q != 0
                    # via subsequent negation, and we want to take
                    # self.prefer_order into consideration.
                    if self.prefer_order and gand is Or:
                        if q == ref_ivl.start:
                            assert not ref_ivl.lopen
                            L.append(self._compose_atom(Le, p, q))
                        elif q == ref_ivl.end:
                            assert not ref_ivl.ropen
                            L.append(self._compose_atom(Ge, p, q))
                        else:
                            L.append(self._compose_atom(Eq, p, q))
                    else:
                        L.append(self._compose_atom(Eq, p, q))
            else:
                # We know that ref_ivl is a proper interval, too, because ivl
                # is a subset of ref_ivl.
                assert not ref_ivl.is_point()
                if ref_ivl.start < ivl.start:
                    if ivl.start in ref_exc:
                        # When gand is Or, weak and strong are dualized via
                        # subsequent negation.
                        if xor(self.prefer_weak, gand is Or):
                            L.append(self._compose_atom(Ge, p, ivl.start))
                        else:
                            L.append(self._compose_atom(Gt, p, ivl.start))
                    else:
                        if ivl.lopen:
                            L.append(self._compose_atom(Gt, p, ivl.start))
                        else:
                            L.append(self._compose_atom(Ge, p, ivl.start))
                elif ref_ivl.start == ivl.start:
                    if not ref_ivl.lopen and ivl.lopen:
                        # When gand is Or, Ne will become Eq via subsequent
                        # nagation. This is generally preferable.
                        if self.prefer_order and gand is And:
                            L.append(self._compose_atom(Gt, p, ivl.start))
                        else:
                            L.append(self._compose_atom(Ne, p, ivl.start))
                if ivl.end < ref_ivl.end:
                    if ivl.end in ref_exc:
                        # When gand is Or, weak and strong are dualized via
                        # subsequent negation.
                        if xor(self.prefer_weak, gand is Or):
                            L.append(self._compose_atom(Le, p, ivl.end))
                        else:
                            L.append(self._compose_atom(Lt, p, ivl.end))
                    else:
                        if ivl.ropen:
                            L.append(self._compose_atom(Lt, p, ivl.end))
                        else:
                            L.append(self._compose_atom(Le, p, ivl.end))
                elif ref_ivl.end == ivl.end:
                    if not ref_ivl.ropen and ivl.ropen:
                        # When gand is Or, Ne will become Eq via subsequent
                        # nagation. This is generally preferable.
                        if self.prefer_order and gand is And:
                            L.append(self._compose_atom(Lt, p, ivl.end))
                        else:
                            L.append(self._compose_atom(Ne, p, ivl.end))
            for q in exc:
                if q not in ref_exc:
                    L.append(self._compose_atom(Ne, p, q))
        if gand is Or:
            L = [atom.to_complement() for atom in L]
        return L

    def next_(self, remove: Optional[Variable] = None) -> Self:
        theory_next = self.__class__(self.prefer_weak, self.prefer_order, self._develop)
        if remove is None:
            theory_next._reference = self._current
        else:
            theory_next._reference = {p: q for p, q in self._current.items()
                                      if remove not in p.variables()}
        theory_next._current = theory_next._reference.copy()
        return theory_next


class Simplify(abc.simplify.Simplify['Theory']):

    AtomicSortKey: TypeAlias = tuple[int, Term]
    SortKey: TypeAlias = tuple[int, int, int, tuple[AtomicSortKey, ...]]

    def __call__(self, f: Formula, assume: list[AtomicFormula] = [],
                 prefer_weak: bool = False, prefer_order: bool = True,
                 log=logging.CRITICAL, _develop: int = 0) -> Formula:
        self.prefer_weak = prefer_weak
        self.prefer_order = prefer_order
        self._develop = _develop
        level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(log)
        try:
            result = self.simplify(f, assume)
        except Simplify.NotInPnf:
            result = self.simplify(pnf(f), assume)
        logging.getLogger().setLevel(level)
        return result

    @lru_cache(maxsize=None)
    def _simpl_at(self, f: AtomicFormula) -> Formula:
        """
        >>> from .rcf import var
        >>> a, b = var.set('a', 'b')
        >>> simplify(Le(-6 * (a+b)**2 + 3, 0))
         Ge(2*a^2 + 4*a*b + 2*b^2 - 1, 0)
        """
        assert isinstance(f, RcfAtomicFormulas)
        lhs = f.lhs - f.rhs
        if lhs.is_constant():
            _python_operator = f.sage_func  # type: ignore
            eval_ = _python_operator(lhs, Ring(0))  # type: ignore
            return T if eval_ else F
        # Switch from Expressions to Polynomials
        lhs, _ = lhs.quo_rem(lhs.content())
        # lhs = lhs / lhs.content()
        factor = lhs.factor()
        factor_list = list(factor)
        func: type[RcfAtomicFormula]
        match f.func:
            case rcf.Eq | rcf.Ne:
                func = f.func
                # Compute squarefree part
                lhs = Ring(1)
                for factor, _ in factor_list:
                    lhs *= factor
                if lhs.lc() < 0:
                    lhs = - lhs
            case rcf.Le | rcf.Ge | rcf.Lt | rcf.Gt:
                unit = factor.unit()
                lhs = Ring(1)
                for factor, multiplicity in factor_list:
                    lhs *= factor ** 2 if multiplicity % 2 == 0 else factor
                if lhs.lc() < 0:
                    lhs = - lhs
                    unit = - unit
                func = f.converse_func if unit < 0 else f.func
            case _:
                assert False
        return func(lhs, Ring(0), chk=False)

    def sort_atoms(self, atoms: list[AtomicFormula]) -> None:
        atoms.sort(key=Simplify._sort_key_at)

    def sort_others(self, others: list[Formula]) -> None:
        others.sort(key=Simplify._sort_key)

    @staticmethod
    def _sort_key(f: Formula) -> SortKey:
        assert isinstance(f, (And, Or))
        atom_sort_keys = tuple(Simplify._sort_key_at(a) for a in f.atoms())
        return (f.depth(), len(f.args), len(atom_sort_keys), atom_sort_keys)

    @staticmethod
    def _sort_key_at(f: AtomicFormula) -> AtomicSortKey:
        match f:
            case Eq():
                return (0, f.lhs)
            case Ge():
                return (1, f.lhs)
            case Le():
                return (2, f.lhs)
            case Gt():
                return (3, f.lhs)
            case Lt():
                return (4, f.lhs)
            case Ne():
                return (5, f.lhs)
            case _:
                assert False

    def _Theory(self) -> Theory:
        return Theory(prefer_weak=self.prefer_weak,
                      prefer_order=self.prefer_order, _develop=self._develop)


simplify = Simplify()
