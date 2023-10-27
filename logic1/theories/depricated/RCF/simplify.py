from functools import lru_cache
from operator import xor
from sympy import FiniteSet, Interval, oo, Rational, S, sqf_list, sqf_part
from typing import Iterable, Optional, Self

from . import rcf  # need qualified names of relations for pattern matching
from .... import abc

from ....firstorder.formula import Formula
from ....firstorder.boolean import And, Or
from ....firstorder.atomic import AtomicFormula
from ....firstorder.truth import T, F
from .rcf import RcfAtomicFormula, RcfAtomicFormulas, Term, Variable, Eq, Ne, Ge, Le, Gt, Lt


class Theory(abc.simplify.Theory):

    _reference: dict[Term, tuple[Interval | FiniteSet, FiniteSet]]
    _current: dict[Term, tuple[Interval | FiniteSet, FiniteSet]]

    def __repr__(self):
        return f'Theory({self._reference}, {self._current})'

    def add(self, gand: type[And] | type[Or], atoms: Iterable[AtomicFormula]) -> None:
        for atom in atoms:
            # rel is the relation of atom, p is the parametric part, and q is
            # the negative of the Rational absolute summand.
            rel, p, q = self._decompose_atom(atom)
            if gand is Or:
                rel = rel.complement_func
            ivl: Interval | FiniteSet
            exc: FiniteSet
            match rel:
                # We model p in ivl \ exc.
                #
                # Compare https://stackoverflow.com/q/71441761/ which suggests
                # the use of __qualname__ here.
                case rcf.Eq:
                    ivl = FiniteSet(q)
                    exc = S.EmptySet
                case rcf.Ne:
                    ivl = S.Reals
                    exc = FiniteSet(q)
                case rcf.Ge:
                    ivl = Interval(q, oo)
                    exc = S.EmptySet
                case rcf.Le:
                    ivl = Interval(-oo, q)
                    exc = S.EmptySet
                case rcf.Gt:
                    ivl = Interval.Lopen(q, oo)
                    exc = S.EmptySet
                case rcf.Lt:
                    ivl = Interval.Ropen(-oo, q)
                    exc = S.EmptySet
                case _:
                    assert False
            if p in self._current:
                cur_ivl, cur_exc = self._current[p]
                ivl = ivl.intersection(cur_ivl)
                if ivl is S.EmptySet:
                    raise self.Inconsistent
                exc = exc.union(cur_exc)
                # Restrict exc to the new ivl.
                exc = exc.intersection(ivl)
                # Note that exc is a subset of ivl now. Fix the case that ivl
                # is closed on either side and the corresponding endpoint is in
                # exc. We are going to use inf and sup in contrast to start and
                # end, because ivl can be a FiniteSet.
                if ivl.inf in exc:
                    # It follows that ivl is left-closed.
                    ivl = ivl - FiniteSet(ivl.inf)
                    if ivl is S.EmptySet:
                        raise self.Inconsistent
                    exc = exc - FiniteSet(ivl.inf)
                if ivl.sup in exc:
                    # It follows that ivl is right-closed.
                    ivl = ivl - FiniteSet(ivl.sup)
                    exc = exc - FiniteSet(ivl.sup)
                assert ivl is not S.EmptySet
            self._current[p] = (ivl, exc)

    def __init__(self, prefer_weak: bool, prefer_order: bool, _develop: int) -> None:
        self.prefer_weak = prefer_weak
        self.prefer_order = prefer_order
        self._develop = _develop
        self._reference = dict()
        self._current = dict()

    @staticmethod
    def _compose_atom(rel: type[AtomicFormula], p: Term, q: Rational)\
            -> AtomicFormula:
        return rel(q.denominator * p - q.numerator, 0)

    @staticmethod
    def _decompose_atom(f: AtomicFormula)\
            -> tuple[type[AtomicFormula], Term, Rational]:
        """Decompose into relation :math:`\rho`, term :math:`p` without
        absolute summand, and rational :math:`q` such that :data:`f` is
        equivalent to :math:`p \rho q`.

        We assume that :data:`f` has gone through :meth:`_simpl_at` so that its
        right hand side is zero and its left hand side polynomial has gone
        through SymPy's :meth:`expand`.

        >>> from sympy.abc import a, b
        >>> f = Le(6*a**2 + 12*a*b + 6*b**2 + 3, 0)
        >>> rel, p, q = Theory._decompose_atom(f); rel, p, q
        (<class 'logic1.theories.depricated.RCF.rcf.Le'>, a**2 + 2*a*b + b**2, -1/2)
        >>> g = Theory._compose_atom(rel, p, q); g
        Le(2*a**2 + 4*a*b + 2*b**2 + 1, 0)
        >>> (f.lhs / g.lhs).simplify()
        3
        """
        c, t = f.args[0].as_coeff_Add()
        co, pp = t.as_content_primitive()
        return f.func, pp, -Rational(c, co)

    def extract(self, gand: type[And] | type[Or]) -> Iterable[AtomicFormula]:
        L: list[AtomicFormula] = []
        for p in self._current:
            if p in self._reference:
                ref_ivl, ref_exc = self._reference[p]
            else:
                ref_ivl, ref_exc = S.Reals, S.EmptySet
            ivl, exc = self._current[p]
            # S.Reals is an instance of Interval. The construction of an empty
            # interval raises an Exception in `add` so that we will not
            # encounter EmptySet as an interval here.
            match ivl:
                case FiniteSet():
                    assert len(ivl) == 1
                    match ref_ivl:
                        case Interval():
                            # FiniteSet.inf gives access to the only element.
                            q = ivl.inf
                            # When gand is And, the equation q = 0 is generally
                            # preferable. Otherwise, q = 0 would become q != 0
                            # via subsequent negation, and we want to take
                            # self.prefer_order into consideration.
                            if self.prefer_order and gand is Or:
                                if q == ref_ivl.start:
                                    assert not ref_ivl.left_open
                                    L.append(self._compose_atom(Le, p, q))
                                elif q == ref_ivl.end:
                                    assert not ref_ivl.right_open
                                    L.append(self._compose_atom(Ge, p, q))
                                else:
                                    L.append(self._compose_atom(Eq, p, q))
                            else:
                                L.append(self._compose_atom(Eq, p, q))
                        case FiniteSet():
                            assert ref_ivl == ivl, f'{ref_ivl} != {ivl}'
                case Interval():
                    # We know that ref_ivl is an Interval because ivl is a
                    # subset of ref_ivl.
                    assert isinstance(ref_ivl, Interval)
                    if ref_ivl.start < ivl.start:
                        if ivl.start in ref_exc:
                            # When gand is Or, weak and strong are dualized via
                            # subsequent negation.
                            if xor(self.prefer_weak, gand is Or):
                                L.append(self._compose_atom(Ge, p, ivl.start))
                            else:
                                L.append(self._compose_atom(Gt, p, ivl.start))
                        else:
                            if ivl.left_open:
                                L.append(self._compose_atom(Gt, p, ivl.start))
                            else:
                                L.append(self._compose_atom(Ge, p, ivl.start))
                    elif ref_ivl.start == ivl.start:
                        if not ref_ivl.left_open and ivl.left_open:
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
                            if ivl.right_open:
                                L.append(self._compose_atom(Lt, p, ivl.end))
                            else:
                                L.append(self._compose_atom(Le, p, ivl.end))
                    elif ref_ivl.end == ivl.end:
                        if not ref_ivl.right_open and ivl.right_open:
                            # When gand is Or, Ne will become Eq via subsequent
                            # nagation. This is generally preferable.
                            if self.prefer_order and gand is And:
                                L.append(self._compose_atom(Lt, p, ivl.end))
                            else:
                                L.append(self._compose_atom(Ne, p, ivl.end))
                case unkown:
                    assert False, f'expected Interval or FiniteSet, got {unkown}'
            for q in exc:
                if q not in ref_exc:
                    L.append(self._compose_atom(Ne, p, q))
        if gand is Or:
            L = [atom.complement_func(*atom.args) for atom in L]
        match self._develop:
            case 0:
                return sorted(L, key=self._sortkey)
            case 1:
                return L
            case _:
                assert False

    def next_(self, remove: Optional[Variable] = None) -> Self:
        theory_next = self.__class__(self.prefer_weak, self.prefer_order, self._develop)
        theory_next._reference = self._current
        if remove is None:
            # This is the regular case. I expect that copy is not slower than a
            # comprehension.
            theory_next._current = self._current.copy()
        else:
            # On the other hand, I expect a comprehension to be faster than
            # copy + del.
            theory_next._current = {p: q for p, q in self._current.items()
                                    if remove not in p.atoms(Variable)}
        return theory_next

    @staticmethod
    def _sortkey(atom: AtomicFormula):
        assert isinstance(atom, RcfAtomicFormulas)
        match atom:
            case Eq():
                return (0, atom.lhs.sort_key())
            case Ge():
                return (1, atom.lhs.sort_key())
            case Le():
                return (2, atom.lhs.sort_key())
            case Gt():
                return (3, atom.lhs.sort_key())
            case Lt():
                return (4, atom.lhs.sort_key())
            case Ne():
                return (5, atom.lhs.sort_key())
            case _:
                assert False


class Simplify(abc.simplify.Simplify['Theory']):

    def __call__(self, f: Formula, prefer_weak: bool = False,
                 prefer_order: bool = True, _develop: int = 0) -> Formula:
        self.prefer_weak = prefer_weak
        self.prefer_order = prefer_order
        self._develop = _develop
        return self.simplify(f)

    @lru_cache(maxsize=None)
    def _simpl_at(self, f: AtomicFormula, implicit_not: bool) -> Formula:
        """
        >>> from sympy.abc import a, b
        >>> simplify(Le(-6 * (a+b)**2 + 3, 0))
        Ge(2*a**2 + 4*a*b + 2*b**2 - 1, 0)
        """
        assert isinstance(f, RcfAtomicFormulas)
        lhs = f.lhs - f.rhs
        lhs = lhs.expand()
        if not lhs.free_symbols:
            return T if xor(f.sympy_func(lhs, S.Zero), implicit_not) else F
        func: type[RcfAtomicFormula]
        match f.func:
            case rcf.Eq | rcf.Ne:
                func = f.func
                lhs = sqf_part(lhs)
            case rcf.Le | rcf.Ge | rcf.Lt | rcf.Gt:
                # The multiplicities in sqf_list are Python ints.
                content, sqf_factors = sqf_list(lhs)
                func = f.converse_func if content < 0 else f.func
                lhs = S.One
                for factor, multiplicity in sqf_factors:
                    lhs *= factor ** 2 if multiplicity % 2 == 0 else factor
            case _:
                assert False
        if implicit_not:
            func = func.complement_func
        return func(lhs, S.Zero)

    def _Theory(self) -> Theory:
        return Theory(prefer_weak=self.prefer_weak,
                      prefer_order=self.prefer_order, _develop=self._develop)


simplify = Simplify()
