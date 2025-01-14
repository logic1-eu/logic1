"""This module provides an implementation of *deep simplifcication* based on
generating and propagating internal representations during recursion in Real
Closed fields. This is essentially the *standard simplifier*, which has been
proposed for Ordered Fields in [DolzmannSturm-1997]_.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from math import prod
from operator import xor
from typing import Collection, Final, Iterable, Iterator, Optional, Self

from gmpy2 import mpfr, mpq, sign

from ... import abc
from ...firstorder import And, _F, Not, Or, _T
from .atomic import AtomicFormula, CACHE_SIZE, DEFINITE, Eq, Ge, Le, Gt, Lt, Ne, Term, Variable
from .substitution import _SubstValue, _Substitution  # type: ignore
from .typing import Formula

from ...support.tracing import trace  # noqa


oo: Final[mpfr] = mpfr('inf')


@dataclass(frozen=True)
class Options(abc.simplify.Options):
    explode_always: bool = True
    lift: bool = True
    prefer_order: bool = True
    prefer_weak: bool = False


@dataclass
class _Range:
    r"""Non-empty range IVL \ EXC, where IVL is an interval with boundaries in
    Q extended by {-oo, oo}, and EXC is a finite subset of the interior of
    IVL. Raises InternalRepresentation.Inconsistent if IVL \ EXC gets empty.
    """

    lopen: bool = True
    start: mpq | mpfr = -oo
    end: mpq | mpfr = oo
    ropen: bool = True
    exc: set[mpq] = field(default_factory=set)

    def __contains__(self, q: mpq) -> bool:
        if q < self.start or (self.lopen and q == self.start):
            return False
        if self.end < q or (self.ropen and q == self.end):
            return False
        if q in self.exc:
            return False
        return True

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise InternalRepresentation.Inconsistent()
        if (self.lopen or self.ropen) and self.start == self.end:
            raise InternalRepresentation.Inconsistent()
        assert self.lopen or self.start is not -oo, self
        assert self.ropen or self.end is not oo, self
        assert all(self.start < x < self.end for x in self.exc), self

    def __str__(self) -> str:
        left = '(' if self.lopen else '['
        start = '-oo' if self.start == -oo else str(self.start)
        end = 'oo' if self.end == oo else str(self.end)
        right = ')' if self.ropen else ']'
        exc_entries = {str(q) for q in self.exc}
        if exc_entries:
            exc = f' \\ {{{", ".join(exc_entries)}}}'
        else:
            exc = ''
        return f'{left}{start}, {end}{right}{exc}'

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
        # Fix the case that ivl is closed on either side and the corresponding
        # endpoint is in self.exc or other.exc
        if start in self.exc or start in other.exc:
            lopen = True
        if end in self.exc or end in other.exc:
            ropen = True
        exc = set()
        for x in self.exc:
            if start < x < end:
                exc.add(x)
        for x in other.exc:
            if start < x < end:
                exc.add(x)
        return self.__class__(lopen, start, end, ropen, exc)

    def is_point(self) -> bool:
        # It is assumed and has been asserted that the interval is not empty.
        return self.start == self.end

    def is_zero(self) -> bool:
        return self.is_point() and self.start == 0

    def transform(self, scale: mpq, shift: mpq) -> Self:
        if scale >= 0:
            lopen = self.lopen
            start = scale * self.start + shift
            end = scale * self.end + shift
            ropen = self.ropen
        else:
            lopen = self.ropen
            start = scale * self.end + shift
            end = scale * self.start + shift
            ropen = self.lopen
        exc = {scale * point + shift for point in self.exc}
        return self.__class__(lopen, start, end, ropen, exc)


@dataclass
class _BasicKnowledge:

    term: Term
    range: _Range

    def __post_init__(self) -> None:
        assert self.term.lc() == 1, self
        assert self.term.constant_coefficient() == 0

    def as_atoms(self, ref_range: _Range, gand: type[And | Or], options: Options) \
            -> list[AtomicFormula]:
        L: list[AtomicFormula] = []
        # range_ cannot be empty because the construction of an empty range
        # raises an exception during `add`.
        if self.range.is_point():
            # Pick the one point of self.range.
            q = self.range.start
            assert isinstance(q, mpq)
            if ref_range.is_point():
                assert q == ref_range.start
                # throw away the point q, which is equal to the point
                # describe by ref_range
            else:
                assert q in ref_range
                # When gand is And, the equation q = 0 is generally
                # preferable. Otherwise, q = 0 would become q != 0
                # via subsequent negation, and we want to take
                # options.prefer_order into consideration.
                if options.prefer_order and gand is Or:
                    if q == ref_range.start:
                        assert not ref_range.lopen
                        L.append(Le(self.term - q, 0))
                    elif q == ref_range.end:
                        assert not ref_range.ropen
                        L.append(Ge(self.term - q, 0))
                    else:
                        L.append(Eq(self.term - q, 0))
                else:
                    L.append(Eq(self.term - q, 0))
        else:
            # print(f'{t=}, {ref_ivl=}, {ivl=}')
            #
            # We know that ref_range is a proper interval, too, because
            # self.range is a subset of ref_range.
            assert not ref_range.is_point()
            if ref_range.start < self.range.start:
                assert isinstance(self.range.start, mpq)
                if self.range.start in ref_range.exc:
                    # When gand is Or, weak and strong are dualized via
                    # subsequent negation.
                    if xor(options.prefer_weak, gand is Or):
                        L.append(Ge(self.term - self.range.start, 0))
                    else:
                        L.append(Gt(self.term - self.range.start, 0))
                else:
                    if self.range.lopen:
                        L.append(Gt(self.term - self.range.start, 0))
                    else:
                        L.append(Ge(self.term - self.range.start, 0))
            elif ref_range.start == self.range.start:
                if not ref_range.lopen and self.range.lopen:
                    assert isinstance(self.range.start, mpq)
                    # When gand is Or, Ne will become Eq via subsequent
                    # nagation. This is generally preferable.
                    if options.prefer_order and gand is And:
                        L.append(Gt(self.term - self.range.start, 0))
                    else:
                        L.append(Ne(self.term - self.range.start, 0))
            else:
                assert False
            if self.range.end < ref_range.end:
                assert isinstance(self.range.end, mpq)
                if self.range.end in ref_range.exc:
                    # When gand is Or, weak and strong are dualized via
                    # subsequent negation.
                    if xor(options.prefer_weak, gand is Or):
                        L.append(Le(self.term - self.range.end, 0))
                    else:
                        L.append(Lt(self.term - self.range.end, 0))
                else:
                    if self.range.ropen:
                        L.append(Lt(self.term - self.range.end, 0))
                    else:
                        L.append(Le(self.term - self.range.end, 0))
            elif ref_range.end == self.range.end:
                if not ref_range.ropen and self.range.ropen:
                    assert isinstance(self.range.end, mpq)
                    # When gand is Or, Ne will become Eq via subsequent
                    # nagation. This is generally preferable.
                    if options.prefer_order and gand is And:
                        L.append(Lt(self.term - self.range.end, 0))
                    else:
                        L.append(Ne(self.term - self.range.end, 0))
            else:
                assert False
        for q in self.range.exc:
            if q not in ref_range.exc:
                L.append(Ne(self.term - q, 0))
        return L

    def as_subst_values(self) -> tuple[_SubstValue, _SubstValue]:
        assert self.is_substitution()
        mons = self.term.monomials()
        if len(mons) == 1:
            assert isinstance(self.range.start, mpq)
            return (_SubstValue(mpq(1), self.term.as_variable()),
                    _SubstValue(self.range.start, None))
        else:
            x1 = mons[0].as_variable()
            x2 = mons[1].as_variable()
            c1 = self.term.monomial_coefficient(x1)
            c2 = self.term.monomial_coefficient(x2)
            return (_SubstValue(c1, x1), _SubstValue(-c2, x2))

    def is_substitution(self) -> bool:
        if not self.range.is_point():
            return False
        mons = self.term.monomials()
        if len(mons) == 1:
            return mons[0].is_variable()
        if len(mons) == 2:
            return mons[0].is_variable() and mons[1].is_variable() and self.range.start == 0
        return False

    def reduce(self, G: Iterable[Term]) -> Optional[Self]:
        t = self.term.reduce(G)
        if t.is_constant():
            if t.constant_coefficient() in self.range:
                return None
            raise InternalRepresentation.Inconsistent()
        lc = t.lc()
        t /= lc
        q = t.constant_coefficient()
        t -= q
        range_ = self.range.transform(1 / lc, -q)
        return self.__class__(t, range_)

    @classmethod
    def from_atom(cls, atom: AtomicFormula) -> Self:
        r"""Convert AtomicFormula into _BasicKnowledge.

        We assume that :data:`f` has gone through :meth:`_simpl_at` so that
        its left hand side is monic and its right hand side is zero.

        >>> from .atomic import VV
        >>> a, b = VV.get('a', 'b')
        >>> f = a**2 + mpq(1,2)*a*b - 6*b**2 + mpq(1,3) <= 0
        >>> _BasicKnowledge.from_atom(f)
        _BasicKnowledge(term=a^2 + 1/2*a*b - 6*b^2, range=_Range(lopen=True,
            start=mpfr('-inf'), end=mpq(-1,3), ropen=False, exc=set()))
        """
        rel = atom.op
        lhs = atom.lhs
        q = -lhs.constant_coefficient()
        term = lhs + q
        # rel is the relation of atom, term is the monic parametric part, and q
        # is the negative constant coefficient.
        if rel is Eq:
            range_ = _Range(False, q, q, False, set())
        elif rel is Ne:
            range_ = _Range(True, -oo, oo, True, {q})
        elif rel is Ge:
            range_ = _Range(False, q, oo, True, set())
        elif rel is Le:
            range_ = _Range(True, -oo, q, False, set())
        elif rel is Gt:
            range_ = _Range(True, q, oo, True, set())
        elif rel is Lt:
            range_ = _Range(True, -oo, q, True, set())
        else:
            assert False, rel
        bknowl = cls(term, range_)
        # This classmethod is essentially a constructor. We eventually call
        # __post_init__() instead of adding assertion earlier.
        bknowl.__post_init__()
        return bknowl


@dataclass
class _Knowledge:

    dict_: dict[Term, _Range] = field(default_factory=dict)

    def __contains__(self, t: Term) -> bool:
        return t in self.dict_

    def __getitem__(self, t: Term) -> _Range:
        return self.dict_[t]

    def __iter__(self) -> Iterator[_BasicKnowledge]:
        for t, range_ in self.dict_.items():
            yield _BasicKnowledge(t, range_)

    def __str__(self) -> str:
        entries = [str(key) + ' in ' + str(range) for key, range in self.dict_.items()]
        return f'{{{", ".join(entries)}}}'

    def add(self, bknowl: _BasicKnowledge) -> None:
        bknowl = self.prune(bknowl)
        self.dict_[bknowl.term] = bknowl.range

    def copy(self) -> Self:
        return self.__class__(self.dict_.copy())

    def get(self, key: Term, default: _Range = _Range(True, -oo, oo, True, set())) -> _Range:
        return self.dict_.get(key, default)

    def is_known(self, bknowl: _BasicKnowledge) -> bool:
        bknowl = self.prune(bknowl)
        return self.get(bknowl.term) != bknowl.range

    def prune(self, bknowl: _BasicKnowledge) -> _BasicKnowledge:
        try:
            range_ = self.dict_[bknowl.term]
        except KeyError:
            return bknowl
        range_ = range_.intersection(bknowl.range)
        return bknowl.__class__(bknowl.term, range_)

    def reduce(self, G: Collection[Term]) -> Self:
        knowl = self.__class__()
        for bknowl in self:
            maybe_bknowl = bknowl.reduce(G)
            if maybe_bknowl is None:
                continue
            knowl.add(maybe_bknowl)
        return knowl


@dataclass
class InternalRepresentation(
        abc.simplify.InternalRepresentation[AtomicFormula, Term, Variable, int]):
    """Implements the abstract methods :meth:`add() <.abc.simplify.InternalRepresentation.add>`,
    :meth:`extract() <.abc.simplify.InternalRepresentation.extract>`, and :meth:`next_()
    <.abc.simplify.InternalRepresentation.next_>` of it super class
    :class:`.abc.simplify.InternalRepresentation`. Required by
    :class:`.Sets.simplify.Simplify` for instantiating the type variable
    :data:`.abc.simplify.ρ` of :class:`.abc.simplify.Simplify`.
    """
    _options: Options
    _knowl: _Knowledge = field(default_factory=_Knowledge)
    _subst: _Substitution = field(default_factory=_Substitution)

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> abc.simplify.RESTART:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.add`.
        """
        if gand is Or:
            atoms = (atom.to_complement() for atom in atoms)
        restart = abc.simplify.RESTART.NONE
        for atom in atoms:
            # print(f'{atom=}')
            # print(f'{self=}')
            # assert all(v not in self._subst.as_dict() for v in atom.fvars())
            assert not atom.lhs.is_constant()
            assert atom.lhs.lc() == 1, atom
            maybe_bknowl = _BasicKnowledge.from_atom(atom).reduce(self._subst.as_gb())
            if maybe_bknowl is None:
                continue
            bknowl = self._knowl.prune(maybe_bknowl)
            if self._knowl.is_known(maybe_bknowl):
                if bknowl.is_substitution():
                    self._propagate(bknowl)
                    restart = abc.simplify.RESTART.ALL
                else:
                    self._knowl.add(bknowl)
                    if restart is not abc.simplify.RESTART.ALL:
                        restart = abc.simplify.RESTART.OTHERS
            # print(f'{self=}')
            # print()
        return restart

    def _propagate(self, bknowl: _BasicKnowledge) -> None:
        # print(f'_propagate: {self=}, {bknowl=}')
        assert bknowl.is_substitution()
        stack = [bknowl]
        while stack:
            for bknowl in stack:
                val1, val2 = bknowl.as_subst_values()
                self._subst.union(val1, val2)
            stack = []
            self._knowl = self._knowl.reduce(self._subst.as_gb())
            for bknowl in self._knowl:
                if bknowl.is_substitution():
                    stack.append(bknowl)

    def extract(self, gand: type[And | Or], ref: Self) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.extract`.
        """

        knowl = ref._knowl.reduce(self._subst.as_gb())
        # print(f'extract: {ref._knowl=}\n'
        #       f'         {self._knowl=}\n'
        #       f'         {self._subst=}\n'
        #       f'         {knowl=}')
        L: list[AtomicFormula] = []
        for bknowl in self._knowl:
            ref_range = knowl.get(bknowl.term)
            L.extend(bknowl.as_atoms(ref_range, gand, self._options))
        known_subst = ref._subst.copy()
        # print(f'{known_subst=}')
        items = sorted(self._subst, key=lambda item: Term.sort_key(item[0]))
        for var, val in items:
            if known_subst.is_redundant(_SubstValue(mpq(1), var), val):
                continue
            known_subst.union(_SubstValue(mpq(1), var), val)
            # print(f'{known_subst=}')
            G = self._subst.as_gb(ignore=var)
            knowl = ref._knowl.reduce(G)
            if val.variable is None:
                t: Term = var
                q = val.coefficient
            else:
                t = var - val.as_term()
                q = mpq(0)
            if self._options.prefer_order and gand is Or and t in knowl:
                ref_range = knowl[t]
                if q == ref_range.start:
                    assert not ref_range.lopen
                    L.append(Le(t - q, 0))
                elif q == ref_range.end:
                    assert not ref_range.ropen
                    L.append(Ge(t - q, 0))
                else:
                    L.append(Eq(t - q, 0))
            else:
                L.append(Eq(t - q, 0))
        if gand is Or:
            L = [atom.to_complement() for atom in L]
        return L

    def next_(self, remove: Optional[Variable] = None) -> Self:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.next_`.
        """
        if remove is None:
            knowl = self._knowl.copy()
            subst = self._subst.copy()
        else:
            knowl = _Knowledge()
            for bknowl in self._knowl:
                if remove not in bknowl.term.vars():
                    knowl.add(bknowl)
            subst = _Substitution()
            for var, val in self._subst:
                if var != remove and (val.variable is None or val.variable != remove):
                    subst.union(_SubstValue(mpq(1), var), val)
        return self.__class__(self._options, knowl, subst)

    def restart(self, ir: Self) -> Self:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.restart`.
        """
        result = self.next_()
        for var, val in ir._subst:
            if val.variable is None:
                t: Term = var
                q = val.coefficient
            else:
                t = var - val.coefficient * val.variable
                q = mpq(0)
            bknowl = _BasicKnowledge(t, _Range(False, q, q, False, set()))
            result._propagate(bknowl)
        return result

    def transform_atom(self, atom: AtomicFormula) -> AtomicFormula:
        """Apply the substitution part of `self` to atom. The result is an atom
        with monic left hand side and zero right hand side.
        """
        G = self._subst.as_gb()
        assert atom.rhs == 0
        lhs = atom.lhs.reduce(G)
        if lhs.is_constant():
            return Eq(0, 0) if atom.op(lhs, 0) else Eq(1, 0)
        # At the time of writing this procedure _simpl_at is applied to the
        # result immediately. Nevertheless, we keep the left hand side monic
        # also here.
        if lhs.lc() < 0:
            op = atom.op.converse()
            lhs = -lhs
        else:
            op = atom.op
        return op(lhs / lhs.lc(), 0)


@dataclass(frozen=True)
class Simplify(abc.simplify.Simplify[
        AtomicFormula, Term, Variable, int, InternalRepresentation, Options]):
    """Deep simplification following [DolzmannSturm-1997]_. Implements the
    abstract methods :meth:`create_initial_representation
    <.abc.simplify.Simplify.create_initial_representation>` and :meth:`simpl_at
    <.abc.simplify.Simplify.simpl_at>` of its super class
    :class:`.abc.simplify.Simplify`.

    The simplifier should be called via :func:`.simplify`, as described below.
    In addition, this class inherits :meth:`.abc.simplify.Simplify.is_valid`,
    which should be called via :func:`.is_valid`, as described below.
    """

    _options: Options = field(default_factory=Options)

    def create_initial_representation(self, assume: Iterable[AtomicFormula]) \
            -> InternalRepresentation:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.create_initial_representation`.
        """
        ir = InternalRepresentation(_options=self._options)
        for atom in assume:
            simplified_atom = self._simpl_at(atom, And, explode_always=False)
            if Formula.is_atomic(simplified_atom):
                ir.add(And, [simplified_atom])
            elif Formula.is_and(simplified_atom):
                args = simplified_atom.args
                assert all(isinstance(arg, AtomicFormula) for arg in args)
                ir.add(And, args)
            elif Formula.is_true(simplified_atom):
                continue
            elif Formula.is_false(simplified_atom):
                raise InternalRepresentation.Inconsistent()
            else:
                assert False, simplified_atom
        return ir

    def _post_process(self, f: Formula) -> Formula:
        """
        >>> from logic1.theories.RCF import *
        >>> x, y = VV.get('x', 'y')
        >>> simplify(8*x + 12*y == 4)
        2*x + 3*y - 1 == 0
        >>> simplify(8*x + 12*y == 4, lift=False)
        x + 3/2*y - 1/2 == 0
        """
        @lru_cache(maxsize=CACHE_SIZE)
        def lift_atom(atom: AtomicFormula) -> AtomicFormula:
            return atom.op(atom.lhs / atom.lhs.content(), 0)

        if self._options.lift is False:
            return f
        return f.traverse(map_atoms=lift_atom, sort_levels=True)

    def simpl_at(self, atom: AtomicFormula, context: Optional[type[And] | type[Or]]) -> Formula:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.simpl_at`.
        """
        # MyPy does not recognize that And[Any, Any, Any] is an instance of
        # Hashable. https://github.com/python/mypy/issues/11470
        return self._simpl_at(
            atom, context, self._options.explode_always)  # type: ignore[arg-type]

    @lru_cache(maxsize=CACHE_SIZE)
    def _simpl_at(self,
                  atom: AtomicFormula,
                  context: Optional[type[And] | type[Or]],
                  explode_always: bool) -> Formula:
        """Simplify atomic formula.

        >>> from .atomic import VV
        >>> a, b = VV.get('a', 'b')
        >>> simplify(-6 * (a+b)**2 + 3 <= 0)
        2*a^2 + 4*a*b + 2*b^2 - 1 >= 0
        """
        def _simpl_at_eq_ne(rel: type[Eq | Ne], lhs: Term) -> Formula:

            def split_definite(term):
                args = []
                for _, power_product in term:
                    if explode_always:
                        args.append(fac_junctor(*(rel(v, 0) for v in power_product.vars())))
                    else:
                        args.append(rel(prod(power_product.vars()), 0))
                return definite_junctor(*args)

            if rel is Eq:
                definite_junctor: type[And | Or] = And
                fac_junctor: type[And | Or] = Or
            else:
                assert rel is Ne
                definite_junctor = Or
                fac_junctor = And
            definite = lhs.is_definite()
            # Definiteness tests on original left hand side
            if definite in (DEFINITE.NEGATIVE, DEFINITE.POSITIVE):
                return definite_junctor.definite_element()
            _, factors = lhs.factor()
            square_free_lhs = Term(1)
            for factor in factors:
                square_free_lhs *= factor
            # Definiteness tests on square-free part:
            square_free_definite = square_free_lhs.is_definite()
            if square_free_definite in (DEFINITE.NEGATIVE, DEFINITE.POSITIVE):
                return definite_junctor.definite_element()
            if explode_always or context == definite_junctor:
                if square_free_definite in (DEFINITE.NEGATIVE_SEMI, DEFINITE.POSITIVE_SEMI):
                    return split_definite(square_free_lhs)
                if definite in (DEFINITE.NEGATIVE_SEMI, DEFINITE.POSITIVE_SEMI):
                    return split_definite(lhs)
            if explode_always or context == fac_junctor:
                args = (rel(factor, 0) for factor in factors)
                return fac_junctor(*args)
            return rel(square_free_lhs, 0)

        def _simpl_at_ge(lhs: Term) -> Formula:

            def definiteness_test(f: Term) -> Optional[Formula]:
                definite = f.is_definite()
                if definite in (DEFINITE.POSITIVE, DEFINITE.POSITIVE_SEMI):
                    return _T()
                if definite is DEFINITE.NEGATIVE:
                    return _F()
                if definite is DEFINITE.NEGATIVE_SEMI:
                    return _simpl_at_eq_ne(Eq, f)
                if definite is DEFINITE.NONE:
                    return None
                assert False

            # Definiteness tests on original left hand side:
            hit = definiteness_test(lhs)
            if hit is not None:
                return hit
            # Factorize
            unit, factors = lhs.factor()
            sgn = sign(unit)
            even_factors = []
            even_factor = Term(1)
            odd_factor = Term(1)
            for factor, multiplicity in factors.items():
                if factor.is_definite() is DEFINITE.POSITIVE:
                    continue
                if multiplicity % 2 == 0:
                    even_factors.append(factor)
                    even_factor *= factor
                else:
                    odd_factor *= factor
            remaining_squarefree_part = odd_factor * even_factor
            # Definiteness tests on factorization
            if (sgn * odd_factor).is_definite() in (DEFINITE.POSITIVE, DEFINITE.POSITIVE_SEMI):
                return _T()
            if (sgn * odd_factor).is_definite() is DEFINITE.NEGATIVE:
                return _simpl_at_eq_ne(Eq, even_factor)
            if (sgn * odd_factor).is_definite() is DEFINITE.NEGATIVE_SEMI:
                return _simpl_at_eq_ne(Eq, sgn * remaining_squarefree_part)
            hit = definiteness_test(sgn * remaining_squarefree_part)
            # Definiteness tests on signed remaining squarefree part:
            if hit is not None:
                return hit
            # All definiteness tests have failed.
            rel = Ge if sgn == 1 else Le
            if explode_always or context is Or:
                odd_part = rel(odd_factor, 0)
                even_part = (Eq(f, 0) for f in even_factors)
                return Or(odd_part, *even_part)
            return rel(remaining_squarefree_part * even_factor, 0)

        lhs = atom.lhs - atom.rhs
        if lhs.is_constant():
            # In the following if-condition, the __bool__ method of atom.op
            # will be called.
            return _T() if atom.op(lhs, 0) else _F()
        if isinstance(atom, Eq):
            return _simpl_at_eq_ne(Eq, lhs)
        if isinstance(atom, Ne):
            return _simpl_at_eq_ne(Ne, lhs)
        if isinstance(atom, Ge):
            return _simpl_at_ge(lhs)
        if isinstance(atom, Le):
            return _simpl_at_ge(-lhs)
        if isinstance(atom, Gt):
            if context is not None:
                context = context.dual()
            return Not(_simpl_at_ge(-lhs)).to_nnf()
        if isinstance(atom, Lt):
            if context is not None:
                context = context.dual()
            return Not(_simpl_at_ge(lhs)).to_nnf()
        assert False, atom


def simplify(f: Formula, assume: Iterable[AtomicFormula] = [], **options) -> Formula:
    r"""Simplify `f` modulo `assume`.

    :param f:
      The formula to be simplified

    :param assume:
      A list of atomic formulas that are assumed to hold. The
      simplification result is equivalent modulo those assumptions. Note
      that assumptions do not affect bound variables.

      >>> from logic1.firstorder import *
      >>> from logic1.theories.RCF import *
      >>> a, b = VV.get('a', 'b')
      >>> simplify(Ex(a, And(a > 5, b > 10)), assume=[a > 10, b > 20])
      Ex(a, a - 5 > 0)

    :param explode_always:
      Simplification can split certain atomic formulas built from products
      or square sums:

      .. admonition:: Example

        1.
          1. :math:`ab = 0` is equivalent to :math:`a = 0 \lor b = 0`
          2. :math:`a^2 + b^2 \neq 0` is equivalent to :math:`a \neq 0 \lor b \neq 0`;

        2.
          1. :math:`ab \neq 0` is equivalent to :math:`a \neq 0 \land b \neq 0`
          2. :math:`a^2 + b^2 = 0` is equivalent to :math:`a = 0 \land b = 0`.

      If `explode_always` is :data:`False`, the splittings in "1." are only applied
      within disjunctions and the ones in "2." are only applied within conjunctions.
      This keeps terms more complex but the boolean structure simpler.

      >>> from logic1.firstorder import *
      >>> from logic1.theories.RCF import *
      >>> a, b, c = VV.get('a', 'b', 'c')
      >>> simplify(And(a * b == 0, c == 0))
      And(c == 0, Or(b == 0, a == 0))
      >>> simplify(And(a * b == 0, c == 0), explode_always=False)
      And(c == 0, a*b == 0)
      >>> simplify(Or(a * b == 0, c == 0), explode_always=False)
      Or(c == 0, b == 0, a == 0)

    :param prefer_order:
      One can sometimes equivalently choose between order inequalities and
      (in)equations.

      .. admonition:: Example

        1. :math:`a > 0 \lor (b = 0 \land a < 0)` is equivalent to
           :math:`a > 0 \lor (b = 0 \land a \neq 0)`
        2. :math:`a \geq 0 \land (b = 0 \lor a > 0)` is equivalent to
           :math:`a \geq 0 \land (b = 0 \lor a \neq 0)`

      By default, the left hand sides in the Example are preferred. If
      `prefer_order` is :data:`False`, then the right hand sides are preferred.

      >>> from logic1.firstorder import *
      >>> from logic1.theories.RCF import *
      >>> a, b = VV.get('a', 'b')
      >>> simplify(And(a >= 0, Or(b == 0, a > 0)))
      And(a >= 0, Or(b == 0, a > 0))
      >>> simplify(And(a >= 0, Or(b == 0, a != 0)))
      And(a >= 0, Or(b == 0, a > 0))
      >>> simplify(And(a >= 0, Or(b == 0, a > 0)), prefer_order=False)
      And(a >= 0, Or(b == 0, a != 0))
      >>> simplify(And(a >= 0, Or(b == 0, a != 0)), prefer_order=False)
      And(a >= 0, Or(b == 0, a != 0))

    :param prefer_weak:
      One can sometimes equivalently choose between strict and weak inequalities.

      .. admonition:: Example

        1. :math:`a = 0 \lor (b = 0 \land a \geq 0)` is equivalent to
           :math:`a = 0 \lor (b = 0 \land a > 0)`
        2. :math:`a \neq 0 \land (b = 0 \lor a \geq 0)` is equivalent to
           :math:`a \neq 0 \land (b = 0 \lor a > 0)`

      By default, the right hand sides in the Example are preferred. If
      `prefer_weak` is :data:`True`, then the left hand sides are
      preferred.

      >>> from logic1.firstorder import *
      >>> from logic1.theories.RCF import *
      >>> a, b = VV.get('a', 'b')
      >>> simplify(And(a != 0, Or(b == 0, a >= 0)))
      And(a != 0, Or(b == 0, a > 0))
      >>> simplify(And(a != 0, Or(b == 0, a > 0)))
      And(a != 0, Or(b == 0, a > 0))
      >>> simplify(And(a != 0, Or(b == 0, a >= 0)), prefer_weak=True)
      And(a != 0, Or(b == 0, a >= 0))
      >>> simplify(And(a != 0, Or(b == 0, a > 0)), prefer_weak=True)
      And(a != 0, Or(b == 0, a >= 0))

    :returns:
      A simplified equivalent of `f` modulo `assume`.
    """
    return Simplify(Options(**options)).simplify(f, assume)


def is_valid(f: Formula, assume: Iterable[AtomicFormula] = [], **options) -> Optional[bool]:
    return Simplify(Options(**options)).is_valid(f, assume)
