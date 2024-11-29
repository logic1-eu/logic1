"""This module provides an implementation of *deep simplifcication* based on
generating and propagating internal representations during recursion in Real
Closed fields. This is essentially the *standard simplifier*, which has been
proposed for Ordered Fields in [DolzmannSturm-1997]_.
"""

from dataclasses import dataclass, field
from functools import lru_cache
from operator import xor

from gmpy2 import mpq
from sage.all import oo, product
from sage.rings.infinity import MinusInfinity, PlusInfinity
from sage.rings.rational import Rational
from typing import Final, Iterable, Iterator, Optional, Self

from ... import abc
from ...firstorder import And, _F, Not, Or, _T
from .atomic import AtomicFormula, Eq, Ge, Le, Gt, Lt, Ne, Term, TSQ, Variable
from .typing import Formula

from ...support.tracing import trace  # noqa


CACHE_SIZE: Final[Optional[int]] = 2**16


@dataclass(frozen=True)
class Options(abc.simplify.Options):
    explode_always: bool = True
    prefer_order: bool = True
    prefer_weak: bool = False


@dataclass
class _Range:
    r"""Non-empty range IVL \ EXC, where IVL is an interval with boundaries in
    QQ, -oo, oo, and EXC is is created. A finite subset of the interior of
    IVL. Raises InternalRepresentation.Inconsistent if IVL \ EXC gets empty.
    """

    lopen: bool = True
    _start: Rational | MinusInfinity = -oo
    _end: Rational | PlusInfinity = oo
    ropen: bool = True
    _exc: set[Rational] = field(default_factory=set)

    @property
    def start(self) -> mpq | MinusInfinity:
        if isinstance(self._start, Rational):
            return mpq(self._start)
        return -oo

    @start.setter
    def start(self, arg: mpq | MinusInfinity) -> None:
        if isinstance(arg, mpq):
            self._start = Rational(arg)
        self._start = -oo

    @property
    def end(self) -> mpq | PlusInfinity:
        if isinstance(self._end, Rational):
            return mpq(self._end)
        return oo

    @end.setter
    def end(self, arg: mpq | PlusInfinity) -> None:
        if isinstance(arg, mpq):
            self._end = Rational(arg)
        self._end = oo

    @property
    def exc(self) -> set[mpq]:
        return {mpq(q) for q in self._exc}

    @exc.setter
    def exc(self, args: set[mpq]) -> None:
        self._exc = {Rational(x) for x in args}

    def __contains__(self, x: mpq) -> bool:
        q = Rational(x)
        if q < self._start or (self.lopen and q == self._start):
            return False
        if self._end < q or (self.ropen and q == self._end):
            return False
        if q in self._exc:
            return False
        return True

    def __init__(self, lopen: bool = True, start: mpq | Rational | MinusInfinity = -oo,
                 end: mpq | Rational | PlusInfinity = oo, ropen: bool = True,
                 exc: set[mpq] | set[Rational] = set()) -> None:
        self.lopen = lopen
        self._start = Rational(start) if isinstance(start, mpq) else start
        self._end = Rational(end) if isinstance(end, mpq) else end
        self.ropen = ropen
        self._exc = {Rational(x) if isinstance(x, mpq) else x for x in exc}
        if self._start > self._end:
            raise InternalRepresentation.Inconsistent()
        if (self.lopen or self.ropen) and self._start == self._end:
            raise InternalRepresentation.Inconsistent()
        assert self.lopen or self._start is not -oo, self
        assert self.ropen or self._end is not oo, self
        assert all(self._start < x < self._end for x in self._exc), self

    def __str__(self):
        left = '(' if self.lopen else '['
        right = ')' if self.ropen else ']'
        return f'{left}{self._start}, {self._end}{right} \\ {self._exc}'

    def intersection(self, other: Self) -> Self:
        if self._start < other._start:
            lopen = other.lopen
            _start = other._start
        elif other._start < self._start:
            lopen = self.lopen
            _start = self._start
        else:
            assert self._start == other._start
            _start = self._start
            lopen = self.lopen or other.lopen
        if self._end < other._end:
            _end = self._end
            ropen = self.ropen
        elif other._end < self._end:
            _end = other._end
            ropen = other.ropen
        else:
            assert self._end == other._end
            _end = self._end
            ropen = self.ropen or other.ropen
        # Fix the case that ivl is closed on either side and the corresponding
        # endpoint is in self.exc or other.exc
        if _start in self._exc or _start in other._exc:
            lopen = True
        if _end in self._exc or _end in other._exc:
            ropen = True
        _exc = set()
        for x in self._exc:
            if _start < x < _end:
                _exc.add(x)
        for x in other._exc:
            if _start < x < _end:
                _exc.add(x)
        return self.__class__(lopen, _start, _end, ropen, _exc)

    def is_point(self) -> bool:
        # It is assumed and has been asserted that the interval is not empty.
        return self._start == self._end

    def is_zero(self) -> bool:
        return self.is_point() and self._start == 0

    def transform(self, scale: mpq, shift: mpq) -> Self:
        _scale = Rational(scale)
        _shift = Rational(shift)
        if _scale >= 0:
            lopen = self.lopen
            _start: Rational | MinusInfinity = _scale * self._start + _shift
            _end: Rational | PlusInfinity = _scale * self._end + _shift
            ropen = self.ropen
        else:
            lopen = self.ropen
            _start = _scale * self._end + _shift
            _end = _scale * self._start + _shift
            ropen = self.lopen
        _exc = {_scale * point + _shift for point in self._exc}
        return self.__class__(lopen, _start, _end, ropen, _exc)


@dataclass(frozen=True)
class _SubstValue:
    coefficient: mpq
    variable: Optional[Variable]

    def __post_init__(self) -> None:
        assert self.coefficient != 0 or self.variable is None

    def as_term(self) -> Term:
        if self.variable is None:
            return Term(self.coefficient)
        else:
            return self.coefficient * self.variable


@dataclass(unsafe_hash=True)
class _Substitution:
    parents: dict[Variable, _SubstValue] = field(default_factory=dict)

    def __iter__(self) -> Iterator[tuple[Variable, _SubstValue]]:
        for var in self.parents:
            yield var, self.find(var)

    def as_dict(self) -> dict[Variable, Term]:
        """Convert this :class:._Substitution` into a dictionary that can used
        as an argument to subsitution methods.
        """
        return {var: val.as_term() for var, val in self}

    def copy(self) -> Self:
        return self.__class__(self.parents.copy())

    def find(self, v: Variable) -> _SubstValue:
        return self.internal_find(v)

    def internal_find(self, v: Optional[Variable]) -> _SubstValue:
        if v is None:
            return _SubstValue(mpq(1), None)
        try:
            parent = self.parents[v]
        except KeyError:
            return _SubstValue(mpq(1), v)
        root = self.internal_find(parent.variable)
        root = _SubstValue(parent.coefficient * root.coefficient, root.variable)
        self.parents[v] = root
        return root

    def union(self, val1: _SubstValue, val2: _SubstValue) -> None:
        root1 = self.internal_find(val1.variable)
        root2 = self.internal_find(val2.variable)
        c1 = val1.coefficient * root1.coefficient
        c2 = val2.coefficient * root2.coefficient
        if root1.variable is not None and root2.variable is not None:
            if root1.variable == root2.variable:
                if c1 != c2:
                    self.parents[root1.variable] = _SubstValue(mpq(0), None)
            elif Variable.sort_key(root1.variable) < Variable.sort_key(root2.variable):
                self.parents[root2.variable] = _SubstValue(c1 / c2, root1.variable)
            else:
                self.parents[root1.variable] = _SubstValue(c2 / c1, root2.variable)
        elif root1.variable is None and root2.variable is not None:
            self.parents[root2.variable] = _SubstValue(c1 / c2, None)
        elif root1.variable is not None and root2.variable is None:
            self.parents[root1.variable] = _SubstValue(c2 / c1, None)
        else:
            if c1 != c2:
                raise InternalRepresentation.Inconsistent()

    def is_redundant(self, val1: _SubstValue, val2: _SubstValue) -> Optional[bool]:
        """Check if the equation ``val1 == val2`` is redundant modulo self.
        """
        root1 = self.internal_find(val1.variable)
        root2 = self.internal_find(val2.variable)
        c1 = val1.coefficient * root1.coefficient
        c2 = val2.coefficient * root2.coefficient
        if root1.variable is None and root2.variable is None:
            return c1 == c2
        if root1.variable is None or root2.variable is None:
            return None
        if root1.variable == root2.variable and c1 == c2:
            return True
        else:
            return None

    def equations(self) -> Iterator[Eq]:
        raise NotImplementedError()


@dataclass
class _BasicKnowledge:

    term: Term
    range: _Range

    def __post_init__(self) -> None:
        assert self.term.lc() == 1, self
        assert self.term.constant_coefficient() == 0

    def as_atoms(self, ref_range: _Range, gand: type[And | Or], options: Options) \
            -> list[AtomicFormula]:
        L = []
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
                        L.append(self._compose_atom(Le, self.term, q))
                    elif q == ref_range.end:
                        assert not ref_range.ropen
                        L.append(self._compose_atom(Ge, self.term, q))
                    else:
                        L.append(self._compose_atom(Eq, self.term, q))
                else:
                    L.append(self._compose_atom(Eq, self.term, q))
        else:
            # print(f'{t=}, {ref_ivl=}, {ivl=}')
            #
            # We know that ref_range is a proper interval, too, because
            # self.range is a subset of ref_range.
            assert not ref_range.is_point()
            if ref_range.start < self.range.start:
                if self.range.start in ref_range.exc:
                    # When gand is Or, weak and strong are dualized via
                    # subsequent negation.
                    if xor(options.prefer_weak, gand is Or):
                        L.append(self._compose_atom(Ge, self.term, self.range.start))
                    else:
                        L.append(self._compose_atom(Gt, self.term, self.range.start))
                else:
                    if self.range.lopen:
                        L.append(self._compose_atom(Gt, self.term, self.range.start))
                    else:
                        L.append(self._compose_atom(Ge, self.term, self.range.start))
            elif ref_range.start == self.range.start:
                if not ref_range.lopen and self.range.lopen:
                    # When gand is Or, Ne will become Eq via subsequent
                    # nagation. This is generally preferable.
                    if options.prefer_order and gand is And:
                        L.append(self._compose_atom(Gt, self.term, self.range.start))
                    else:
                        L.append(self._compose_atom(Ne, self.term, self.range.start))
            else:
                assert False
            if self.range.end < ref_range.end:
                if self.range.end in ref_range.exc:
                    # When gand is Or, weak and strong are dualized via
                    # subsequent negation.
                    if xor(options.prefer_weak, gand is Or):
                        L.append(self._compose_atom(Le, self.term, self.range.end))
                    else:
                        L.append(self._compose_atom(Lt, self.term, self.range.end))
                else:
                    if self.range.ropen:
                        L.append(self._compose_atom(Lt, self.term, self.range.end))
                    else:
                        L.append(self._compose_atom(Le, self.term, self.range.end))
            elif ref_range.end == self.range.end:
                if not ref_range.ropen and self.range.ropen:
                    # When gand is Or, Ne will become Eq via subsequent
                    # nagation. This is generally preferable.
                    if options.prefer_order and gand is And:
                        L.append(self._compose_atom(Lt, self.term, self.range.end))
                    else:
                        L.append(self._compose_atom(Ne, self.term, self.range.end))
            else:
                assert False
        for q in self.range.exc:
            if q not in ref_range.exc:
                L.append(self._compose_atom(Ne, self.term, q))
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

    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def _compose_atom(rel: type[AtomicFormula], t: Term, q: mpq) -> AtomicFormula:
        lhs = t - q
        lhs /= lhs.content()
        return rel(lhs, 0)

    def is_substitution(self) -> bool:
        if not self.range.is_point():
            return False
        mons = self.term.monomials()
        if len(mons) == 1:
            return mons[0].is_variable()
        if len(mons) == 2:
            return mons[0].is_variable() and mons[1].is_variable() and self.range.start == 0
        return False

    def subs(self, sigma: dict[Variable, Term]) -> Optional[Self]:
        G = [key - val for key, val in sigma.items()]
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
    @lru_cache(maxsize=CACHE_SIZE)
    def from_atom(cls, f: AtomicFormula) -> Self:
        r"""Convert AtomicFormula into _BasicKnowledge.

        We assume that :data:`f` has gone through :meth:`_simpl_at` so that
        its right hand side is zero.

        >>> from .atomic import VV
        >>> a, b = VV.get('a', 'b')
        >>> f = 6*a**2 + 12*a*b + 6*b**2 + 3 <= 0
        >>> bknowl = _BasicKnowledge.from_atom(f); print(bknowl)
        _BasicKnowledge(term=a^2 + 2*a*b + b^2,
            range=_Range(lopen=True, _start=-Infinity, _end=-1/2, ropen=False, _exc=set()))
        >>> g = InternalRepresentation._compose_atom(f.op, bknowl.term, bknowl.range.end); g
        2*a^2 + 4*a*b + 2*b^2 + 1 <= 0
        >>> (f.lhs.poly / g.lhs.poly)
        3
        """
        rel = f.op
        lhs = f.lhs
        lc = lhs.lc()
        if lc < 0:
            rel = rel.converse()
        lhs /= lc
        q = -lhs.constant_coefficient()
        term = lhs + q
        # rel is the (possibly conversed) relation of atom, term is the monic
        # parametric part, and q is the negative constant coefficient.
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
        return cls(term, range_)


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

    def subs(self, subst: dict[Variable, Term]) -> Self:
        knowl = self.__class__()
        for bknowl in self:
            maybe_bknowl = bknowl.subs(subst)
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

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> abc.simplify.Restart:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.add`.
        """
        if gand is Or:
            atoms = (atom.to_complement() for atom in atoms)
        restart = abc.simplify.Restart.NONE
        for atom in atoms:
            # print(f'{atom=}')
            # print(f'{self=}')
            # assert all(v not in self._subst.as_dict() for v in atom.fvars())
            assert not atom.lhs.is_constant()
            maybe_bknowl = _BasicKnowledge.from_atom(atom).subs(self._subst.as_dict())
            if maybe_bknowl is None:
                continue
            bknowl = self._knowl.prune(maybe_bknowl)
            if self._knowl.is_known(maybe_bknowl):
                if bknowl.is_substitution():
                    self._propagate(bknowl)
                    restart = abc.simplify.Restart.ALL
                else:
                    self._knowl.add(bknowl)
                    if restart is not abc.simplify.Restart.ALL:
                        restart = abc.simplify.Restart.OTHERS
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
            self._knowl = self._knowl.subs(self._subst.as_dict())
            for bknowl in self._knowl:
                if bknowl.is_substitution():
                    stack.append(bknowl)

    @staticmethod
    @lru_cache(maxsize=CACHE_SIZE)
    def _compose_atom(rel: type[AtomicFormula], t: Term, q: mpq) -> AtomicFormula:
        num = mpq(q.numerator)
        den = mpq(q.denominator)
        return rel(den * t - num, 0)

    def extract(self, gand: type[And | Or], ref: Self) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.InternalRepresentation.extract`.
        """

        knowl = ref._knowl.subs(self._subst.as_dict())
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
            subst = {v: q for v, q in self._subst.as_dict().items() if v != var}
            knowl = ref._knowl.subs(subst)
            if val.variable is None:
                t: Term = var
                q = val.coefficient
            else:
                t = mpq(val.coefficient.denominator) * var \
                    - mpq(val.coefficient.numerator) * val.variable
                if t.lc() < 0:
                    t = -t
                q = mpq(0)
            if self._options.prefer_order and gand is Or and t in knowl:
                ref_range = knowl[t]
                if q == ref_range.start:
                    assert not ref_range.lopen
                    L.append(self._compose_atom(Le, t, q))
                elif q == ref_range.end:
                    assert not ref_range.ropen
                    L.append(self._compose_atom(Ge, t, q))
                else:
                    L.append(self._compose_atom(Eq, t, q))
            else:
                L.append(self._compose_atom(Eq, t, q))
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
        G = {key - val for key, val in self._subst.as_dict().items()}
        assert atom.rhs == 0
        lhs = atom.lhs.reduce(G)
        atom = atom.op(lhs, 0)
        if atom.lhs.is_constant():
            return Eq(0, 0) if bool(atom) else Eq(1, 0)
        if atom.lhs.lc() < 0:
            atom = atom.op.converse()(-atom.lhs, 0)
        return atom


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
            match simplified_atom:
                case AtomicFormula():
                    ir.add(And, [simplified_atom])
                case And(args=args):
                    assert all(isinstance(arg, AtomicFormula) for arg in args)
                    ir.add(And, args)
                case _T():
                    continue
                case _F():
                    raise InternalRepresentation.Inconsistent()
                case _:
                    assert False, simplified_atom
        return ir

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
        def _simpl_at_eq_ne(rel, lhs, context):

            def split_tsq(term):
                args = []
                for _, power_product in term:
                    if explode_always:
                        args.append(fac_junctor(*(rel(v, 0) for v in power_product.vars())))
                    else:
                        args.append(rel(product(power_product.vars()), 0))
                return tsq_junctor(*args)

            if rel is Eq:
                tsq_junctor = And
                fac_junctor = Or
            else:
                assert rel is Ne
                tsq_junctor = Or
                fac_junctor = And
            tsq = lhs.is_definite()
            if tsq == TSQ.STRICT:
                return tsq_junctor.definite_element()
            unit, _, factors = lhs.factor()
            primitive_lhs = Term(1)
            for factor in factors:
                # Square-free part
                primitive_lhs *= factor
            primitive_tsq = primitive_lhs.is_definite()
            if primitive_tsq == TSQ.STRICT:
                return tsq_junctor.definite_element()
            if primitive_tsq == TSQ.WEAK and (explode_always or context == tsq_junctor):
                return split_tsq(primitive_lhs)
            if tsq == TSQ.WEAK and (explode_always or context == tsq_junctor):
                return split_tsq(lhs)
            if explode_always or context == fac_junctor:
                args = (rel(factor, 0) for factor in factors if not factor.is_constant())
                return fac_junctor(*args)
            return rel(primitive_lhs, 0)

        def tsq_test_ge(f: Term, context: Optional[type[And | Or]]) -> Optional[Formula]:
            if f.is_definite() in (TSQ.STRICT, TSQ.WEAK):
                return _T()
            neg_tsq = (-f).is_definite()
            if neg_tsq == TSQ.STRICT:
                return _F()
            if neg_tsq == TSQ.WEAK:
                return _simpl_at_eq_ne(Eq, f, context)
            return None

        def _simpl_at_ge(lhs, context):
            # TSQ tests on original left hand side
            hit = tsq_test_ge(lhs, context)
            if hit is not None:
                return hit
            # Factorize
            unit, _, factors = lhs.factor()
            even_factors = []
            odd_factor = unit
            for factor, multiplicity in factors.items():
                if factor.is_definite() is TSQ.STRICT:
                    continue
                if multiplicity % 2 == 0:
                    even_factors.append(factor)
                else:
                    odd_factor *= factor
            even_factor = product(even_factors)
            signed_remaining_squarefree_part = odd_factor * even_factor
            # TSQ tests on factorization
            if odd_factor.is_definite() in (TSQ.STRICT, TSQ.WEAK):
                return _T()
            neg_tsq = (-odd_factor).is_definite()
            if neg_tsq == TSQ.STRICT:
                return _simpl_at_eq_ne(Eq, even_factor, context)
            if neg_tsq == TSQ.WEAK:
                return _simpl_at_eq_ne(Eq, signed_remaining_squarefree_part, context)
            hit = tsq_test_ge(signed_remaining_squarefree_part, context)
            if hit is not None:
                return hit
            # TSQ tests have failed
            if unit < 0:
                rel = Le
                odd_factor = - odd_factor
            else:
                rel = Ge
            if context is Or or explode_always:
                odd_part = rel(odd_factor, 0)
                even_part = (Eq(f, 0) for f in even_factors)
                return Or(odd_part, *even_part)
            return rel(odd_factor * even_factor ** 2, 0)

        lhs = atom.lhs - atom.rhs
        if lhs.is_constant():
            # In the following if-condition, the __bool__ method of atom.op
            # will be called.
            return _T() if atom.op(lhs, 0) else _F()
        lhs /= lhs.content()
        match atom:
            case Eq():
                return _simpl_at_eq_ne(Eq, lhs, context)
            case Ne():
                return _simpl_at_eq_ne(Ne, lhs, context)
            case Ge():
                return _simpl_at_ge(lhs, context)
            case Le():
                return _simpl_at_ge(-lhs, context)
            case Gt():
                if context is not None:
                    context = context.dual()
                return Not(_simpl_at_ge(-lhs, context)).to_nnf()
            case Lt():
                if context is not None:
                    context = context.dual()
                return Not(_simpl_at_ge(lhs, context)).to_nnf()
            case _:
                assert False


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
