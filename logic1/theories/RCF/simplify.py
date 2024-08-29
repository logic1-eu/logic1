"""This module provides an implementation of *deep simplifcication* based on
generating and propagating internal theories during recursion in Real Closed
fields. This is essentially the *standard simplifier*, which has been proposed
for Ordered Fields in [DolzmannSturm-1997]_.
"""

from functools import lru_cache
from operator import xor
from sage.all import oo, product, Rational  # type: ignore
from sage.rings.infinity import MinusInfinity, PlusInfinity  # type: ignore
from typing import Iterable, Optional, Self

from ... import abc

from ...firstorder import And, _F, Not, Or, _T
from .atomic import AtomicFormula, Eq, Ge, Le, Gt, Lt, Ne, Term, TSQ, Variable
from .typing import Formula

from ...support.tracing import trace  # noqa


class Theory(abc.simplify.Theory[AtomicFormula, Term, Variable, int]):
    """Implements the abstract methods :meth:`add() <.abc.simplify.Theory.add>`,
    :meth:`extract() <.abc.simplify.Theory.extract>`, and :meth:`next_()
    <.abc.simplify.Theory.next_>` of it super class
    :class:`.abc.simplify.Theory`. Required by
    :class:`.Sets.simplify.Simplify` for instantiating the type variable
    :data:`.abc.simplify.θ` of :class:`.abc.simplify.Simplify`.
    """

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

    def __init__(self, prefer_weak: bool, prefer_order: bool) -> None:
        self.prefer_weak = prefer_weak
        self.prefer_order = prefer_order
        self._reference = dict()
        self._current = dict()

    def __repr__(self):
        return f'Theory({self._reference}, {self._current})'

    def add(self, gand: type[And | Or], atoms: Iterable[AtomicFormula]) -> None:
        """Implements the abstract method :meth:`.abc.simplify.Theory.add`.
        """
        for atom in atoms:
            # rel is the relation of atom, p is the parametric part, and q is
            # the negative of the Rational absolute summand.
            rel, p, q = self._decompose_atom(atom)
            if gand is Or:
                rel = rel.complement()
            # We model p in ivl \ exc.
            if rel is Eq:
                ivl = Theory._Interval(False, q, q, False)
                exc = set()
            elif rel is Ne:
                ivl = Theory._Interval(True, -oo, oo, True)
                exc = {q}
            elif rel is Ge:
                ivl = Theory._Interval(False, q, oo, True)
                exc = set()
            elif rel is Le:
                ivl = Theory._Interval(True, -oo, q, False)
                exc = set()
            elif rel is Gt:
                ivl = Theory._Interval(True, q, oo, True)
                exc = set()
            elif rel is Lt:
                ivl = Theory._Interval(True, -oo, q, True)
                exc = set()
            else:
                assert False, rel
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
                    # It follows that ivl is left-closed. Theory._Interval
                    # raises Inconsistent if ivl gets empty.
                    ivl = Theory._Interval(True, ivl.start, ivl.end, ivl.ropen)
                    exc = exc.difference({ivl.start})
                if ivl.end in exc:
                    # It follows that ivl is right-closed. ivl cannot get emty
                    # here.
                    ivl = Theory._Interval(ivl.lopen, ivl.start, ivl.end, True)
                    exc = exc.difference(ivl.end)
            self._current[p] = (ivl, exc)

    @staticmethod
    @lru_cache(maxsize=None)
    def _compose_atom(rel: type[AtomicFormula], p: Term, q: Rational)\
            -> AtomicFormula:
        num = q.numerator()
        den = q.denominator()
        return rel(den * p - num, 0)

    @staticmethod
    @lru_cache(maxsize=None)
    def _decompose_atom(f: AtomicFormula)\
            -> tuple[type[AtomicFormula], Term, Rational]:
        r"""Decompose into relation :math:`\rho`, term :math:`p` without
        absolute summand, and rational :math:`q` such that :data:`f` is
        equivalent to :math:`p \rho q`.

        We assume that :data:`f` has gone through :meth:`_simpl_at` so that
        its right hand side is zero and its left hand side polynomial has
        gone through SymPy's :meth:`expand`.

        >>> from .atomic import VV
        >>> a, b = VV.get('a', 'b')
        >>> f = 6*a**2 + 12*a*b + 6*b**2 + 3 <= 0
        >>> rel, p, q = Theory._decompose_atom(f); rel, p, q
        (<class 'logic1.theories.RCF.atomic.Le'>, a^2 + 2*a*b + b^2, -1/2)
        >>> g = Theory._compose_atom(rel, p, q); g
        2*a^2 + 4*a*b + 2*b^2 + 1 <= 0
        >>> (f.lhs.poly / g.lhs.poly)
        3
        """
        lhs = f.lhs
        q = -lhs.constant_coefficient()
        p = lhs + q
        c = p.content()
        p /= c
        # Given that _simpl_at has produced a primitive polynomial, q != 0
        # will not be divisible by c. This is relevant for the reconstruction
        # in _compose_atom to work. assert c == 1 or not q % c == 0,
        # f'{c} divides {q}'
        return f.op, p, Rational(q) / Rational(c)

    def extract(self, gand: type[And | Or]) -> list[AtomicFormula]:
        """Implements the abstract method :meth:`.abc.simplify.Theory.extract`.
        """
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
        """Implements the abstract method :meth:`.abc.simplify.Theory.next_`.
        """
        theory_next = self.__class__(self.prefer_weak, self.prefer_order)
        if remove is None:
            theory_next._reference = self._current
        else:
            theory_next._reference = {p: q for p, q in self._current.items()
                                      if remove not in p.vars()}
        theory_next._current = theory_next._reference.copy()
        return theory_next


class Simplify(abc.simplify.Simplify[AtomicFormula, Term, Variable, int, Theory]):
    """Deep simplification following [DolzmannSturm-1997]_. Implements the
    abstract methods :meth:`create_initial_theory
    <.abc.simplify.Simplify.create_initial_theory>` and :meth:`simpl_at
    <.abc.simplify.Simplify.simpl_at>` of its super class
    :class:`.abc.simplify.Simplify`.

    The simplifier should be called via :func:`.simplify`, as described below.
    In addition, this class inherits :meth:`.abc.simplify.Simplify.is_valid`,
    which should be called via :func:`.is_valid`, as described below.
    """

    explode_always: bool = True
    prefer_order: bool = True
    prefer_weak: bool = False

    def create_initial_theory(self) -> Theory:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.create_initial_theory`.
        """
        return Theory(prefer_weak=self.prefer_weak, prefer_order=self.prefer_order)

    def simplify(self,
                 f: Formula,
                 assume: Iterable[AtomicFormula] = [],
                 explode_always: bool = True,
                 prefer_order: bool = True,
                 prefer_weak: bool = False) -> Formula:
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
        self.explode_always = explode_always
        self.prefer_order = prefer_order
        self.prefer_weak = prefer_weak
        return super().simplify(f, assume)

    def simpl_at(self,
                 atom: AtomicFormula,
                 context: Optional[type[And] | type[Or]]) -> Formula:
        """Implements the abstract method
        :meth:`.abc.simplify.Simplify.simpl_at`.
        """
        # MyPy does not recognize that And[Any, Any, Any] is an instance of
        # Hashable. https://github.com/python/mypy/issues/11470
        return self._simpl_at(atom, context, self.explode_always)  # type: ignore[arg-type]

    @lru_cache(maxsize=None)
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
                return _simpl_at_ge(- lhs, context)
            case Gt():
                if context is not None:
                    context = context.dual()
                return Not(_simpl_at_ge(- lhs, context)).to_nnf()
            case Lt():
                if context is not None:
                    context = context.dual()
                return Not(_simpl_at_ge(lhs, context)).to_nnf()
            case _:
                assert False


simplify = Simplify().simplify


is_valid = Simplify().is_valid
