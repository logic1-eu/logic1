"""This module :mod:`logic1.abc.simplify` provides a generic abstract
implementation of *deep simplifcication* based on generating and propagating
internal theories during recursion. This is essentially the *standard
simplifier*, which has been proposed for Ordered Fields in [DolzmannSturm-1997]_.
"""

import more_itertools

from abc import abstractmethod
from typing import cast, Generic, Iterable, Optional, Self, TypeVar

from ..firstorder import (
    All, And, AtomicFormula, Ex, _F, Formula, Or, _T, Term, Variable)

from ..support.tracing import trace  # noqa

# About Generic:
# https://stackoverflow.com/q/74103528/
# https://peps.python.org/pep-0484/

α = TypeVar('α', bound='AtomicFormula')
τ = TypeVar('τ', bound='Term')
χ = TypeVar('χ', bound='Variable')
σ = TypeVar('σ')

θ = TypeVar('θ', bound='Theory')


class Theory(Generic[α, τ, χ, σ]):
    """This abstract class serves as an upper bound for the type variable
    :data:`θ` in :class:`.abc.simplify.Simplify`. It specifies an interface
    comprising methods required there.

    The principal idea is that a :class:`Theory` should hold two abstract
    pieces of information, *reference* and *current*. Both *reference* and
    *current* hold  information that is equivalent to a conjunction of atomic
    formulas. In the course of recursive simplification in
    :class:`.abc.simplify.Simplify`, *reference*  is inherited from above;
    *current* starts with the information from *reference* and is enriched
    with information from all atomic formulas on the toplevel of the
    subformula currently under consideration.
    """

    class Inconsistent(Exception):
        pass

    @abstractmethod
    def add(self, gand: type[And[α, τ, χ, σ] | Or[α, τ, χ, σ]], atoms: Iterable[α]) -> None:
        """Add to this theory's *current* information originating from `atoms`.
        If `gand` is :class:`.And`, consider ``atoms``. If `gand` is
        :class:`.Or`, consider ``(Not(at) for at in atoms)``. This is where
        simplification is supposed to take place.
        """
        ...

    @abstractmethod
    def extract(self, gand: type[And[α, τ, χ, σ] | Or[α, τ, χ, σ]]) -> Iterable[α]:
        """Comapare *current* and *reference* to identify and extract from this
        theory information that must be represented on the toplevel of the
        subformula currently under consideration. If `gand` is :class:`.And`,
        the result represents a conjunction.  If `gand` is :class:`.Or`,  it
        represents a disjunction.
        """
        ...

    @abstractmethod
    def next_(self, remove: Optional[χ] = None) -> Self:
        """Copy  *current* to *reference*, removing all information involving
        the variable `remove`. If not :obj:`None`, the variable `remove` is
        quantified in the current recursion step.
        """
        ...


class Simplify(Generic[α, τ, χ, σ, θ]):
    """Deep simplification following [DolzmannSturm-1997]_.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.simplify.Simplify`,
      :class:`.Sets.simplify.Simplify`
    """

    @abstractmethod
    def create_initial_theory(self) -> θ:
        """Create a fresh instance of :class:`.θ`.
        """
        ...

    def is_valid(self, f: Formula[α, τ, χ, σ], assume: Iterable[α] = []) \
            -> Optional[bool]:
        """Simplification-based heuristic test for vailidity of a formula.

        .. admonition:: Mathematical definition

          A first-order formula is *valid* if it holds for all values all free
          variables.

        :param f:
          The formula to be tested for validity

        :param assume:
          A list of atomic formulas that are assumed to hold. The result of the
          validity test is correct modulo these assumptions.

        :returns: Returns :data:`True` or :data:`False` if
          :meth:`.abc.simplify.Simplify.simplify` succeeds in heuristically
          simplifying `f` to :data:`.T` or :data:`.F`, respectively. Returns
          :data:`None` in the sense of "don't know" otherwise.
        """
        match self.simplify(f, assume):
            case _T():
                return True
            case _F():
                return False
            case _:
                return None

    @abstractmethod
    def simpl_at(self,
                 atom: α,
                 context: Optional[type[And[α, τ, χ, σ]] | type[Or[α, τ, χ, σ]]]) \
            -> Formula[α, τ, χ, σ]:
        """Simplify the atomic formula `atom`. The `context` tells whether
        `atom` occurs within a conjunction or a disjunction. This can be taken
        into consideration for the inclusion of certain simplification
        strategies. For instance, simplification of ``xy == 0`` to ``Or(x == 0,
        y == 0)`` over the reals could be desirable within a disjunction but
        not otherwise.
        """
        # Does not receive the theory, by design.
        ...

    def simplify(self, f: Formula[α, τ, χ, σ], assume: Iterable[α] = []) -> Formula[α, τ, χ, σ]:
        """Simplify `f` modulo `assume`.

        :param f:
          The formula to be simplified

        :param assume: A list of atomic formulas that are assumed to hold. The
          simplification result is equivalent modulo those assumptions.

        :returns:
          A simplified equivalent of `f` modulo `assume`.
        """
        th = self.create_initial_theory()
        try:
            th.add(And, assume)
        except th.Inconsistent:
            return _T()
        th = th.next_()
        f = f.to_pnf()
        quantifiers = []
        while isinstance(f, (Ex, All)):
            th = th.next_(remove=f.var)
            quantifiers.append((f.op, f.var))
            f = f.arg
        f = self._simpl_nnf(f, th)
        free_vars = set(f.fvars())
        for Q, var in reversed(quantifiers):
            if var in free_vars:
                f = Q(var, f)
        return f

    def _simpl_nnf(self, f: Formula[α, τ, χ, σ], th: θ) -> Formula[α, τ, χ, σ]:
        match f:
            case And() | Or():
                return self._simpl_and_or(f, th)
            case _F() | _T():
                return f
            case AtomicFormula():
                # Build a trivial binary And in order to apply th. Unary And
                # does not exist.
                return self._simpl_and_or(And(f, _T()), th)
            case _:
                raise NotImplementedError(f'Simplify does not know {f.op!r}')

    def _simpl_and_or(self, f: And[α, τ, χ, σ] | Or[α, τ, χ, σ], th: θ) -> Formula[α, τ, χ, σ]:
        """
        `f` must be in negation normal form (NNF).
        """

        def split(args: Iterable[Formula[α, τ, χ, σ]]) -> tuple[set[Formula[α, τ, χ, σ]], set[α]]:
            """
            Returns the set of non-atoms and an iterator of atoms contained in
            :data:`args`, in that order.
            """
            i1, i2 = more_itertools.partition(Formula.is_atomic, args)
            return set(i1), cast(set[α], set(i2))

        gand = f.op
        others, atoms = split(f.args)
        simplified_atoms = (self.simpl_at(atom, f.op) for atom in atoms)
        new_others, atoms = split(simplified_atoms)
        others = others.union(new_others)
        try:
            th.add(gand, atoms)
        except th.Inconsistent:
            return gand.definite_element()
        simplified_others: set[Formula[α, τ, χ, σ]] = set()
        while others:
            arg = others.pop()
            simplified_arg = self._simpl_nnf(arg, th.next_())
            if isinstance(simplified_arg, gand.definite()):
                return simplified_arg
            elif isinstance(simplified_arg, gand.neutral()):
                new_others = set()
                new_atoms: Iterable[α] = ()
            elif isinstance(simplified_arg, gand):
                new_others, new_atoms = split(simplified_arg.args)
            elif Formula.is_atomic(simplified_arg):
                new_others = set()
                new_atoms = (simplified_arg,)
            elif isinstance(simplified_arg, gand.dual()):
                new_others = {simplified_arg}
                new_atoms = ()
            else:
                raise NotImplementedError(f'unknown operator {simplified_arg.op} in {f}')
            if new_atoms:
                try:
                    th.add(gand, new_atoms)  # Can save resimp if th does not change
                except th.Inconsistent:
                    return gand.definite_element()
                others = others.union(simplified_others)
                simplified_others = new_others
            else:
                simplified_others = simplified_others.union(new_others)
        final_atoms = list(th.extract(gand))
        final_atoms.sort()
        final_others = list(simplified_others)
        final_others.sort()
        return gand(*final_atoms, *final_others)
