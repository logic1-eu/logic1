"""This module :mod:`logic1.abc.simplify` provides a generic abstract
implementation of *deep simplifcication* based on generating and propagating
internal representations during recursion. This is essentially the *standard
simplifier*, which has been proposed for Ordered Fields in
[DolzmannSturm-1997]_.
"""

from abc import abstractmethod
from collections import deque
from enum import auto, Enum
from typing import Generic, Iterable, Optional, Self, TypeVar

from ..firstorder import (
    And, AtomicFormula, _F, Formula, Or, QuantifiedFormula, _T, Term, Variable)

from ..support.tracing import trace  # noqa

# About Generic:
# https://stackoverflow.com/q/74103528/
# https://peps.python.org/pep-0484/

α = TypeVar('α', bound='AtomicFormula')
τ = TypeVar('τ', bound='Term')
χ = TypeVar('χ', bound='Variable')
σ = TypeVar('σ')

ρ = TypeVar('ρ', bound='InternalRepresentation')


class Restart(Enum):
    NONE = auto()
    OTHERS = auto()
    ALL = auto()


class InternalRepresentation(Generic[α, τ, χ, σ]):
    """This abstract class serves as an upper bound for the type variable
    :data:`ρ` in :class:`.abc.simplify.Simplify`. It specifies an interface
    comprising methods required there.

    The principal idea is that a :class:`InternalRepresentation` should hold
    two abstract pieces of information, *reference* and *current*. Both
    *reference* and *current* hold  information that is equivalent to a
    conjunction of atomic formulas. In the course of recursive simplification
    in
    :class:`.abc.simplify.Simplify`, *reference*  is inherited from above;
    *current* starts with the information from *reference* and is enriched
    with information from all atomic formulas on the toplevel of the
    subformula currently under consideration.
    """

    class Inconsistent(Exception):
        pass

    @abstractmethod
    def add(self, gand: type[And[α, τ, χ, σ] | Or[α, τ, χ, σ]], atoms: Iterable[α]) -> Restart:
        """Add *current* information originating from `atoms`.
        If `gand` is :class:`.And`, consider ``atoms``. If `gand` is
        :class:`.Or`, consider ``(Not(at) for at in atoms)``. This is where
        simplification is supposed to take place.
        """
        ...

    @abstractmethod
    def extract(self, gand: type[And[α, τ, χ, σ] | Or[α, τ, χ, σ]]) -> Iterable[α]:
        """Comapare *current* and *reference* to identify and extract
        information that must be represented on the toplevel of the
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


class Simplify(Generic[α, τ, χ, σ, ρ]):
    """Deep simplification following [DolzmannSturm-1997]_.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.simplify.Simplify`,
      :class:`.Sets.simplify.Simplify`
    """

    @abstractmethod
    def create_initial_representation(self, assume: Iterable[α]) -> ρ:
        """Create a fresh instance of :class:`.ρ`.
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
    def simpl_at(self, atom: α, context: Optional[type[And[α, τ, χ, σ]] | type[Or[α, τ, χ, σ]]]) \
            -> Formula[α, τ, χ, σ]:
        """Simplify the atomic formula `atom`. The `context` tells whether
        `atom` occurs within a conjunction or a disjunction. This can be taken
        into consideration for the inclusion of certain simplification
        strategies. For instance, simplification of ``xy == 0`` to ``Or(x == 0,
        y == 0)`` over the reals could be desirable within a disjunction but
        not otherwise.
        """
        # Does not receive the internal representation, by design.
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
        try:
            ir = self.create_initial_representation(assume)
        except InternalRepresentation.Inconsistent:
            return _T()
        f = f.to_nnf(to_positive=True)
        return self._simpl_nnf(f, ir)

    def _simpl_nnf(self, f: Formula[α, τ, χ, σ], ir: ρ) -> Formula[α, τ, χ, σ]:
        if Formula.is_atomic(f):
            ir = ir.next_()
            return self._simpl_atomic(f, ir)
        if Formula.is_and(f) or Formula.is_or(f):
            ir = ir.next_()
            return self._simpl_and_or(f, ir)
        if Formula.is_quantified_formula(f):
            ir = ir.next_(remove=f.var)
            return self._simpl_quantified(f, ir)
        if Formula.is_true(f) or Formula.is_false(f):
            return f
        assert False, f

    def _simpl_and_or(self, f: And[α, τ, χ, σ] | Or[α, τ, χ, σ], ir: ρ) -> Formula[α, τ, χ, σ]:
        """
        `f` must be in negation normal form (NNF).
        """
        gand = f.op
        queue = deque(f.args)
        simplified_others: set[Formula[α, τ, χ, σ]] = set()
        while queue:
            arg = queue.popleft()
            if Formula.is_atomic(arg):
                simplified_arg = self.simpl_at(self.transform_atom(arg, ir), gand)
            else:
                simplified_arg = self._simpl_nnf(arg, ir)
            if isinstance(simplified_arg, gand.definite()):
                return simplified_arg
            elif isinstance(simplified_arg, gand.neutral()):
                new_atoms: set[α] = set()
                new_others: set[Formula[α, τ, χ, σ]] = set()
            elif isinstance(simplified_arg, gand):
                new_atoms = set()
                new_others = set()
                for arg in simplified_arg.args:
                    if Formula.is_atomic(arg):
                        new_atoms.add(arg)
                    else:
                        new_others.add(arg)
            elif Formula.is_atomic(simplified_arg):
                new_atoms = {simplified_arg}
                new_others = set()
            else:
                assert isinstance(simplified_arg, gand.dual()) \
                    or Formula.is_quantified_formula(simplified_arg)
                new_atoms = set()
                new_others = {simplified_arg}
            if new_atoms:
                try:
                    restart = ir.add(gand, new_atoms)
                except ir.Inconsistent:
                    return gand.definite_element()
                if restart is Restart.NONE:
                    simplified_others = simplified_others.union(new_others)
                else:  # Save resimp if ir has not changed
                    for simplified_other in simplified_others:
                        if simplified_other not in queue:
                            queue.append(simplified_other)
                    simplified_others = new_others  # subtle but correct
                    if restart is Restart.ALL:
                        pass
            else:
                simplified_others = simplified_others.union(new_others)
        final_atoms = list(ir.extract(gand))
        final_atoms.sort()
        final_others = list(simplified_others)
        final_others.sort()
        return gand(*final_atoms, *final_others)

    def _simpl_atomic(self, atom: α, ir: ρ) -> Formula[α, τ, χ, σ]:
        # This method is called for toplevel atoms and for atoms whose context
        # is a quantifier. Atoms with context And, Or are handled directly in
        # _simpl_and_or. At the moment a quantifier context is treated the same
        # way as a toplevel context.
        f = self.simpl_at(atom, context=None)
        if not Formula.is_atomic(f):
            return self._simpl_nnf(f, ir)
        try:
            ir.add(And, [f])
        except ir.Inconsistent:
            return _F()
        final_atoms = list(ir.extract(And))
        match len(final_atoms):
            case 0:
                return _T()
            case 1:
                return final_atoms[0]
            case _:
                assert False, final_atoms

    def _simpl_quantified(self, f: QuantifiedFormula[α, τ, χ, σ], ir: ρ) -> Formula[α, τ, χ, σ]:
        """
        `f` must be in negation normal form (NNF).
        """
        simplified_arg = self._simpl_nnf(f.arg, ir)
        return f.op(f.var, simplified_arg)

    def transform_atom(self, atom: α, ir: ρ) -> α:
        return atom
