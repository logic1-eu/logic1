"""This module :mod:`logic1.abc.simplify` provides a generic abstract
implementation of *deep simplifcication* based on generating and propagating
internal representations during recursion. This is essentially the *standard
simplifier*, which has been proposed for Ordered Fields in
[DolzmannSturm-1997]_.
"""

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
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

ω = TypeVar('ω', bound='Options')
"""A type variable denoting a options for
:meth:`.Simplify.simplify` with upper bound :class:`.Options`.
"""


class RESTART(Enum):
    """Used for the return value of :meth:`.InternalRepresentation:add`.
    """
    NONE = auto()
    """No formulas of the current level require resimplification.
    """

    OTHERS = auto()
    """Non-atoms of the current level require resimplification.
    """

    ALL = auto()
    """All formulas of the current level require resimplification.
    """


class InternalRepresentation(Generic[α, τ, χ, σ]):
    """This abstract class serves as an upper bound for the type variable
    :data:`ρ` in :class:`.abc.simplify.Simplify`. It specifies an interface
    comprising methods required there.

    The principal idea is that a :class:`InternalRepresentation` should holds
    information that corresponds to a conjunction of atomic formulas. In the
    course of recursive simplification in:class:`.abc.simplify.Simplify`,
    instances of this class are inherited from higher levels and are enriched
    with information from all atomic formulas on the toplevel of the subformula
    currently under consideration.
    """

    class Inconsistent(Exception):
        """Indicates that an instance of :class:`InternalRepresentation`
        contains inconsistent information. This exception is typically handled
        in :class:`.abc.Simplify` and its derived classes, where appropriate
        values are returned.
        """
        pass

    @abstractmethod
    def add(self, gand: type[And[α, τ, χ, σ] | Or[α, τ, χ, σ]], atoms: Iterable[α]) -> RESTART:
        """Add information originating from `atoms`. If `gand` is
        :class:`.And`, consider ``atoms``. If `gand` is :class:`.Or`, consider
        ``(Not(at) for at in atoms)``. Simplification among atoms
         is supposed to take place here.
        """
        ...

    @abstractmethod
    def extract(self, gand: type[And[α, τ, χ, σ] | Or[α, τ, χ, σ]], ref: Self) -> Iterable[α]:
        """Comapare `self`and `ref` to identify and extract information that
        must be represented on the toplevel of the subformula currently under
        consideration. If `gand` is :class:`.And`, the result represents a
        conjunction.  If `gand` is :class:`.Or`,  it represents a disjunction.
        """
        ...

    @abstractmethod
    def next_(self, remove: Optional[χ] = None) -> Self:
        """Create a copy of `self`, optionally removing all information
        involving the variable `remove`.
        """
        ...

    def restart(self, ir: Self) -> Self:
        assert False

    def transform_atom(self, atom: α) -> α:
        return atom


class Options(ABC):
    """This class holds options that can be provided to
    :meth:`.Simplify.simplify`. Theories subclassing
    :class:`.Simplify` can add further options by subclassing
    :class:`.Options`.

    This is an upper bound for the type variable :data:`.ω`.
    """
    pass


@dataclass(frozen=True)
class Simplify(Generic[α, τ, χ, σ, ρ, ω]):
    """Deep simplification following [DolzmannSturm-1997]_.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.simplify.Simplify`,
      :class:`.Sets.simplify.Simplify`
    """

    _options: ω
    """The options that have been passed to :meth:`.simplify`.
    """

    @abstractmethod
    def create_initial_representation(self, assume: Iterable[α]) -> ρ:
        """Create a fresh instance of :class:`.ρ`.
        """
        ...

    def is_valid(self, f: Formula[α, τ, χ, σ], assume: Iterable[α] = []) -> Optional[bool]:
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
        f = self.simplify(f, assume)
        if f is _T():
            return True
        if f is _F():
            return False
        return None

    def _post_process(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        """A hook for post-processing the final result in subclasses.
        """
        return f

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
        f = self._simpl_nnf(f, ir)
        f = self._post_process(f)
        return f

    def _simpl_nnf(self, f: Formula[α, τ, χ, σ], ir: ρ) -> Formula[α, τ, χ, σ]:
        """Simplify the negation normal form `f` modulo `ir`.
        """
        if Formula.is_atomic(f):
            return self._simpl_atomic(f, ir)
        if Formula.is_and(f) or Formula.is_or(f):
            return self._simpl_and_or(f, ir)
        if Formula.is_quantified_formula(f):
            return self._simpl_quantified(f, ir)
        if Formula.is_true(f) or Formula.is_false(f):
            return f
        assert False, f

    def _simpl_and_or(self, f: And[α, τ, χ, σ] | Or[α, τ, χ, σ], ir: ρ) -> Formula[α, τ, χ, σ]:
        """Simplify the negation normal form `f`, which starts with either
        :class:`.And` or :class:`.Or`, modulo `ir`.
        """

        # def log(msg: str):
        #     from IPython.lib import pretty
        #     print('+' + (78 - len(msg)) * '-' + ' ' + msg)
        #     pretty_f = pretty.pretty(f, newline='\n|   ')
        #     pretty_nodes = pretty.pretty(ir._subst.nodes, newline='\n| ' + len('ir._subst.nodes=') * ' ')
        #     pretty_ref_nodes = pretty.pretty(ref._subst.nodes, newline='\n| ' + len('ref._subst.nodes=') * ' ')
        #     print(f'| f={pretty_f}\n| {ir._knowl=!s}\n| ir._subst.nodes={pretty_nodes}\n| {ref._knowl=!s}\n| ref._subst.nodes={pretty_ref_nodes}')
        #     print('+' + 79 * '-')

        ref = ir
        ir = ir.next_()
        gand = f.op
        queue = deque(f.args)
        simplified_others: set[Formula[α, τ, χ, σ]] = set()
        while queue:
            arg = queue.popleft()
            if Formula.is_atomic(arg):
                simplified_arg = self.simpl_at(ir.transform_atom(arg), context=gand)
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
                assert isinstance(simplified_arg, (gand.dual(), QuantifiedFormula))
                new_atoms = set()
                new_others = {simplified_arg}
            if new_atoms:
                try:
                    restart = ir.add(gand, new_atoms)
                except ir.Inconsistent:
                    return gand.definite_element()
                if restart is RESTART.NONE:  # Save resimp if ir has not changed
                    simplified_others = simplified_others.union(new_others)
                elif restart is RESTART.OTHERS:
                    for simplified_other in simplified_others:
                        if simplified_other not in queue:
                            queue.append(simplified_other)
                    simplified_others = new_others  # subtle but correct
                else:
                    assert restart is RESTART.ALL
                    for simplified_other in simplified_others:
                        if simplified_other not in queue:
                            queue.append(simplified_other)
                    simplified_others = new_others
                    for atom in ir.extract(gand, ref):
                        if atom not in queue:
                            queue.appendleft(atom)
                    # queue = deque(f.args)
                    ir = ref.restart(ir)
            else:
                simplified_others = simplified_others.union(new_others)
        final_atoms = list(ir.extract(gand, ref))
        final_atoms.sort()
        final_others = list(simplified_others)
        final_others.sort()
        return gand(*final_atoms, *final_others)

    def _simpl_atomic(self, atom: α, ir: ρ) -> Formula[α, τ, χ, σ]:
        """Simplify `atom`, which either stands on the toplevel or is the
        argument formula of a quantifier, modulo `ir`. At the moment, there is
        no difference made between these two cases. Argument atoms of
        :class:`.And`, :class:`.Or` are handled directly in
        :meth:`._simpl_and_or`.
        """
        ref = ir
        ir = ir.next_()
        f = self.simpl_at(atom, context=None)
        if not Formula.is_atomic(f):
            return self._simpl_nnf(f, ir)
        try:
            ir.add(And, [f])
        except ir.Inconsistent:
            return _F()
        final_atoms = list(ir.extract(And, ref))
        if len(final_atoms) == 0:
            return _T()
        if len(final_atoms) == 1:
            return final_atoms[0]
        assert False, final_atoms

    def _simpl_quantified(self, f: QuantifiedFormula[α, τ, χ, σ], ir: ρ) -> Formula[α, τ, χ, σ]:
        """Simplify the negation normal form `f`, which starts with either
        :class:`.Ex` or :class:`.All`, modulo `ir`.
        """
        ir = ir.next_(remove=f.var)
        simplified_arg = self._simpl_nnf(f.arg, ir)
        return f.op(f.var, simplified_arg)
