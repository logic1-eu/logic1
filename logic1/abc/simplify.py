"""This Module provides a generic abstract implementation of simplifcication
 based on generating and propagating internal theories during recursion. This
 is the essentially the *standard simplifier*, which has been proposed for
 ordered fields in [DS97]_.
"""

import more_itertools

from abc import abstractmethod
from typing import cast, Generic, Iterable, Optional, Self, TypeVar

from ..firstorder import All, And, AtomicFormula, Ex, _F, Formula, Or, _T
from ..firstorder.formula import α, τ, χ

from ..support.tracing import trace  # noqa

# About Generic:
# https://stackoverflow.com/q/74103528/
# https://peps.python.org/pep-0484/

θ = TypeVar('θ', bound='Theory')


# discuss: The two instances could be abstract properties. They are not
# specified by the abstract class, because Simplify need not know about them.
# Other techniques (namely labels) are possible but not expected.
class Theory(Generic[α, τ, χ]):
    """This abstract class serves as an upper bound for the type variable
    :data:`θ` in :class:`.abc.simplify.Simplify`. It specifies an interface
    comprising methods required there.

    The principal idea is that a :class:`Theory` holds two instances of
    information equivalent to a conjunction of atomic formulas. In the course
    of recursive simplification in :class:`.abc.simplify.Simplify`, one of
    those is inherited from above, and the other one is enrichted with
    information from the toplevel of the subformula currently under
    consideration.
    """

    class Inconsistent(Exception):
        pass

    # discuss: do we need __init__ here?
    @abstractmethod
    def __init__(self) -> None:
        ...

    # discuss: Kann man das Or im Kommentar so schreiben? Du hattest mal
    # gesagt, dass man da was verbessern könnte. __iter__?
    @abstractmethod
    def add(self, gand: type[And[α, τ, χ] | Or[α, τ, χ]], atoms: Iterable[α]) -> None:
        """Add to this theory information originating from `atoms`. If `gand`
        is :class:`.And`, consider ``And(*atoms)``. If `gand` is
        :class:`.Or`, consider ``Or(Not(at) for at in atoms)``.
        """
        ...

    @abstractmethod
    def extract(self, gand: type[And[α, τ, χ] | Or[α, τ, χ]]) -> Iterable[α]:
        """Extract from this theory information that must be represented on the
        toplevel of the subformula currently under consideration. If `gand` is
        :class:`.And`, the result represents a conjunction.  If `gand` is
        :class:`.Or`,  it represents a disjunction.
        """
        ...

    @abstractmethod
    def next_(self, remove: Optional[χ] = None) -> Self:
        """Copy make the current information the inherited information, while
        removing all information involving the variable `remove`. If not
        :obj:`None`, the variable `remove` is quantified in the current
        recursion step.
        """
        ...


class Simplify(Generic[α, τ, χ, θ]):
    """Deep simplification following [DS97]_.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.simplify.Simplify`,
      :class:`.Sets.simplify.Simplify`
    """

    # Discuss: Polymorphie
    @property
    @abstractmethod
    def class_alpha(self) -> type[α]:
        """The class used to instantiate :data:`.α`. This allows to generically
        generate instances of that class within this abstract simplifier.
        Furthermore, it finds use in structural pattern matching.
        """
        ...

    @property
    @abstractmethod
    def class_theta(self) -> type[θ]:
        """The class used to instantiate :data:`.θ`. This allows to generically
        generate instances of that class within this abstract simplifier.
        Furthermore, it finds use in structural pattern matching.
        """
        ...

    # discuss: Kann man die nicht irgendwie mit currying in class_theta packen?
    # Ich wollte aus dem bool ein object machen, das schmeisst Typfehler.
    @property
    @abstractmethod
    def class_theta_kwargs(self) -> dict[str, bool]:
        """Keyword arguments to pass when using :attr:`class_theta` as a
        constructor.
        """
        ...

    # discsuss: rename to __call__?
    def simplify(self, f: Formula[α, τ, χ], assume: Optional[list[α]]) -> Formula[α, τ, χ]:
        """The main entry point to be used by the `__call__` method of
        subclasses within theories.
        """
        if assume is None:
            assume = []
        th = self.class_theta(**self.class_theta_kwargs)
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

    def _simpl_nnf(self, f: Formula[α, τ, χ], th: θ) -> Formula[α, τ, χ]:
        match f:
            case And() | Or():
                return self._simpl_and_or(f, th)
            case _F() | _T():
                return f
            case self.class_alpha():
                # Build a trivial binary And in order to apply th. Unary And
                # does not exist.
                return self._simpl_and_or(And(f, _T()), th)
            case _:
                raise NotImplementedError(f'Simplify does not know {f.op!r}')

    def _simpl_and_or(self, f: And[α, τ, χ] | Or[α, τ, χ], th: θ) -> Formula[α, τ, χ]:
        """
        `f` must be in negation normal form (NNF).
        """

        def split(args: Iterable[Formula[α, τ, χ]]) -> tuple[set[Formula[α, τ, χ]], set[α]]:
            """
            Returns the set of non-atoms and an iterator of atoms contained in
            :data:`args`, in that order.
            """
            def is_AT(f: Formula[α, τ, χ]) -> bool:
                if isinstance(f, self.class_alpha):
                    return True
                assert not isinstance(f, AtomicFormula), (type(f), f)
                return False

            i1, i2 = more_itertools.partition(is_AT, args)
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
        simplified_others: set[Formula] = set()
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
            elif isinstance(simplified_arg, self.class_alpha):
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

    @abstractmethod
    def simpl_at(self,
                 atom: α,
                 context: Optional[type[And[α, τ, χ]] | type[Or[α, τ, χ]]]) \
            -> Formula[α, τ, χ]:
        """Simplify the atomic formula `atom`. The `context` tells whether
        `atom` occurs within a conjunction or a disjunction. This can be taken
        into consideration for the inclusion of certain simplification
        strategies. For instance, simplification of ``xy == 0`` to ``Or(x == 0,
        y == 0)`` over the reals could be desirable within a disjunction but
        not otherwise.
        """
        # Does not receive the theory, by design.
        ...


# discuss: This is parameterized with an instance of Simplify in a weird way.
# Should is_valid better be a method of Simplify?
class IsValid(Generic[α, τ, χ]):
    """Simplification-based heuristic test for vailidity of a formula.

    .. admonition:: Mathematical definition

      A first-order formula is *valid* if it holds for all values all free
      variables.

    .. seealso::
      Derived classes in various theories: :class:`.RCF.simplify.IsValid`,
      :class:`.Sets.simplify.IsValid`
    """

    # discsuss: rename to __call__?
    def is_valid(self, f: Formula[α, τ, χ], assume: Optional[list[α]]) -> Optional[bool]:
        """The main entry point to be used by the `__call__` method of
        subclasses within theories. Returns :data:`True` or :data:`False` if
        :meth:`_simplify` succeeds in heuristically simplifying `f` to ``_T()``
        or ``_F()``, respectively. Returns :data:`None` in the sense of
        "don't know" otherwise.
        """
        if assume is None:
            assume = []
        match self._simplify(f, assume):
            case _T():
                return True
            case _F():
                return False
            case _:
                return None

    # discuss: _simplify ist das einzige was dokomentiert wird und hat einen
    # Underscore.
    @abstractmethod
    def _simplify(self, f: Formula[α, τ, χ], assume: list[α]) -> Formula[α, τ, χ]:
        """The simplifier to be used by :meth:`is_valid`.
        """
        ...
