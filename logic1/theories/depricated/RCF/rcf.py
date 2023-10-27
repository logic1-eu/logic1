import sympy
from typing import TypeAlias

from logic1 import atomlib
from logic1.support.decorators import classproperty
from logic1.firstorder.truth import Formula, F, T


Term: TypeAlias = atomlib.sympy.Term
Variable: TypeAlias = atomlib.sympy.Variable


class Eq(atomlib.sympy.Eq):

    @classproperty
    def complement_func(cls):
        """The complement relation Ne of Eq.
        """
        return Ne

    @classproperty
    def converse_func(cls):
        """The converse relation Eq of Eq.
        """
        return Eq

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        lhs = lhs.expand()
        if lhs == sympy.Integer(0):
            return T
        if not lhs.free_symbols:
            return F
        return Eq(lhs, 0)


class Ne(atomlib.sympy.Ne):

    @classproperty
    def complement_func(cls):
        """The complement relation Eq of Ne.
        """
        return Eq

    @classproperty
    def converse_func(cls):
        """The converse relation Ne of Ne.
        """
        return Ne

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        lhs = lhs.expand()
        if lhs == sympy.Integer(0):
            return F
        if not lhs.free_symbols:
            return T
        return Ne(lhs, 0)


class Ge(atomlib.sympy.Ge):

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Lt` of :class:`Ge`.
        """
        return Lt

    @classproperty
    def converse_func(cls):
        """The converse relation Le of Ge.
        """
        return Le

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        lhs = lhs.expand()
        if not lhs.free_symbols:
            return T if sympy.GreaterThan(lhs, sympy.Integer(0)) else F
        return Ge(lhs, 0)


class Le(atomlib.sympy.Le):

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Lt` of :class:`Ge`.
        """
        return Gt

    @classproperty
    def converse_func(cls):
        """The converse relation Le of Ge.
        """
        return Ge

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        lhs = lhs.expand()
        if not lhs.free_symbols:
            return T if sympy.LessThan(lhs, sympy.Integer(0)) else F
        return Le(lhs, 0)


class Gt(atomlib.sympy.Gt):

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Le` of :class:`Gt`.
        """
        return Le

    @classproperty
    def converse_func(cls):
        """The converse relation Lt of Gt.
        """
        return Lt

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        lhs = lhs.expand()
        if not lhs.free_symbols:
            return T if sympy.StrictGreaterThan(lhs, sympy.Integer(0)) else F
        return Gt(lhs, 0)


class Lt(atomlib.sympy.Lt):

    @classproperty
    def complement_func(cls):
        """The complement relation :class:`Ge` of :class:`Lt`.
        """
        return Ge

    @classproperty
    def converse_func(cls):
        """The converse relation Gt of Lt.
        """
        return Gt

    def simplify(self, Theta=None) -> Formula:
        lhs = self.lhs - self.rhs
        lhs = lhs.expand()
        if not lhs.free_symbols:
            return T if sympy.StrictLessThan(lhs, sympy.Integer(0)) else F
        return Lt(lhs, 0)


# We want to refer to RCF instances of AtomicFormula for type safety. The type
# alias supports :code:`cast(RcfAtomicFormula, ...)`. The tuple form supports
# :code:`assert isinstance(..., RcfAtomicFormulas)`.
RcfAtomicFormula: TypeAlias = Eq | Ne | Ge | Le | Gt | Lt
RcfAtomicFormulas = (Eq, Ne, Ge, Le, Gt, Lt)
