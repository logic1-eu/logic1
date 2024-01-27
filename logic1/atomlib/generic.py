from ..firstorder import F, T
from ..support.decorators import classproperty


class BinaryAtomicFormulaMixin:

    @classproperty
    def converse_func(cls):
        # Should be an abstract class property
        raise NotImplementedError

    @classproperty
    def dual_func(cls):
        """The dual class of :class:`cls`.

        There is an implicit assumption that there are abstract class
        properties `complement_func` and `converse_func` specified, which is
        technically not possible at the moment.
        """
        return cls.complement_func.converse_func

    @property
    def lhs(self):
        """The left-hand side of a binary relation.
        """
        return self.args[0]

    @property
    def rhs(self):
        """The right-hand side of a binary relation.
        """
        return self.args[1]


class EqMixin:

    def simplify(self):
        if self.lhs == self.rhs:
            return T
        return self


class NeMixin:

    def simplify(self):
        if self.lhs == self.rhs:
            return F
        return self


class GeMixin:

    def simplify(self):
        if self.lhs == self.rhs:
            return T
        return self


class LeMixin:

    def simplify(self):
        if self.lhs == self.rhs:
            return T
        return self


class GtMixin:

    def simplify(self):
        if self.lhs == self.rhs:
            return F
        return self


class LtMixin:

    def simplify(self):
        if self.lhs == self.rhs:
            return F
        return self
