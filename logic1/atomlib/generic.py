from abc import ABC, abstractmethod

from ..firstorder import F, T


class BinaryAtomicFormulaMixin(ABC):

    @classmethod
    @abstractmethod
    def converse(cls):
        ...

    @classmethod
    def dual(cls):
        """The dual class of :class:`cls`.
        """
        return cls.complement().converse()

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
