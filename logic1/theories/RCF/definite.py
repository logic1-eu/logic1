from enum import Enum, auto
from gmpy2 import mpq

class Definite(Enum):
    """Information whether a certain term has positive or negative definiteness
    properties; typically as a result of a heuristic test as in
    :meth:`.Term.is_definite`.
    """

    # This is an ordered Enum, the order of the following properties should not
    # be changed.
    UNKNOWN = auto()
    """It has not been derived that any the other cases holds.
    """

    ZERO = auto()
    """The polynomial is the zero polynomial.
    """

    POSITIVE = auto()
    """The polynomial positive definite, i.e., positive for all real choices of
    variables.
    """

    POSITIVE_SEMI = auto()
    """The polynomial positive semi-definite, i.e., non-negative for all real
    choices of variables.
    """

    NEGATIVE = auto()
    """The polynomial negative definite, i.e., negative for all real choices of
    variables.
    """

    NEGATIVE_SEMI = auto()
    """The polynomial negative semi-definite, i.e., non-positive for all real
    choices of variables.
    """

    # The following is an implementation of OrderedEnum as described in
    # https://docs.python.org/3/howto/enum.html#orderedenum
    def __ge__(self, other):
        if self.__class__ is other.__class__:
            return self.value >= other.value
        return NotImplemented

    def __gt__(self, other):
        if self.__class__ is other.__class__:
            return self.value > other.value
        return NotImplemented

    def __le__(self, other):
        if self.__class__ is other.__class__:
            return self.value <= other.value
        return NotImplemented

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return NotImplemented

    @staticmethod
    def add(x: Definite, y: Definite) -> Definite:
        """Compute Definite of a sum from Definite of the summands.

        >>> l = list(Definite)

        >>> for x in l:
        ...     for y in l:
        ...             print(f'{x.name} + {y.name} = {Definite.add(x,y).name}')
        ...
        UNKNOWN + UNKNOWN = UNKNOWN
        UNKNOWN + ZERO = UNKNOWN
        UNKNOWN + POSITIVE = UNKNOWN
        UNKNOWN + POSITIVE_SEMI = UNKNOWN
        UNKNOWN + NEGATIVE = UNKNOWN
        UNKNOWN + NEGATIVE_SEMI = UNKNOWN
        ZERO + UNKNOWN = UNKNOWN
        ZERO + ZERO = ZERO
        ZERO + POSITIVE = POSITIVE
        ZERO + POSITIVE_SEMI = POSITIVE_SEMI
        ZERO + NEGATIVE = NEGATIVE
        ZERO + NEGATIVE_SEMI = NEGATIVE_SEMI
        POSITIVE + UNKNOWN = UNKNOWN
        POSITIVE + ZERO = POSITIVE
        POSITIVE + POSITIVE = POSITIVE
        POSITIVE + POSITIVE_SEMI = POSITIVE
        POSITIVE + NEGATIVE = UNKNOWN
        POSITIVE + NEGATIVE_SEMI = UNKNOWN
        POSITIVE_SEMI + UNKNOWN = UNKNOWN
        POSITIVE_SEMI + ZERO = POSITIVE_SEMI
        POSITIVE_SEMI + POSITIVE = POSITIVE
        POSITIVE_SEMI + POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE_SEMI + NEGATIVE = UNKNOWN
        POSITIVE_SEMI + NEGATIVE_SEMI = UNKNOWN
        NEGATIVE + UNKNOWN = UNKNOWN
        NEGATIVE + ZERO = NEGATIVE
        NEGATIVE + POSITIVE = UNKNOWN
        NEGATIVE + POSITIVE_SEMI = UNKNOWN
        NEGATIVE + NEGATIVE = NEGATIVE
        NEGATIVE + NEGATIVE_SEMI = NEGATIVE
        NEGATIVE_SEMI + UNKNOWN = UNKNOWN
        NEGATIVE_SEMI + ZERO = NEGATIVE_SEMI
        NEGATIVE_SEMI + POSITIVE = UNKNOWN
        NEGATIVE_SEMI + POSITIVE_SEMI = UNKNOWN
        NEGATIVE_SEMI + NEGATIVE = NEGATIVE
        NEGATIVE_SEMI + NEGATIVE_SEMI = NEGATIVE_SEMI

        This addition is commutative:
        >>> all(Definite.add(x, y) is Definite.add(y, x) for x in l for y in l)
        True

        Definite.zero is a (unique) neutral element:
        >>> all(Definite.add(x, Definite.ZERO) is x for x in l)
        True
        """
        x, y = sorted([x, y])
        if x is Definite.UNKNOWN:
            return Definite.UNKNOWN
        if x is Definite.ZERO:
            return y
        if x is Definite.POSITIVE:
            if y is Definite.POSITIVE or y is Definite.POSITIVE_SEMI:
                return Definite.POSITIVE
            assert y is Definite.NEGATIVE or y is Definite.NEGATIVE_SEMI, (x, y)
            return Definite.UNKNOWN
        if x is Definite.POSITIVE_SEMI:
            if y is Definite.POSITIVE_SEMI:
                return Definite.POSITIVE_SEMI
            assert y is Definite.NEGATIVE or y is Definite.NEGATIVE_SEMI, (x, y)
            return Definite.UNKNOWN
        if x is Definite.NEGATIVE:
            assert y is Definite.NEGATIVE or y is Definite.NEGATIVE_SEMI, (x, y)
            return Definite.NEGATIVE
        assert x is Definite.NEGATIVE_SEMI, (x, y)
        assert y is Definite.NEGATIVE_SEMI, (x, y)
        return Definite.NEGATIVE_SEMI

    @staticmethod
    def from_constant(q: int | mpq) -> Definite:
        """Compute Definite of a number.

        >>> print(Definite.from_constant(mpq(42)))
        Definite.POSITIVE

        >>> print(Definite.from_constant(mpq(-4711)))
        Definite.NEGATIVE

        >>> print(Definite.from_constant(mpq(0)))
        Definite.ZERO
        """
        assert isinstance(q, (int, mpq)), q
        if q > 0:
            return Definite.POSITIVE
        if q < 0:
            return Definite.NEGATIVE
        assert q == 0, q
        return Definite.ZERO

    @staticmethod
    def mul(x: Definite, y: Definite) -> Definite:
        """Compute Definite of a product from Definite of the factors.

        >>> l = list(Definite)

        The multiplication table:
        >>> for x in l:
        ...     for y in l:
        ...             print(f'{x.name} * {y.name} = {Definite.mul(x,y).name}')
        ...
        UNKNOWN * UNKNOWN = UNKNOWN
        UNKNOWN * ZERO = ZERO
        UNKNOWN * POSITIVE = UNKNOWN
        UNKNOWN * POSITIVE_SEMI = UNKNOWN
        UNKNOWN * NEGATIVE = UNKNOWN
        UNKNOWN * NEGATIVE_SEMI = UNKNOWN
        ZERO * UNKNOWN = ZERO
        ZERO * ZERO = ZERO
        ZERO * POSITIVE = ZERO
        ZERO * POSITIVE_SEMI = ZERO
        ZERO * NEGATIVE = ZERO
        ZERO * NEGATIVE_SEMI = ZERO
        POSITIVE * UNKNOWN = UNKNOWN
        POSITIVE * ZERO = ZERO
        POSITIVE * POSITIVE = POSITIVE
        POSITIVE * POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE * NEGATIVE = NEGATIVE
        POSITIVE * NEGATIVE_SEMI = NEGATIVE_SEMI
        POSITIVE_SEMI * UNKNOWN = UNKNOWN
        POSITIVE_SEMI * ZERO = ZERO
        POSITIVE_SEMI * POSITIVE = POSITIVE_SEMI
        POSITIVE_SEMI * POSITIVE_SEMI = POSITIVE_SEMI
        POSITIVE_SEMI * NEGATIVE = NEGATIVE_SEMI
        POSITIVE_SEMI * NEGATIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE * UNKNOWN = UNKNOWN
        NEGATIVE * ZERO = ZERO
        NEGATIVE * POSITIVE = NEGATIVE
        NEGATIVE * POSITIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE * NEGATIVE = POSITIVE
        NEGATIVE * NEGATIVE_SEMI = POSITIVE_SEMI
        NEGATIVE_SEMI * UNKNOWN = UNKNOWN
        NEGATIVE_SEMI * ZERO = ZERO
        NEGATIVE_SEMI * POSITIVE = NEGATIVE_SEMI
        NEGATIVE_SEMI * POSITIVE_SEMI = NEGATIVE_SEMI
        NEGATIVE_SEMI * NEGATIVE = POSITIVE_SEMI
        NEGATIVE_SEMI * NEGATIVE_SEMI = POSITIVE_SEMI

        This multiplication is commutative:
        >>> all(Definite.mul(x, y) is Definite.mul(y, x) for x in l for y in l)
        True

        Definite.POSITIVE is a (unique) neutral element:
        >>> all(Definite.mul(x, Definite.POSITIVE) is x for x in l)
        True
        """
        x, y = sorted([x, y])
        if x is Definite.UNKNOWN:
            if y is Definite.ZERO:
                return Definite.ZERO
            return Definite.UNKNOWN
        if x is Definite.ZERO:
            return Definite.ZERO
        if x is Definite.POSITIVE:
            return y
        if x is Definite.POSITIVE_SEMI:
            if y is Definite.POSITIVE_SEMI:
                return Definite.POSITIVE_SEMI
            assert y is Definite.NEGATIVE or y is Definite.NEGATIVE_SEMI, (x, y)
            return Definite.NEGATIVE_SEMI
        if x is Definite.NEGATIVE:
            if y is Definite.NEGATIVE:
                return Definite.POSITIVE
            assert y is Definite.NEGATIVE_SEMI, (x, y)
            return Definite.POSITIVE_SEMI
        assert x is Definite.NEGATIVE_SEMI, (x, y)
        assert y is Definite.NEGATIVE_SEMI, (x, y)
        return Definite.POSITIVE_SEMI

    @staticmethod
    def square(x: Definite) -> Definite:
        if x is Definite.UNKNOWN:
            return Definite.POSITIVE_SEMI
        return Definite.mul(x, x)
