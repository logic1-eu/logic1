class GetVars():

    @property
    def all(self) -> set:
        """
        The set of all variables, free and bound.

        >>> from sympy.abc import a, b, c, x, y, z
        >>> u = GetVars(free={a, b, x}, bound={a, b, y, z})
        >>> u.all ==  {a, b, x, y, z}
        True
        """
        return self.free | self.bound

    def __eq__(self, other):
        """
        >>> from sympy.abc import x, y, z
        >>> v = GetVars(free={x}, bound={y, z})
        >>> w = GetVars(free={x}, bound={y, z})
        >>> v == w
        True
        >>> v is w
        False
        """
        return self.free == other.free and self.bound == other.bound

    def __init__(self, free: set = set(), bound: set = set()):
        self.free = free
        self.bound = bound

    def __ior__(self, other):
        """Override the bitwise or operator ``|=`` for union.
        """
        return self.update(other)

    def __or__(self, other):
        """Override the bitwise or operator ``|`` for union.
        """
        return self.union(other)

    def __repr__(self):
        return f'GetVars(free={self.free}, bound={self.bound})'

    def union(self, other):
        """
        >>> from sympy.abc import a, b, c, x, y, z
        >>> u = GetVars(free={a, b}, bound= {c})
        >>> v = GetVars(free={a, x}, bound={y, z})
        >>> _u = u
        >>> _v = v
        >>> w = u.union(v)
        >>> w == GetVars(free={a, b, x}, bound={c, y, z})
        True
        >>> u is _u and v is _v
        True
        """
        return self.__class__(free=self.free | other.free,
                              bound=self.bound | other.bound)

    def update(self, other):
        self.free = self.free | other.free
        self.bound = self.bound | other.bound
        return self
