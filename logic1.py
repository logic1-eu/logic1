import sympy


class Formula:
    """A class for representing first-order formulas.

    Attributes:
    -----------
    func: a logic1 class
    args: a tuple of ``Formula``
    """

    def __and__(*args):
        """Hijack the bitwise and operator ``&`` for our logical And.

        >>> Ne(1, 0) & Ne(1 + 1, 0)
        And(Ne(1, 0), Ne(2, 0))
        """
        return And(*args)

    def __invert__(a):
        """Hijack the bitwise invert operator ``~`` for our logical Not.

        >>> ~ Eq(1,0)
        Not(Eq(1, 0))
        """
        return Not(a)

    def __lshift__(a1, a2):
        """Hijack the bitwise left shift operator ``>>`` for our logical
        Implies.

        >>> from sympy.abc import x, y
        >>> Gt(x + 1, y) << Ge(x, y)
        Implies(Ge(x, y), Gt(x + 1, y))
        """
        return Implies(a2, a1)

    def __or__(*args):
        """Hijack the bitwise or operator ``|`` for our logical And.

        >>> from sympy.abc import x, y
        >>> Eq(x, y) | Lt(x, y) | Gt(x, y)
        Or(Or(Eq(x, y), Lt(x, y)), Gt(x, y))
        """
        return Or(*args)

    def __rshift__(a1, a2):
        """Hijack the bitwise right shift operator ``<<`` for our logical
        Implies with reversed sides.

        >>> from sympy.abc import x, y
        >>> Ge(x, y) >> Gt(x + 1, y)
        Implies(Ge(x, y), Gt(x + 1, y))
        """
        return Implies(a1, a2)

    def __eq__(self, other):
        """Recursive equality of the formulas self and other.

        >>> e1 = Gt(1, 0)
        >>> e2 = Gt(1, 0)
        >>> e1 == e2
        True
        >>> e1 is e2
        False
        """
        return self.func == other.func and self.args == other.args

    def __init__(self, *args):
        """An initializer that always raises an exception.

        >>> Formula(">", 1, 0)
        Traceback (most recent call last):
        ...
        NotImplementedError

        This provides a hands-on implementation of an abstract class. It is
        inherited down the hierarchy. Only the the leaf classes, which
        correspond to logic operators should be instantiated.
        """
        raise NotImplementedError

    def __repr__(self):
        """Representation of the Formula suitable for use as an input.

        __str__() falls back to this unless explicitly implemented in a
        subclass.
        """
        repr = self.func.__name__
        repr += "("
        if self.args:
            repr += self.args[0].__repr__()
            for a in self.args[1:]:
                repr += ", " + a.__repr__()
        repr += ")"
        return repr

    def _repr_latex_(self):
        """A LaTeX representation of the formula as it is used within jupyter
        notebooks

        >> Eq(1, 0)._repr_latex()_
        '$\\displaystyle 1 = 0$'

        Subclasses have latex() methods yielding plain LaTeX without the
        surrounding $\\displaystyle ... $.
        """
        return "$\\displaystyle " + self.latex() + "$"

    def simplify(self, Theta=None):
        """Identity as a default implemenation of a simplifier for formulas.

        This should be overridden in the majority of the classes that
        are finally instantiated.
        """
        return self

    def sympy(self, **kwargs):
        """Provide a sympy representation of the Formula if possible.

        Subclasses that have no match in sympy can raise NotImplementedError.

        >>> from sympy.abc import x, y
        >>> e1 = Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> e1
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1)
        <class 'logic1.Equivalent'>
        >>> e1.sympy()
        Equivalent(Eq(x, y), Eq(x + 1, y + 1))
        >>> type(e1.sympy())
        Equivalent

        >>> e2 = Equivalent(Eq(x, y), Eq(y, x))
        >>> e2
        Equivalent(Eq(x, y), Eq(y, x))
        >>> e2.sympy()
        True

        >>> e3 = T
        >>> e3.sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError

        >>> e4 = All(x, Ex(y, Eq(x, y)))
        >>> e4.sympy()
        Traceback (most recent call last):
        ...
        NotImplementedError
        """
        return self._sympy_func(*(a.sympy(**kwargs) for a in self.args))


class QuantifiedFormula(Formula):

    _latex_precedence = 99
    _latex_symbol_spacing = "\\,"

    is_atom = False
    is_boolean = False
    is_quantified = True

    @property
    def variable(self):
        """The variable of the quantifier.

        >>> from sympy.abc import x, y
        >>> e1 = All(x, Ex(y, Eq(x, y)))
        >>> e1.variable
        x
        """
        return self.args[0]

    @property
    def matrix(self):
        """The subformula in the scope of the QuantifiedFormula.

        >>> from sympy.abc import x, y
        >>> e1 = All(x, Ex(y, Eq(x, y)))
        >>> e1.matrix
        Ex(y, Eq(x, y))
        """
        return self.args[1]

    def latex(self):
        r"""A LaTeX representation of the QuantifiedFormula.

        >>> from sympy.abc import x, y
        >>> All(x, Ex(y, Or(Eq(y, x), Lt(x, y)))).latex()
        '\\forall x \\,\\exists y \\, (y = x \\, \\vee \\, x < y)'
        """
        self_latex = self._latex_symbol
        self_latex += " " + str(self.args[0])
        self_latex += " " + self._latex_symbol_spacing
        if not self.args[1].is_quantified:
            self_latex += " ("
        self_latex += self.args[1].latex()
        if not self.args[1].is_quantified:
            self_latex += ")"
        return self_latex

    def simplify(self, Theta=None):
        self.func(self.variable, self.matrix.simplify)

    def sympy(self, *args, **kwargs):
        print(f"sympy representation of {type(self)} is not available.")
        raise NotImplementedError


class Ex(QuantifiedFormula):
    """
    >>> from sympy.abc import x
    >>> Ex(x, Eq(x, 1))
    Ex(x, Eq(x, 1))
    """
    _latex_symbol = "\\exists"

    def __init__(self, variable, matrix):
        self.func = Ex
        self.args = (variable, matrix)


class All(QuantifiedFormula):
    """
    >>> from sympy.abc import x, y
    >>> All(x, All(y, Eq((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    All(x, All(y, Eq((x + y)**2 + 1, x**2 + 2*x*y + y**2)))
    """
    _latex_symbol = "\\forall"

    def __init__(self, variable, matrix):
        self.func = All
        self.args = (variable, matrix)


class BooleanFormula(Formula):

    _latex_symbol_spacing = "\\,"

    is_atom = False
    is_boolean = True
    is_quantified = False

    def latex(self):
        def latex_in_parens(outer, inner):
            inner_Latex = inner.latex()
            if outer._latex_precedence >= inner._latex_precedence:
                inner_Latex = "(" + inner_Latex + ")"
            return inner_Latex

        if self._latex_style == "constant":
            return self._latex_symbol
        if self._latex_style == "prefix":
            self_latex = self._latex_symbol
            self_latex += " " + self._latex_symbol_spacing
            self_latex += " " + latex_in_parens(self, self.args[0])
            return self_latex
        if self._latex_style == "infix":
            self_latex = latex_in_parens(self, self.args[0])
            for a in self.args[1:]:
                self_latex += " " + self._latex_symbol_spacing
                self_latex += " " + self._latex_symbol
                self_latex += " " + self._latex_symbol_spacing
                self_latex += " " + latex_in_parens(self, a)
            return self_latex


class Equivalent(BooleanFormula):

    _latex_style = "infix"
    _latex_symbol = "\\longleftrightarrow"
    _latex_precedence = 10
    _sympy_func = sympy.Equivalent

    @property
    def lhs(self):
        """The left-hand side of the Equivalence."""
        return self.args[0]

    @property
    def rhs(self):
        """The right-hand side of the Equivalence."""
        return self.args[1]

    def __init__(self, lhs, rhs):
        self.func = Equivalent
        self.args = (lhs, rhs)

    def simplify(self, Theta=None):
        """Recursively simplify the Equivalence.

        >>> from sympy.abc import x, y
        >>> e1 = Equivalent(Not(Eq(x, y)), F)
        >>> e1.simplify()
        Eq(x, y)
        """
        lhs = self.lhs.simplify(Theta=Theta)
        rhs = self.rhs.simplify(Theta=Theta)
        if lhs is T:
            return rhs
        if rhs is T:
            return lhs
        if lhs is F:
            if rhs.func is Not:
                return rhs.arg
            return Not(rhs)
        if rhs is F:
            if lhs.func is Not:
                return lhs.arg
            return Not(lhs)
        if lhs == rhs:
            return True
        return Equivalent(lhs, rhs)


class Implies(BooleanFormula):

    _latex_style = "infix"
    _latex_symbol = "\\longrightarrow"
    _latex_precedence = 10

    _sympy_func = sympy.Implies

    @property
    def lhs(self):
        """The left-hand side of the Implies."""
        return self.args[0]

    @property
    def rhs(self):
        """The right-hand side of the Implies."""
        return self.args[1]

    def __init__(self, lhs, rhs):
        self.func = Implies
        self.args = (lhs, rhs)

    def simplify(self, Theta=None):
        if self.rhs is T:
            return self.lhs
        lhs_simplify = self.lhs.simplify(Theta=Theta)
        if lhs_simplify is F:
            return T
        rhs_simplify = self.rhs.simplify(Theta=Theta)
        if rhs_simplify is T:
            return T
        if lhs_simplify is T:
            return rhs_simplify
        if rhs_simplify is F:
            return involutive_not(lhs_simplify)
        assert not isinstance(lhs_simplify, TruthValue)
        assert not isinstance(rhs_simplify, TruthValue)
        if lhs_simplify == rhs_simplify:
            return T
        return Implies(lhs_simplify, rhs_simplify)


class AndOr(BooleanFormula):
    _latex_style = "infix"
    _latex_precedence = 50

    def simplify(self, Theta=None):
        return _simplify_gAnd(self)


def _simplify_gAnd(f):
    """Simplify a ``generic And,`` which can be one of And, Or.

    >>> from sympy.abc import x
    >>> _simplify_gAnd(And(Ne(x, 0), T, Ne(x, 0), And(Ne(x, 1), Ne(x, 2))))
    And(Ne(x, 0), Ne(x, 1), Ne(x, 2))
    >>> _simplify_gAnd(Or(Ge(x, 0), Or(Ge(x, 1), Ge(x, 2)), And(Ge(x, 1), Ge(x, 2))))
    Or(Ge(x, 0), Ge(x, 1), Ge(x, 2), And(Ge(x, 1), Ge(x, 2)))
    """
    if f.func is And:
        gAnd = And
        gT = T
        gF = F
    else:
        assert f.func is Or
        gAnd = Or
        gT = F
        gF = T
    simplified_args = []
    for arg in f.args:
        arg_simplify = arg.simplify()
        if arg_simplify is gF:
            return gF
        if arg_simplify is gT:
            continue
        if arg_simplify in simplified_args:
            continue
        if arg_simplify.func is gAnd:
            simplified_args.extend(arg_simplify.args)
        else:
            simplified_args.append(arg_simplify)
    if not simplified_args:
        return gT
    return gAnd(*simplified_args)


class And(AndOr):
    """Constructor for conjunctions of Formulas.

    >>> And()
    T
    >>> And(Ne(1, 0))
    Ne(1, 0)
    >>> And(Ne(1, 0), Ne(2, 0), Ne(3, 0))
    And(Ne(1, 0), Ne(2, 0), Ne(3, 0))
    """
    _latex_symbol = "\\wedge"
    _sympy_func = sympy.And

    def __new__(cls, *args):
        if not args:
            return T
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args):
        self.func = And
        self.args = args


class Or(AndOr):
    """Constructor for disjunctions of Formulas.

    >>> Or()
    F
    >>> Or(Eq(1, 0))
    Eq(1, 0)
    >>> Or(Eq(1, 0), Eq(2, 0), Eq(3, 0))
    Or(Eq(1, 0), Eq(2, 0), Eq(3, 0))
    """
    _latex_symbol = "\\vee"
    _sympy_func = sympy.Or

    def __new__(cls, *args):
        if not args:
            return F
        if len(args) == 1:
            return args[0]
        return super().__new__(cls)

    def __init__(self, *args):
        self.func = Or
        self.args = args


class Not(BooleanFormula):

    _latex_style = "prefix"
    _latex_symbol = "\\neg"
    _latex_precedence = 90

    _sympy_func = sympy.Not

    @property
    def arg(self):
        """The one argument of the Not."""
        return self.args[0]

    def __init__(self, arg):
        self.func = Not
        self.args = (arg, )

    def simplify(self, Theta=None):
        arg_simplify = self.arg.simplify(Theta=Theta)
        if arg_simplify is T:
            return F
        if arg_simplify is F:
            return T
        return involutive_not(arg_simplify)


def involutive_not(arg: Formula):
    """Construct a formula equivalent Not(arg) using the involutive law if
    applicable.

    >>> involutive_not(Ne(1, 0))
    Not(Ne(1, 0))
    >>> involutive_not(Not(Eq(1, 0)))
    Eq(1, 0)
    >>> involutive_not(T)
    Not(T)
    """
    if arg.func is Not:
        return arg.arg
    return Not(arg)


class TruthValue(BooleanFormula):

    _latex_style = "constant"
    _latex_precedence = 99

    def sympy(self):
        raise NotImplementedError


class _T(TruthValue):
    """The constant Formula that is always true.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _T to be a
    subclass itself.
    """
    _latex_symbol = "\\top"

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.func = _T
        self.args = ()

    def __repr__(self):
        return "T"


T = _T()


class _F(TruthValue):
    """The constant Formula that is always false.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _F to be a
    subclass itself.
    """
    _latex_symbol = "\\bot"

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.func = _F
        self.args = ()

    def __repr__(self):
        return "F"


F = _F()


class AtomicFormula(BooleanFormula):

    _latex_symbol_spacing = ""
    _latex_precedence = 99

    is_atom = True
    is_boolean = False
    is_quantified = False

    # Override Formula.sympy() to prevent recursion into terms
    def sympy(self, **kwargs):
        return self._sympy_func(*self.args, **kwargs)


class BinaryAtomicFormula(AtomicFormula):

    @property
    def lhs(self):
        """The left-hand side of the BinaryAtomicFormula."""
        return self.args[0]

    @property
    def rhs(self):
        """The right-hand side of the BinaryAtomicFormula."""
        return self.args[1]

    # Override BooleanFormula.latex() to prevent recursion into terms
    def latex(self):
        self_latex = sympy.latex(self.lhs)
        self_latex += " " + self._latex_symbol_spacing
        self_latex += self._latex_symbol
        self_latex += " " + self._latex_symbol_spacing
        self_latex += sympy.latex(self.rhs)
        return self_latex


class Eq(BinaryAtomicFormula):
    """
    >>> from sympy.abc import x
    >>> Eq(x, x)
    Eq(x, x)
    """
    _latex_symbol = "="
    _sympy_func = sympy.Eq

    def __init__(self, lhs, rhs):
        self.func = Eq
        self.args = (lhs, rhs)


class Ne(BinaryAtomicFormula):
    """
    >>> Ne(1, 0)
    Ne(1, 0)
    """
    _latex_symbol = "\\neq"
    _sympy_func = sympy.Ne

    def __init__(self, lhs, rhs):
        self.func = Ne
        self.args = (lhs, rhs)


class Ge(BinaryAtomicFormula):

    _latex_symbol = "\\geq"
    _sympy_func = sympy.Ge

    def __init__(self, lhs, rhs):
        self.func = Ge
        self.args = (lhs, rhs)


class Le(BinaryAtomicFormula):

    _latex_symbol = "\\leq"
    _sympy_func = sympy.Le

    def __init__(self, lhs, rhs):
        self.func = Le
        self.args = (lhs, rhs)


class Gt(BinaryAtomicFormula):

    _latex_symbol = ">"
    _sympy_func = sympy.Gt

    def __init__(self, lhs, rhs):
        self.func = Gt
        self.args = (lhs, rhs)


class Lt(BinaryAtomicFormula):

    _latex_symbol = "<"
    _sympy_func = sympy.Lt

    def __init__(self, lhs, rhs):
        self.func = Lt
        self.args = (lhs, rhs)
