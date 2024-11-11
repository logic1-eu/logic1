from sage.rings.polynomial.term_order import TermOrder
from sage.rings.ring import Ring


class MPolynomialRing_base(Ring):

    def term_order(self) -> TermOrder:
        ...
