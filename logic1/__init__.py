__version__ = 0.1

___author___ = 'Nicolas Faroß, Thomas Sturm'
___contact___ = 'https://logic1.eu/'
___copyright__ = 'Copyright 2023, N. Faroß, T. Sturm, Germany'
___license__ = 'GNU GENERAL PUBLIC LICENSE Version 3'
___status__ = 'Prototype'

from . import firstorder

from .firstorder import (Formula, AtomicFormula, Term, Variable,  # noqa
                         BooleanFormula, Equivalent, Implies, And, Or, Not,
                         _T, T, _F, F, QuantifiedFormula, Ex, All, pnf)

from . import theories

from .theories import RCF, Sets  # noqa

__all__ = firstorder.__all__ + theories.__all__
