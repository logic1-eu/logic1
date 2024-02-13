from __future__ import annotations

from typing import final, Optional

from ..support.decorators import classproperty

from .boolean import BooleanFormula


@final
class _T(BooleanFormula):
    """The constant Formula that is always true.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _T to be a
    subclass itself.

    >>> _T() is _T()
    True
    """
    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`_F`, which implements
        the dual operator :math:`\bot` or :math:`\top`.
        """
        return _F

    @classproperty
    def func(cls):
        """A class property yielding the class :class:`_T` itself.
        """
        return cls

    _instance: Optional[_T] = None

    def __init__(self) -> None:
        self.args = ()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'T'


T = _T()
"""Support use as a constant without parentheses.
"""


@final
class _F(BooleanFormula):
    """The constant Formula that is always false.

    This is a quite basic implementation of a singleton class. It does not
    support subclassing. We do not use a module because we need _F to be a
    subclass itself.

    >>> _F() is _F()
    True
    """
    @classproperty
    def dual_func(cls):
        r"""A class property yielding the class :class:`_T`, which implements
        the dual operator :math:`\top` or :math:`\bot`.
        """
        return (lambda: _T)()

    @classproperty
    def func(cls):
        """A class property yielding the class :class:`_F` itself.
        """
        return cls

    _instance: Optional[_F] = None

    def __init__(self) -> None:
        self.args = ()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return 'F'


F = _F()
"""Support use as a constant without parentheses.
"""
