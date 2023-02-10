from sympy import Symbol

_key = "R"
_index = 0
_stack = []


def push() -> None:
    """
    Push the counter for rename() and start from 1.
    """
    global _index, _stack
    _stack.append(_index)
    _index = 0


def pop() -> None:
    """
    Pop the counter for rename(). The current counter is lost.
    """
    global _index, _stack
    _index = _stack.pop()


def rename(var: Symbol) -> Symbol:
    """
    >>> from sympy.abc import x, y
    >>> rename(x)
    x_R1
    >>> rename(_)
    x_R2
    >>> rename(y)
    y_R3
    >>> rename(x)
    x_R4
    """
    global _index, _key
    _index += 1
    var_str = var.__str__()
    L = var_str.split("_" + _key)
    if len(L) == 2:
        var_str = L[0]
    return Symbol(f"{var_str}_{_key}{_index}")
