from typing import Any


class LatexExpr(str):
    ...


class LatexCall:

    def __call__(self, x: Any, combine_all: bool = False) -> LatexExpr:
        ...


class Latex(LatexCall):
    ...


latex: Latex
