# mypy: strict_optional = False
from dataclasses import dataclass
import more_itertools
import logging

from abc import abstractmethod
from time import time
from typing import Any, Generic, Optional, TypeAlias, TypeVar

from ..firstorder import (
    All, And, AtomicFormula, _F, Formula, Not, Or, Prefix, _T)
from ..firstorder.formula import α, τ, χ, σ

Variable: TypeAlias = Any

π = TypeVar('π', bound='Pool')


class FoundT(Exception):
    pass


class Pool(list[tuple[list[χ], Formula[α, τ, χ, σ]]], Generic[α, τ, χ, σ]):

    def __init__(self, vars_: list[χ], f: Formula[α, τ, χ, σ]) -> None:
        self.push(vars_, f)

    @abstractmethod
    def push(self, vars_: list[χ], f: Formula[α, τ, χ, σ]) -> None:
        ...


class PoolOneExistential(Pool[α, τ, χ, σ]):

    def push(self, vars_: list[χ], f: Formula[α, τ, χ, σ]) -> None:
        logging.debug(f'res = {f}')
        if f is not _F():
            split_f = [*f.args] if f.op is Or else [f]
            self.extend([(vars_.copy(), mt) for mt in split_f])


class PoolOnePrimitive(Pool[α, τ, χ, σ]):

    def push(self, vars_: list[χ], f: Formula[α, τ, χ, σ]) -> None:
        logging.debug(f'res = {f}')
        dnf = self.dnf(f)
        if dnf is _T():
            raise FoundT
        if dnf is not _F():
            split_dnf = [*dnf.args] if dnf.op is Or else [dnf]
            self.extend([(vars_.copy(), mt) for mt in split_dnf])

    @abstractmethod
    def dnf(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        ...


@dataclass
class QuantifierElimination(Generic[α, τ, χ, σ, π]):

    blocks: Optional[Prefix[χ]] = None
    matrix: Optional[Formula[α, τ, χ, σ]] = None
    negated: Optional[bool] = None
    pool: Optional[π] = None
    finished: Optional[list[Formula[α, τ, χ, σ]]] = None

    def __repr__(self) -> str:
        return (f'QuantifierElimination(blocks={self.blocks!r}, '
                f'matrix={self.matrix!r}, '
                f'negated={self.negated!r}, '
                f'pool={self.pool!r}, '
                f'finished={self.finished!r})')

    def __str__(self) -> str:
        if self.blocks is not None:
            blocks = f'[{self.blocks}]'
        else:
            blocks = None
        if self.negated is None:
            read_as = ''
        elif self.negated is False:
            read_as = '  # read as Ex'
        else:
            assert self.negated is True
            read_as = '  # read as Not All'
        if self.pool is not None:
            _h1 = [f'({str(job[0])}, {str(job[1])})' for job in self.pool]
            _h = ',\n                '.join(_h1)
            pool = f'[{_h}]'
        else:
            pool = None
        if self.finished is not None:
            _h1 = [f'{str(f)}' for f in self.finished]
            _h = ',\n                '.join(_h1)
            finished = f'[{_h}]'
        else:
            finished = None
        return (f'{self.__class__} [\n'
                f'    blocks   = {blocks},\n'
                f'    matrix   = {self.matrix},\n'
                f'    negated  = {self.negated},{read_as}\n'
                f'    pool     = {str(pool)},\n'
                f'    finished = {finished}\n'
                f']')

    def collect_finished(self) -> None:
        self.matrix = Or(*self.finished)
        if self.negated:
            self.matrix = Not(self.matrix)
        self.negated = None
        self.pool = None
        self.finished = None
        logging.info(f'{self.collect_finished.__qualname__}: {self}')

    def select_and_pop(self, vars_: list[χ], f: Formula[α, τ, χ, σ]) -> χ:
        return vars_.pop()

    @abstractmethod
    def simplify(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        ...

    @abstractmethod
    def _Pool(self, vars_: list[χ], f: Formula[α, τ, χ, σ]) -> π:
        ...

    def pop_block(self) -> None:
        assert self.matrix, "no matrix"
        quantifier, vars_ = self.blocks.pop()
        matrix = self.matrix
        self.matrix = None
        if quantifier is All:
            self.negated = True
            matrix = Not(matrix)
        else:
            self.negated = False
        self.pool = self._Pool(vars_, self.simplify(matrix))
        self.finished = []
        logging.info(f'{self.pop_block.__qualname__}: {self}')

    def process_pool(self) -> None:
        while self.pool:
            vars_, f = self.pool.pop()
            v = self.select_and_pop(vars_, f)
            match f:
                case AtomicFormula():
                    if v in f.fvars():
                        result = self.simplify(self.qe1(v, f))
                    else:
                        result = f
                case And(args=args):
                    other_args, v_args = more_itertools.partition(
                        lambda arg: v in arg.fvars(), args)
                    f_v: Formula = And(*v_args)
                    if f_v is not _T():
                        f_v = self.qe1(v, f_v)
                    result = self.simplify(And(f_v, *other_args))
                case _:
                    assert False, f
            if vars_:
                self.pool.push(vars_, result)
            else:
                self.finished.append(result)
            logging.info(f'{self.process_pool.__qualname__}: {self}')

    def qe(self, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        # The following manipulation of a private property is dirty. There
        # seems to be no supported way to reset reference time.
        logging._startTime = time()  # type: ignore
        self.setup(f)
        while self.blocks:
            try:
                self.pop_block()
                self.process_pool()
            except FoundT:
                self.pool = None
                self.finished = [_T()]
            self.collect_finished()
        return self.simplify(self.matrix.to_nnf())

    @abstractmethod
    def qe1(self, v: χ, f: Formula[α, τ, χ, σ]) -> Formula[α, τ, χ, σ]:
        """Elimination of the existential quantifier from Ex(v, f).

        It is guaranteed that v occurs in f. :meth:`qe1` need not apply
        :meth:`simplify`. Its result  will go through :meth:`simplify` soon.
        """
        ...

    def setup(self, f: Formula[α, τ, χ, σ]) -> None:
        self.matrix, self.blocks = f.to_pnf().matrix()
        logging.info(f'{self.setup.__qualname__}: {self}')
