# mypy: strict_optional = False

import more_itertools
import logging

from abc import ABC, abstractmethod
from time import time
from typing import Any, Generic, Optional, TypeAlias, TypeVar

from ..firstorder import All, And, AtomicFormula, F, Formula, Not, Or, QuantifiedFormula, T

Variable: TypeAlias = Any

P = TypeVar('P', bound='Pool')


class FoundT(Exception):
    pass


class Pool(ABC, list[tuple[list[Variable], Formula]]):

    def __init__(self, vars_: list[Variable], f: Formula) -> None:
        self.push(vars_, f)

    @abstractmethod
    def push(self, vars_: list[Variable], f: Formula) -> None:
        ...


class PoolOneExistential(Pool):

    def push(self, vars_: list[Variable], f: Formula) -> None:
        logging.debug(f'res = {f}')
        if f is not F:
            split_f = [*f.args] if f.func is Or else [f]
            self.extend([(vars_.copy(), mt) for mt in split_f])


class PoolOnePrimitive(Pool):

    def push(self, vars_: list[Variable], f: Formula) -> None:
        logging.debug(f'res = {f}')
        dnf = self.dnf(f)
        if dnf is T:
            raise FoundT
        if dnf is not F:
            split_dnf = [*dnf.args] if dnf.func is Or else [dnf]
            self.extend([(vars_.copy(), mt) for mt in split_dnf])

    @abstractmethod
    def dnf(self, f: Formula) -> Formula:
        ...


class QuantifierElimination(ABC, Generic[P]):

    # Types
    Quantifier: TypeAlias = type[QuantifiedFormula]
    QuantifierBlock: TypeAlias = tuple[Quantifier, list[Variable]]
    Job: TypeAlias = tuple[list[Variable], Formula]

    # Properties
    blocks: Optional[list[QuantifierBlock]]
    matrix: Optional[Formula]
    negated: Optional[bool]
    pool: Optional[P]
    finished: Optional[list[Formula]]

    def __init__(self, blocks=None, matrix=None, negated=None, pool=None,
                 finished=None) -> None:
        self.blocks = blocks
        self.matrix = matrix
        self.negated = negated
        self.pool = pool
        self.finished = finished

    def __repr__(self):
        return (f'QuantifierElimination(blocks={self.blocks!r}, '
                f'matrix={self.matrix!r}, '
                f'negated={self.negated!r}, '
                f'pool={self.pool!r}, '
                f'finished={self.finished!r})')

    def __str__(self):
        if self.blocks is not None:
            _h = [q.__qualname__ + ' ' + str(v) for q, v in self.blocks]
            _h = '  '.join(_h)
            blocks = f'[{_h}]'
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
            _h = [f'({str(job[0])}, {str(job[1])})' for job in self.pool]
            _h = ',\n                '.join(_h)
            pool = f'[{_h}]'
        else:
            pool = None
        if self.finished is not None:
            _h = [f'{str(f)}' for f in self.finished]
            _h = ',\n                '.join(_h)
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

    def select_and_pop(self, vars_: list, f: Formula) -> Variable:
        return vars_.pop()

    @abstractmethod
    def simplify(self, f: Formula) -> Formula:
        ...

    @abstractmethod
    def pnf(self, f: Formula) -> Formula:
        ...

    @abstractmethod
    def _Pool(self, vars_: list[Variable], f: Formula) -> P:
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
        self.pool = self._Pool(vars_, matrix)
        self.finished = []
        logging.info(f'{self.pop_block.__qualname__}: {self}')

    def process_pool(self) -> None:
        while self.pool:
            vars_, f = self.pool.pop()
            v = self.select_and_pop(vars_, f)
            match f:
                case AtomicFormula():
                    if v in f.get_vars().free:
                        result = self.simplify(self.qe1(v, f))
                    else:
                        result = f
                case And(args=args):
                    other_args, v_args = more_itertools.partition(
                        lambda arg: v in arg.get_vars().free, args)
                    f_v: Formula = And(*v_args)
                    if f_v is not T:
                        f_v = self.qe1(v, f_v)
                    result = self.simplify(And(f_v, *other_args))
            if vars_:
                self.pool.push(vars_, result)
            else:
                self.finished.append(result)
            logging.info(f'{self.process_pool.__qualname__}: {self}')

    def qe(self, f: Formula) -> Formula:
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
                self.finished = [T]
            self.collect_finished()
        return self.simplify(self.matrix.to_nnf())

    @abstractmethod
    def qe1(self, v: Variable, f: Formula) -> Formula:
        """Elimination of the existential quantifier from Ex(v, f).

        It is guaranteed that v occurs in f. :meth:`qe1` need not apply
        :meth:`simplify`. Its result  will go through :meth:`simplify` soon.
        """
        ...

    def setup(self, f: Formula) -> None:
        f = self.pnf(f)
        blocks = []
        vars_ = []
        while isinstance(f, QuantifiedFormula):
            q = type(f)
            while isinstance(f, q):
                vars_.append(f.var)
                f = f.arg
            blocks.append((q, vars_))
            vars_ = []
        self.blocks = blocks
        self.matrix = f
        logging.info(f'{self.setup.__qualname__}: {self}')
