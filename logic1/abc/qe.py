# mypy: strict_optional = False

import logging

from abc import ABC, abstractmethod
from time import time
from typing import Any, Optional, TypeAlias

from ..firstorder import All, And, F, Formula, Not, Or, QuantifiedFormula, T

Variable: TypeAlias = Any


class FoundT(Exception):
    pass


class QuantifierElimination(ABC):

    class Pool(list[tuple[list[Variable], Formula]]):
        def __init__(self, vars_: list[Variable], f: Formula) -> None:
            self.push(vars_, f)

        def push(self, vars_: list[Variable], f: Formula) -> None:
            logging.debug(f'res = {f}')
            dnf = f.to_dnf()  # type: ignore
            if dnf is T:
                raise FoundT
            if dnf is not F:
                split_dnf = [*dnf.args] if dnf.func is Or else [dnf]
                self.extend([(vars_.copy(), mt) for mt in split_dnf])

    # Types
    Quantifier: TypeAlias = type[QuantifiedFormula]
    QuantifierBlock: TypeAlias = tuple[Quantifier, list[Variable]]
    Job: TypeAlias = tuple[list[Variable], Formula]

    # Properties
    blocks: Optional[list[QuantifierBlock]]
    matrix: Optional[Formula]
    negated: Optional[bool]
    pool: Optional[Pool]
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

    @classmethod
    def get_best(cls, vars_: list, f: Formula) -> Variable:
        return vars_.pop()

    @abstractmethod
    def pnf(self, f: Formula) -> Formula:
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
        logging.info(f'simplify({matrix!r}) == {self.simplify(matrix)!r}')
        matrix = self.simplify(matrix)
        self.pool = self.Pool(vars_, matrix)
        self.finished = []
        logging.info(f'{self.pop_block.__qualname__}: {self}')

    def process_pool(self) -> None:
        while self.pool:
            vars_, f = self.pool.pop()
            v = self.get_best(vars_, f)
            f_v, f_other = self._split_And(f, v)
            if f_v is T:
                result = f_other
            else:
                result = self._join_And(self.qe1p(v, f_v), f_other)
            if vars_:
                result = self.simplify(result)
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
    def qe1p(self, v: Variable, f: Formula) -> Formula:
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

    # Some static helpers
    @staticmethod
    @abstractmethod
    def is_valid_atom(f: Formula) -> bool:
        ...

    @staticmethod
    def _join_And(f1: Formula, f2: Formula) -> Formula:
        args1 = f1.args if f1.func is And else (f1,)
        args2 = f2.args if f2.func is And else (f2,)
        return And(*args1, *args2)

    @abstractmethod
    def simplify(cls, f: Formula) -> Formula:
        ...

    @classmethod
    def _split_And(cls, f: Formula, v: Variable) -> tuple[Formula, Formula]:
        v_atoms = []
        other_atoms = []
        args = f.args if f.func is And else (f,)
        for atom in args:
            assert cls.is_valid_atom(atom), f'bad atom {atom} - {type(atom)}'
            if v in atom.get_vars().free:
                v_atoms.append(atom)
            else:
                other_atoms.append(atom)
        return And(*v_atoms), And(*other_atoms)
