
.. _api-RCF-redlog:

*Real Closed Fields*

**********************
Redlog Interface
**********************

.. automodule:: logic1.theories.RCF.redlog

  Wrapped Redlog Functions
  ------------------------

  .. autofunction:: cnf(f: RCF.types.Formula, bnfsm: bool = False, bnfsac: bool = True) -> RCF.types.Formula
  .. autofunction:: dnf(f: RCF.types.Formula, bnfsm: bool = False, bnfsac: bool = True) -> RCF.types.Formula
  .. autofunction:: gqe(f: RCF.types.Formula, generic: Generic = Generic.FULL) -> tuple[list[RCF.atomic.AtomicFormula], RCF.types.Formula]
  .. autofunction:: gsn(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], form: str = 'auto', bnfsm: bool = False, bnfsac: bool = True) -> RCF.types.Formula
  .. autofunction:: qe(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = []) -> RCF.types.Formula
  .. autofunction:: qea(f: RCF.types.Formula) -> list[tuple[RCF.types.Formula, list[str]]]
  .. autofunction:: simplify(f: RCF.types.Formula, assume: Iterable[RCF.atomic.AtomicFormula] = [], explode_always: bool = True, prefer_order: bool = True, prefer_weak: bool = False) -> RCF.types.Formula

  Parsing Redlog Formulas
  -----------------------
  .. autofunction:: to_logic1(s: str) -> RCF.types.Formula


  Accessing Interactive Redlog Help
  ---------------------------------
  .. autofunction:: help
