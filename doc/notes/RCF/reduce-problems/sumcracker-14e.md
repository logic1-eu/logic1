# sumcracker-14e: unresolved Reduce/CSL crash

Recorded on 2026-11-06. Experiments below were performed on 2026-11-05.
This note preserves the investigation for later continuation, especially with
a newer Reduce build. It is separate from the published benchmark tables.

## Status and next step

**The defect has not been identified, its cause is not understood, and there
is no verified fix.** Several failures were reproduced and invalid native
runtime state was captured. Successful runs after changing the surrounding
program establish neither correctness of a workaround nor the origin of the
failure. Rare failures can disappear when allocation or execution changes.

The latest plan, proposed by the user, takes priority over further debugging:

1. Reproduce the error once more with the existing March 2026 snapshot and
   the unchanged saved failing program.
2. The user downloads a recent Reduce nightly, preferably beside the old
   installation. Repeat the same test with the newer executable.
3. Record both versions and outcomes. If the newer build succeeds, establish
   reproducibility before concluding that the issue has been resolved;
   investigate relevant changes if an explanation is needed.

**That final old-build rerun never started.** It was interrupted after
read-only inspection. No nightly was downloaded or tested in this
investigation. Do not confuse this pending comparison with earlier runs.

The user has experience with RLISP and requires discussion of proposed
RLISP code **before modifying `logic1/theories/RCF/redlog.py`**. The crash
investigation left that file and `test_redlog.py` unchanged. No experimental
wrapper was installed and no diagnostic timings replaced benchmark values.
Documentation work on 2026-11-06 does not resume the computations.

## Input, environment, and provenance

The local benchmark name is **sumcracker-14e**, without an additional hyphen
before the `e`. Preserve its exact source identity:

- Registry: `benchmarks/theories/RCF/smtlib/smtlib_inputs.py`.
- Original path below the SMT-LIB QF_NRA archive:
  `20220314-Uncu/Rational_Function_Proof_Calls_With_Disjunctive_Denominator_Simplifications/DDC_ProveIneq_JSC_SUMCracker_14e.smt2`.
- Release: QF_NRA 2025.08.04, [archive DOI](https://doi.org/10.5281/zenodo.16740866),
  from [SMT-LIB](https://smt-lib.org/).
- Source SHA-256:
  `cd675da749f62046300f1f84128c27ddd99048493219769ae6352e5cd183df9a`.
- Attribution: Ali K. Uncu, Matthew England, and James H. Davenport;
  SUMCracker-ProveInequality generator by Manuel Kauers, translated through
  Maple's SMTLIB package. Source generation date: 2022-02-03.
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Input: 54 atom occurrences, five variables, V1 through V5. The saved
  translation preserves these names, expands polynomial expressions, and
  moves comparison right-hand sides to zero. It applies no DNF simplification.
- Successful reference result: **34,816 atom occurrences**. This is a size
  check, not a proof of equivalence or a guarantee of identical formulas.

The original application is a CAD call associated with In[14], page 12, of
M. Kauers, *SumCracker: A Package for Manipulating Symbolic Sums and Related
Objects*, Journal of Symbolic Computation 41(9), 2006, pp. 1039–1057.
Clearing rational denominators introduces disjunctions of sign conditions.

The diagnostic configuration was:

| Item | Recorded value |
| --- | --- |
| Machine | Apple Silicon, macOS 27.2, arm64 |
| Reduce launcher | `/Applications/Reduce-snapshot/csl/redcsl` |
| Native executable | `/Applications/Reduce-snapshot/csl/reduce.app/Contents/MacOS/reduce` |
| Reduce banner | CSL revision 7327, 7-Mar-2026 |
| Redlog banner | Revision 7176, 2025-09-24 |
| Native binary UUID | `209DC5E3-8E4D-3F1B-B9D9-5256303443F9` |
| BNF settings | `rlbnfsac` on and `rlbnfsm` on, unless explicitly stated otherwise |
| Diagnostic guard | 600 seconds per large computation |
| Python environment | `/Users/sturm/miniforge3/envs/logic1_dev/bin/python` |
| Logic1 backend | Sage for formula export and actual Python-wrapper calls |

Direct Reduce replays do not execute Sage or Flint arithmetic in their timed
interval. No Flint validation was performed for this crash investigation.
Fresh interactive shells selected only the March snapshot on 2026-11-05,
but an existing agent process retained an older PATH. Always verify the
actual executable and its banner; do not rely on shell configuration alone.

The repository was on `feature/native-dnf`, with HEAD
`72152cd2c5f0cea143da2e556eb9213d0ee5ca1b`. The investigated `redlog.py` had
SHA-256 `534a5a630ce0df30cf6bbd62acf027396f230ff57ac857abc06b020eafc2c276`.
Uncommitted benchmark changes existed and must be preserved.

## How the issue arose; distinguish the earlier interface defect

An earlier problem with gj2007 was identified and fixed before this crash
investigation. The old interface assembled a complete Python-expression
string in Lisp, causing excessive copying and exceeding a CSL string/vector
allocation limit. Missing result markers could then cause Python to evaluate
part of the Reduce banner, yielding the misleading `2026 ...` syntax error.
The corrected interface prints the expression in pieces, checks process
status and both markers, and reports captured output on failure. That change
was committed on `develop` and merged into `feature/native-dnf`.

The current sumcracker problem persists with that corrected interface.
Failures also occur without calling the converters or reconstructing a
Python formula. It is therefore not explained by the earlier conversion
defect. An earlier CSL `newcslgc.cpp:266` abort on const1 disappeared after
upgrading from August 2024 to March 2026; it has not been shown to share the
cause of this sumcracker failure. The possible role of a macOS upgrade was
raised, but no operating-system comparison established it.

The formal 600-second benchmark rerun used Sage and two concurrent workers.
For sumcracker-14e it recorded:

| Method | Result atoms | Seconds or outcome |
| --- | ---: | ---: |
| Redlog off | 34816 | 44.411 |
| Redlog on | unknown | SIGSEGV; exit code 1 |
| Cartesian NONE | 34816 | 15.041 |
| Cartesian FINAL | 34816 | 50.492 |
| Cartesian AFTER_STEP | 34816 | 50.250 |
| Cartesian DURING_STEP | 34816 | 51.529 |

The same rerun had insufficient-memory errors on all-diff2 and a stack
overflow on gr2006. Those are separate observations, not demonstrated
instances of the same defect. Nothing in the run established concurrency
as their cause. Earlier limits and measurements remain in the SMT-LIB
directory's `NOTES.md` and
[dnf_results.md](../../../../benchmarks/theories/RCF/smtlib/dnf_results.md).

The user then ran [sumcracker-14e.red](sumcracker-14e.red)
successfully outside Python, with `on backtrace, gc;`, and supplied its
log, [sumcracker-14e.out](sumcracker-14e.out). It reported 34,816 atoms,
98.987 seconds computation plus 2.267 seconds GC CPU time. These CPU times
are not directly comparable to the wall times below. A GC report showed
24.0 MB in a 2048.0 MB heap; this is neither peak
resident memory nor proof that memory management is correct.

## Diagnostics and I/O matrix

All thirteen trials below used the March snapshot, a fresh process, and
both BNF switches enabled. They ran sequentially. Times are diagnostic wall
times, not replacement benchmark-helper measurements. No trial reached its
600-second guard.

Here **diagnostics** means `backtrace` and `gc`, varied together. It does
not mean the BNF settings called “Redlog on/off” in the benchmark tables.
For standalone “off”, both diagnostic switches were explicitly disabled.
For wrapper “off”, production startup defaults were retained.

| Trial | Program | Diagnostics | stdin | stdout/stderr | Seconds | Outcome |
| --- | --- | --- | --- | --- | ---: | --- |
| 01 | Standalone | on | file | merged file | 101.590 | 34816 atoms |
| 02 | Standalone | off | file | merged file | 102.368 | 34816 atoms |
| 03 | Standalone | off | pipe | merged file | 102.463 | 34816 atoms |
| 04 | Standalone | on | pipe | merged file | 102.640 | 34816 atoms |
| 05 | Standalone | on | pipe | separate pipes | 102.928 | 34816 atoms |
| 06 | Standalone | off | pipe | separate pipes | 103.107 | 34816 atoms |
| 07 | Python wrapper | defaults | pipe | separate pipes | 76.783 | SIGSEGV |
| 08 | Python wrapper | on | pipe | separate pipes | 94.277 | SIGBUS |
| 09 | Python wrapper | on | pipe | merged file | 93.869 | SIGBUS |
| 10 | Python wrapper | defaults | pipe | merged file | 76.133 | SIGSEGV |
| 11 | Exact generated command, no Logic1 | defaults | pipe | separate pipes | 75.831 | SIGSEGV |
| 12 | DNF assignment before conversion | defaults | pipe | separate pipes | 93.558 | SIGBUS |
| 13 | Named input, retained enclosing blocks | defaults | pipe | separate pipes | 93.001 | SIGSEGV |

Trials 12 and 13 printed `BEGIN_DNF` but not `DNF_DONE`. These instrumented
variants fail during the DNF stage. Instrumentation changes the program;
this does not by itself pinpoint the failure stage of every earlier trial.
None of the failed wrapper calls reached Python's `_eval` reconstruction.

Important qualifications:

- Bash `&>` redirects **stdout and stderr**, not stdin. Input redirection
  is separate. Redirecting the outer Python process does not change the
  pipes that Python gives Reduce; the experiment varied Reduce's streams.
- Changing trial 02 to 03 also replaced a Bash-exec launcher with a direct
  executable launch. It was not a perfectly isolated stdin-only change.
- The switches and output destination did not remove the wrapper failure
  in these trials. This does not prove that they cannot affect a rare bug.
- A shell can repeat the whole failing command in stderr, including marker
  literals. Trial 09 initially appeared to contain an end marker for this
  reason. It did not complete successfully. Check runtime output, not just
  a substring anywhere in merged output.
- Turning on `gc` controls reporting, not whether collection happens.
  `backtrace` may help diagnose a native signal but cannot guarantee a dump
  or recovery from `SIGSEGV` or `SIGBUS`.

## Removing definitions, converter calls, and enclosing blocks

The next experiments retained the same input and BNF settings. The labels
refer to `/tmp/logic1-sumcracker-minimize-2026-11-05/`.

| Trial | Change | Seconds | Outcome |
| --- | --- | ---: | --- |
| 01-no-converter | Remove converter definitions and call; keep enclosing blocks | 101.986 | 34816 atoms |
| 02-flat-full | Keep definitions and conversion; remove enclosing blocks | unavailable | DNF and serialization completed |
| 03-no-converter-call | Keep definitions and blocks; never invoke converter | 93.654 | SIGSEGV before DNF_DONE |
| 04-flat-no-converter-call | As 03, but remove enclosing blocks | 102.053 | 34816 atoms |
| 05-blocked-repeat | Repeat exactly 03 | 93.527 | SIGSEGV before DNF_DONE |
| 06-flat-repeat | Repeat exactly 04 | 103.303 | 34816 atoms |

Trial 02 emitted `DNF_DONE`, 34,816 atoms, `SERIALIZATION_DONE`, and the final
marker, in 12,669,716 bytes of stdout. Its diagnostic runner then stalled in
quadratic regular-expression postprocessing of the long output line. The
runner was interrupted before saving exact process time and exit status.
Completion was recovered from the output; **time and exit status remain
unknown**. The runner was corrected to save process results before analysis
and use linear line filtering. Do not classify this runner defect as a
Reduce hang.

The most useful repeated pair is 03 versus 04. Both keep unused converter
definitions, neither calls a converter, and their complete sources differ
only by removing two enclosing algebraic blocks and a separating semicolon:

```diff
 endmodule;

-<< lisp << terpri(); prin2 start!* >>;
-<< on rlbnfsac; on rlbnfsm; diag_input := <unchanged formula>;
+lisp << terpri(); prin2 start!* >>;
+on rlbnfsac; on rlbnfsm; diag_input := <unchanged formula>;
 lisp << terpri(); prin2t "BEGIN_DNF"; flush() >>;
 diag_result := rldnf diag_input;
 lisp << terpri(); prin2t "DNF_DONE"; flush() >>;
 write "DNF_ATOMS=", rlatnum diag_result;
- >>;
-lisp prin2 end!* >>;
+lisp prin2 end!*;
 quit;
```

The changed outcome does not establish that the removed blocks are wrong.
The unused definitions also affect the outcome, despite never being called.
Allocation history or object lifetime could matter; neither was traced.
The successful variant is not a verified general production workaround.

## RLISP, Reduce, and Standard Lisp lessons

These points incorporate the user's corrections and the supplied manuals.
Use them when interpreting or changing the embedded program.

### Modes and sequencing

- `symbolic;` and `lisp;` are equivalent global mode changes. The mode
  remains symbolic until another mode change, such as `algebraic;`.
- `symbolic(<expr>)` or `lisp(<expr>)` applies symbolic translation and
  evaluation to that expression without changing the surrounding mode.
  `algebraic(<expr>)` is the dual local prefix. These are mode-directed
  syntax, not ordinary Lisp functions that first evaluate an argument in
  the surrounding mode.
- `module ... endmodule` enters symbolic mode for the module and restores
  the preceding mode afterwards. It is not a lexical namespace or a reason
  to assume subsequent code is symbolic. In the failing program, the outer
  blocks after `endmodule` are algebraic; only the prefixed marker
  expressions are symbolic.
- In symbolic code, `<< ... >>` translates to `progn`: evaluate forms in
  order and return the last value. It introduces no local bindings. `begin
  scalar ... end` translates through `prog` and has different binding and
  control-flow behavior.
- **A trailing semicolon matters:** `<< expr1; expr2 >>` returns the value
  of `expr2`; `<< expr1; expr2; >>` has a final empty statement and returns
  `nil`. In an algebraic block the empty statement can appear as
  `(aeval (quote nil))` in the translation.
- Work in symbolic mode wherever practical. Use an algebraic prefix only
  for the expression requiring it, or construct the appropriate quoted
  prefix data for an explicit evaluator call. Do not add `<< ... >>` just
  as visual grouping: identify why that sequencing call is needed.
- A single evaluation containing start marker, computation, conversion,
  and end marker has a specific purpose: an error skips the end marker.
  It also avoids top-level prompts between those operations. Independent
  top-level statements do not preserve that error boundary.

### Inspecting translation

Use `on defn;` to inspect generated Lisp without running the computation.
The default `dfprint1` recursively flattens `progn` when displaying forms,
so default output can hide the very nesting under investigation. These
statements preserved it in the tested snapshot:

```reduce
symbolic;
dfprint!* := 'print;
algebraic;
on defn;
% Paste the algebraic program being inspected here.
off defn;
```

Use symbolic mode for the inspected body when that is its intended mode.
An earlier attempt using `lisp dfprint!* := ...` did not install the hook as
intended; the explicit mode change above worked. `setk` in a translation
assigns an algebraic value, unlike a symbolic `setq` assignment.

Full translations of the saved failing and successful pair were captured.
Their input assignment and DNF assignment are **exactly identical Lisp
forms**. The surrounding programs are not: nesting differs, as do evaluator
wrappers around switch statements, the trailing empty statement, and the
final marker. For example, the failing final marker becomes
`(aeval (prin2 end!*))`, whereas the flat variant has `(prin2 end!*)`.
Top-level switch directives also receive special treatment under `defn`.
Thus the earlier description as “only different Lisp grouping” was too
strong. Some source differences occur after DNF evaluation, but the entire
block is read before it runs, so allocation can already differ.

### Evaluation and data representations

- `aeval` evaluates a Lisp object representing an algebraic expression.
  Do not supply exported infix text as though quoting it made it correct
  prefix data, or accidentally execute a Lisp call before passing its
  result to the algebraic evaluator.
- Redlog can return a pseudo-prefix `!*fof` representation through `aeval`.
  The symbolic converter `r2py_formula` expects full prefix form. The
  existing algebraic operator interface uses `revlis`/`reval` on its
  arguments; a direct symbolic call must preserve this conversion.
- A small direct call without `reval` produced no formula output because
  the converter did not recognize the internal representation. With
  `r2py_formula reval result`, the example printed `VV['x'] >= 0`.
  The same issue occurs inside GQE result lists and QEA guards. Small
  linear examples verified the corresponding conversions.
- Literal prefix input decoded from the translator's `list`/`quote`
  construction was compared with input evaluated through an algebraic
  prefix. Their full-prefix forms were structurally equal, with 54 atoms.
- The primer's historical cache description needs correction for the
  checked source: `aeval` is `reval1(u,nil)`; `aeval!*` explicitly clears
  `alglist!*` first. The primer describes the opposite distinction. Do not
  use that historical detail to explain the crash without source checking.

Other relevant lessons from the manuals: fluid bindings are dynamically
active and restored on exit; `progn` adds no bindings. Lisp `ERRORSET`
handles Lisp errors and unwinds bindings, but the Standard Lisp report
does not specify native-signal recovery. Function pointers must remain
valid throughout execution, including relocation. Their representation,
the heap, GC roots, and CSL environment vectors are implementation details.
Destructive list operations and shared tails require care; algebraic
expressions, standard forms, and standard quotients have distinct contracts.
These are background principles, not identified causes of this failure.

## Symbolic-mode prototypes

The following body was tested after the unchanged converter definitions.
`<formula>` below abbreviates the original full input:

```reduce
symbolic;
<<
    terpri();
    prin2 start!*;
    on rlbnfsac;
    on rlbnfsm;
    r2py_formula(reval algebraic(rldnf(<formula>)));
    prin2 end!*
>>;
quit;
```

Its one outer `progn` supplies the error and output boundary described
above. There is no trailing empty statement. Its full translation has
symbolic switch and converter calls, with algebraic evaluation only around
the service expression. A deliberate `rederr` in a small probe printed the
start marker but not the end marker, even though Reduce exited with code 0;
the existing incomplete-marker check therefore remained effective.

This code is a discussion candidate, **not a fix**. The large trials were:

| Variant | Seconds | Outcome |
| --- | ---: | --- |
| Separate symbolic statements; algebraic input prefix; stored input/result; explicit `aeval` | 600.015 | Timeout during DNF |
| One symbolic block; narrow algebraic service expression; direct symbolic converter | 78.899 | SIGSEGV |
| Literal prefix input; explicit `aeval`; one symbolic block | 79.830 | SIGBUS |

All used both BNF switches and the same snapshot. Some runs overlapped;
these times are diagnostic only. None produced a completed large result.
The first run did print `BEGIN_DNF`; its original anchored marker matcher
missed the preceding Reduce prompt. The corrected summary records that
phase while preserving the raw log and original record. In the SIGBUS run,
shell stderr repeated the source; those marker literals are not runtime
phase evidence. Mixed-mode input syntax is not required for a failure.

## Native debugging evidence and its limits

### Crashes in `negf`

macOS crash reports for initial trials 08, 09, and 12 place invalid indirect
calls in native `negf`, reached through `ofsf_at2ir`, `ofsf_smupdknowl`,
`ofsf_smwupdknowl`, `rl_smupdknowl`, and `cl_simpl`.
Fresh-process metadata gave `negf`'s literal environment as
`[negf dmode!* nreverse !:minus]`.

- Trials 08/09 intended `!:minus(-1)` through environment slot 3 and `qfn1`;
  the invalid branch target was `0x1700000000`.
- Trial 12 intended `nreverse(nil)` through slot 2; the target was
  `0x1700000001`.
- The object used as a function symbol had low tag 0, whereas a symbol
  requires tag 4. This locates invalid state at dispatch, not its origin.

### Two earlier LLDB captures in `gcdfd`

The repeated blocked reproducer also failed under LLDB in native `gcdfd`,
attempting to call `gcdfd1(3,2)`. The stack passed through `gcdf1`, `gcdf`,
`sfto_gcdf`, `sfto_dcontentf`, `ofsf_at2ir`, and the simplification path.
Fresh-process metadata gave `[gcdfd v dmode!* gcdfd1]` as the literal vector.
The second capture established:

| Item | Captured value |
| --- | --- |
| Active environment (`x24`) | `0x70650467d1`, vector tag 1 |
| Saved Lisp-stack environment | Same value, at `0x128025128` |
| Header at `0x70650467d0` | `0x7065046620`, low bits 0 instead of header tag 2 |
| Alleged function symbol from slot 3 | `0x70650467e0`, tag 0 instead of symbol tag 4 |
| Misread `qfn2` value | `0x650464c000000070` |
| Numeric arguments | Valid fixnums 3 and 2 |

The pointed-to words were consistent with list-cell data rather than a
valid vector. Allocation history was not captured. The active register
agreed with its saved root, so this was not merely a disagreement between
those two references. **The function symbol's current `qenv` was not
captured**, leaving an important comparison missing.

Disassembly showed no CSL allocation between loading this environment at
entry to that `gcdfd` invocation and the bad branch; the intervening external
call was `memset_pattern16`. This excludes a missed reload after a collection
*inside that particular invocation*, but not an earlier invalid reference,
object overwrite, or relocation problem.

### Final, incomplete capture of the symbolic prototype

One LLDB run of the direct symbolic prototype stopped in
`bhashtable_gethash+108`, called by `cl_simplat` and `cl_simplifyatom`.
The instruction `ldur x9,[x0,#0x4c]` attempted to read a function field with
`x0=0x7`, CSL's fixnum zero, giving fault address `0x53`. The captured
environment reference before that load was `0x707e06d6f1`.

The subsequent caller-symbol/current-`qenv` inspection **never executed**.
The debugger follow-up stalled in the tool/permission interaction and was
interrupted after 431.6 seconds. There is no completed final analysis report
in that run directory. This capture does not establish the same malformed
vector observed in `gcdfd`; it establishes another invalid dispatch operand.

The debugger supervisor had a 240-second deadline, but killing its process
group left the Reduce target stopped and orphaned. A subsequent process
check confirmed PID 14169 had PPID 1 and state T; it was then successfully
terminated with SIGKILL. All three symbolic trials also finished or were
killed by their guards. No diagnostic computation was intentionally left running.
On resumption, do not reuse old PIDs; they may have been reassigned.

Two permission interactions were dysfunctional from the user's perspective.
Do not mistake a stalled tool call for a long Reduce computation, or claim
that a requested debugger command ran without checking its recorded output.
If debugging resumes, prepare bounded captures and ensure cleanup explicitly
covers the inferior as well as the debugger. Preserve ASLR, as in the earlier
captures, unless a deliberate comparison requires changing it.

## Suspected layer and conditional debugging strategy

The user's layer distinction is useful:

| Layer | Current assessment, not a diagnosis |
| --- | --- |
| Python in `redlog.py` | Low suspicion: generated commands fail independently of Python/Logic1. |
| Embedded Reduce program | A reproducible trigger; no erroneous operation has been shown to explain the invalid state. |
| Redlog RLISP package | Still possible, e.g. an invalid destructive operation or violated internal precondition. |
| RLISP part of Reduce | Lower suspicion after failure with literal prefix input and explicit evaluation. |
| Standard Lisp part of Reduce | Still possible; internal operations can violate lower-level assumptions. |
| Lisp/compiler part of CSL | Serious possibility: generated code or preservation of live references. |
| C/C++ part of CSL | Leading suspicion: allocation, relocation, reference handling, or an earlier overwrite. |

Neither GC nor the compiler has been proven responsible. Different crash
sites have not been proven to have a common cause. The user correctly
emphasized that a rare failure disappearing after an edit is not a solution.
Do not resume by trying many wrapper rearrangements or unrelated examples.

If comparison with the nightly leaves the issue unresolved, the proposed
next native observation is to compare three references at the same stop:
the function symbol's `qenv`, the active native environment, and its saved
Lisp-stack root. Read memory without invoking Lisp printers or other target
code in the damaged process. Validate tags before following pointers.

For the *old, observed* `gcdfd/gcdf1` arm64 callsite only, the saved caller
environment was at `*sp`; its slot 8 was at tagged-vector offset `0x47` and
held the `gcdfd` symbol. That symbol's `qenv` was at offset `0xc`. The active
environment was in `x24`, saved at `x22+8`. Re-derive these offsets for any
other callsite, binary, or build; they are not portable debugger commands.

A symbol with a valid environment but a stale active reference would suggest
one class of preservation problem. An invalid environment already attached
to the symbol would require looking earlier. A source-matched diagnostic
build could then check where references first become invalid, including
before/after collection. A watchpoint on a raw address must account for
legitimate object movement. Instrumentation changing the failure is not
evidence of a fix. Verification should establish the defective mechanism,
its correction, and repeated success of the unchanged original reproducer.

## Sources and manuals

The user supplied these local references, which were read for the relevant
semantics:

- H. Melenk, *REDUCE Symbolic Mode Primer*:
  `/Users/sturm/Documents/Managed/BibDesk/Papers/manual/Melenk/94/primer.pdf`.
  Relevant sections: 2.2 (p. 3, mode changes/prefixes), 3.4 (pp. 17–18,
  evaluators and `defn`), 5.1 (pp. 26–27, algebraic values and `setk`),
  6.2 (p. 33, error handling), and 7.1 (p. 34, module mode restoration).
  The cache-behavior caveat is recorded above.
- Jed Marti, A. C. Hearn, M. L. Griss, and C. Griss,
  *The Standard Lisp Report*:
  `/Users/sturm/Documents/Managed/BibDesk/Papers/article/MartiHearn/80a/sl.pdf`.
  Relevant sections: 2.1 (pp. 5–6, function-pointer validity), 2.5
  (pp. 8–9, EXPR/FEXPR/MACRO), 3.6 (pp. 19–22, bindings), 3.7
  (p. 24, PROGN/PROG), 3.8 (p. 25, ERRORSET), 3.14 (pp. 39–41,
  EVAL/APPLY), 4 (p. 47, GC reporting), and 5 (p. 50, extended syntax).
  The report deliberately leaves implementation details unspecified.

The [Reduce SourceForge repository](https://sourceforge.net/p/reduce-algebra/code/)
contains CSL under `trunk/csl/cslbase` and Reduce/Redlog under `trunk/packages`.
Direct SVN retrieval worked when the web tree browser did not. Use sources
matching the executable, for example
[revision-7327 tags.h](https://svn.code.sf.net/p/reduce-algebra/code/!svn/bc/7327/trunk/csl/cslbase/tags.h).
Replace the basename in that URL to retrieve other files at the same revision.

Eight selected files were fetched at revision 7327 and matched the local
revision-7360 checkout after normalizing SVN keyword expansion:
`tags.h`, `newcslgc.cpp`, `newallocate.h`, `newallocate.cpp`, `ccomp.red`,
`compiler.lsp`, `bytes1.cpp`, and `eval1.cpp`. This validates those specific
files, not the entire checkout at `/Users/sturm/reduce-algebra-code`.
The installed binary had native function symbols but no full DWARF/dSYM
information or matching generated C++ sources available locally.

Useful source locations recorded during the investigation:

- `packages/rlisp/superv.red`: `dfprint1` flattens `progn` by default.
- `packages/alg/reval.red`: `aeval`, `aeval!*`, and `opfneval`/`revlis`.
- `packages/redlog/rl/rlami.red`: `rl_cleanup` and formula representations.
- `packages/poly/polrep.red`: `negf`.
- `packages/redlog/ofsf/ofsfsism.red`: `ofsf_at2ir`.
- `csl/cslbase/tags.h`: `Symbol_Head`, tags, vector headers, `qenv`, `qfn1/2`.
- `csl/cslbase/ccomp.red`: emitted environment loading, reloading, and
  `RealSave` of live references.
- `csl/cslbase/newcslgc.cpp`: `evacuate` (r7327 lines 562–647), global and
  Lisp-stack roots (654–669), pinned objects (686–738), copied objects
  (823–870), and collector phase order (1556–1565). Both symbol environment
  fields and vector slots are traversed; this alone does not prove that
  every reference is correctly handled at runtime.

## Resuming without the temporary directories

The appendix embeds the exact, repeatedly failing source so this note
survives deletion of `/tmp`. Preserve it byte for byte, including the long
input line, unused definitions, modes, and trailing semicolons. It is a
diagnostic fixture, not recommended RLISP style. It computes and counts the
DNF without calling the converters. Expected success markers are `DNF_DONE`,
`DNF_ATOMS=34816`, and the final marker. The known old-build failure prints
`BEGIN_DNF` and stops before `DNF_DONE`.
Require exit status 0, these markers in the stated order in runtime stdout,
and the expected atom count before recording success. Marker text repeated
by the shell in stderr does not satisfy this check.

From the repository root, the following Python code **only extracts** the
failing and successful control files and reconstructs the saved shell
launchers. It does not run Reduce:

```python
from pathlib import Path
import hashlib

note = Path('doc/notes/RCF/reduce-problems/sumcracker-14e.md').read_text()
begin = '<!-- BEGIN EXACT FAILING SOURCE -->\n```reduce\n'
end = '```\n<!-- END EXACT FAILING SOURCE -->'
failing = note.split(begin, 1)[1].split(end, 1)[0]
working = failing.replace('<< lisp << terpri(); prin2 start!* >>;\n<< on ',
                          'lisp << terpri(); prin2 start!* >>;\non ', 1)
working = working.replace(' >>;\nlisp prin2 end!* >>;',
                          'lisp prin2 end!*;', 1)
hashes = {
    'failing': '56b0b32631d0978b9dbe20da84e63ceb8ca9ec3a5ab2620f74609a989ab1a3ed',
    'working': '7c4ac2fcf9c2032523334bf1ffe32a015324b0fe8fa84cd750d6ad84ecea0e3b'}
destination = Path('/tmp/logic1-sumcracker-resume')
destination.mkdir(exist_ok=True)
for name, program in [('failing', failing), ('working', working)]:
    assert hashlib.sha256(program.encode()).hexdigest() == hashes[name]
    # Reconstruct this fixed, checked historical command, not arbitrary input.
    assert not any(c in program for c in ('$','`','\\'))
    (destination / f'{name}.red').write_text(program)
    command = 'echo "' + program.replace('"', r'\"') + '" | redcsl -w\n'
    (destination / f'{name}.sh').write_text(command)
```

The reconstructed `failing.sh` has SHA-256
`bc36b291a956ea9f9024ba4bd1b5d5330cfef8c3d01720c94e502a57c5e0d056`;
`working.sh` has
`4b699ee2102d1dcbe006fb839e6224d58a40e197bd06ed6d06e39fa8a9e03f83`.
The historical command uses an `echo`-to-Reduce pipe, with Python capturing
stdout and stderr separately. A direct `< failing.red` invocation changes
stdin from a pipe to a file; record it as a different launch condition.

For the version comparison, select the desired launcher by prepending its
directory to the child PATH. Record its resolved path, banner, native binary
hash/UUID, OS, source hash, options, wall time, exit status, signals, and
runtime markers. Run one computation at a time and keep the 600-second
guard. Keep both output streams, saving process results before parsing them.
A suitable execution pattern, after deliberately selecting the build, is:

```python
import json
import os
from pathlib import Path
import signal
import subprocess
import time

run_dir = Path('/tmp/logic1-sumcracker-resume/old-build-run-1')
run_dir.mkdir()  # Refuse to overwrite an earlier observation.
launcher_dir = '/Applications/Reduce-snapshot/csl'
env = dict(os.environ, PATH=launcher_dir + os.pathsep + os.environ['PATH'])
command = Path('/tmp/logic1-sumcracker-resume/failing.sh').read_text()
started = time.perf_counter()
p = subprocess.Popen(command, shell=True, env=env, stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE, start_new_session=True)
timed_out = False
try:
    stdout, stderr = p.communicate(timeout=600)
except subprocess.TimeoutExpired:
    timed_out = True
    os.killpg(p.pid, signal.SIGKILL)
    stdout, stderr = p.communicate()
seconds = time.perf_counter() - started
(run_dir / 'stdout.log').write_bytes(stdout)
(run_dir / 'stderr.log').write_bytes(stderr)
(run_dir / 'result.json').write_text(json.dumps({
    'launcher_dir': launcher_dir, 'wall_seconds': seconds,
    'returncode': p.returncode, 'timed_out': timed_out}, indent=2))
```

Use a new output directory and the nightly's launcher directory for the
second build. Do not overwrite the old installation if retaining it is
possible. This is the next comparison, not an instruction to run it merely
because the note has been opened. Discuss unexpected outcomes with the user
before expanding the experiment. A newer successful run is useful evidence;
it does not by itself identify which change fixed the defect.

## Additional local evidence

These directories existed when the note was written. They are temporary;
the essential findings and failing fixture are preserved above and below.

| Directory | Contents |
| --- | --- |
| `/tmp/logic1-sumcracker-io-2026-11-05/` | Thirteen-trial matrix, manifest, sources, per-run logs/results, `summary.json`, and `original-user.out` |
| `/tmp/logic1-sumcracker-minimize-2026-11-05/` | Six minimization runs, repeated pair, `reproducer/`, recovered full-output result, native evidence, version-pinned CSL sources |
| `/tmp/logic1-symbolic-redlog-2026-11-05/` | Full translations and JSON trees, literal input, symbolic prototypes, small conversion/input checks, three large trials, corrected `summary.json` |
| `/tmp/logic1-symbolic-redlog-2026-11-05/04-symbolic-direct-lldb/` | Incomplete final native capture: `capture.lldb`, `run_debugger.py`, `session.log`, and output streams; no completed final report |
| `/tmp/logic1-symbolic-wrapper-review-2026-11-05/` | Independent small mode, conversion, GQE/QEA, translation, and error-marker probes |
| `/tmp/logic1-primer-reading/` | Extracted manual text and selected rendered pages |
| `/tmp/logic1-smtlib-lower-600s-2026-11-05/` | Formal benchmark snapshot, source/input hashes, per-call records, prior/current table values |

Particularly useful files in the minimization directory are
`negf-callsite-analysis.json`, `crash-stacks.json`,
`lldb-first-crash-summary.json`, `lldb-second-crash-summary.json`,
`lldb-first.log`, `lldb-session-unrestricted.log`, and
`native-next-probes.md`. Their raw addresses belong to those process runs.
macOS reports used were `reduce-2026-11-05-151755.ips`,
`reduce-2026-11-05-151930.ips`, and `reduce-2026-11-05-152432.ips` under
`/Users/sturm/Library/Logs/DiagnosticReports/`.

The successful standalone input [sumcracker-14e.red](sumcracker-14e.red) is
**not** the failing fixture. Do not substitute it when testing the newer
build and then claim that the failure disappeared. Likewise, do not
regenerate the input from a changed Python exporter when comparing builds;
use the frozen source.

## Appendix: exact failing source

The following source is copied without reformatting. Its SHA-256 is
`56b0b32631d0978b9dbe20da84e63ceb8ca9ec3a5ab2620f74609a989ab1a3ed`.

<!-- BEGIN EXACT FAILING SOURCE -->
```reduce
module redlog;

% Print Python syntax in pieces. Whole-result strings can exceed CSL's
% allocation limit, and repeated string concatenation copies large prefixes.

off1 'output;
off1 'nat;

linelength(2**24);

global '(start!* end!*);

start!* := "889e0d7343405c079195e7b8903c8c9e";
end!* := "b0061974914468de549a2af8ced10316";

rl_set '(r);

algebraic operator vv;

procedure privv(v); <<
    prin2!* "VV['";
    prin2!* cadr v;
    prin2!* "']";
    nil
>>;

put('vv, 'prifn, 'privv);

operator r2py_qea;
procedure r2py_qea(l);
    % Print an algebraic-mode rlqea result as a Python list of answer pairs.
    begin scalar separator;
        prin2 "[";
        separator := "";
        for each pair in cdr l do <<
            prin2 separator;
            separator := ", ";
            prin2 "(";
            r2py_formula cadr pair;
            prin2 ", ";
            r2py_equation_list caddr pair;
            prin2 ")"
        >>;
        prin2 "]";
        return nil
    end;

procedure r2py_equation_list(l);
    % Print an algebraic list of equations as a Python list of strings.
    begin scalar separator;
        prin2 "[";
        separator := "";
        for each equation in cdr l do <<
            prin2 separator;
            separator := ", ";
            prin2 "'";
            prin2 ioto_smaprin equation;
            prin2 "'"
        >>;
        prin2 "]";
        return nil
    end;

operator r2py_formula;
procedure r2py_formula(f);
    % Print a Lisp Prefix Form formula as a Python expression.
    begin scalar map1, map2, map3, map4, op, argl, w, separator;
        map1 := '((true . "T") (false . "F"));
        map2 := '((equal . "==") (neq . "!=") (lessp . "<")
                  (greaterp . ">") (leq . "<=")  (geq . ">="));
        map3 := '((and . "And") (or . "Or") (impl . "Implies")
                  (equiv . "Equivalent") (not . "Not"));
        map4 := '((ex . "Ex") (all . "All"));
        op := if atom f then f else car f;
        if w := atsoc(op, map1) then <<
            prin2 cdr w;
            return nil
        >>;
        argl := cdr f;
        if op eq 'repl then <<
            op := 'impl;
            argl := reverse argl;
        >>;
        if w := atsoc(op, map2) then <<
            prin2 ioto_smaprin subsvv car argl;
            prin2 " ";
            prin2 cdr w;
            prin2 " ";
            prin2 ioto_smaprin subsvv cadr argl;
            return nil
        >>;
        if w := atsoc(op, map3) then <<
            prin2 cdr w;
            prin2 "(";
            separator := "";
            for each arg in argl do <<
                prin2 separator;
                separator := ", ";
                r2py_formula arg
            >>;
            prin2 ")";
            return nil
        >>;
        if w := atsoc(op, map4) then <<
            prin2 cdr w;
            prin2 "(";
            prin2 ioto_smaprin subsvv car argl;
            prin2 ", ";
            r2py_formula cadr argl;
            prin2 ")";
            return nil
        >>
    end;

procedure subsvv(f); <<
    f := numr simp f;
    for each v in kernels f do
        f := numr subf(f, {v . {'vv, v}});
    prepf f
>>;

operator r2py_gqe;
procedure r2py_gqe(l);
    % Print an algebraic-mode rlgqe result as (assumptions, formula).
    begin scalar atoms, formula, separator;
        atoms := cdadr l;
        formula := caddr l;
        prin2 "([";
        separator := "";
        for each at in atoms do <<
            prin2 separator;
            separator := ", ";
            r2py_formula at
        >>;
        prin2 "], ";
        r2py_formula formula;
        prin2 ")";
        return nil
    end;

endmodule;

<< lisp << terpri(); prin2 start!* >>;
<< on rlbnfsac; on rlbnfsm; diag_input := ((((-V1 - V2 + 1 < 0) and (-V1*V5 - V2*V5 + V3**2 + 2*V3*V4 + V4**2 - 6*V3 - 6*V4 + V5 + 9 <= 0)) or ((V1 + V2 - 1 < 0) and (V1*V5 + V2*V5 - V3**2 - 2*V3*V4 - V4**2 + 6*V3 + 6*V4 - V5 - 9 <= 0))) and (((((-V2 < 0) and (-V1 - 2*V2 + 1 < 0)) or ((V2 < 0) and (V1 + 2*V2 - 1 < 0))) and (-V1*V2*V5 - V1*V4**2 - 2*V2**2*V5 + V2*V3**2 + 4*V2*V3*V4 + 2*V2*V4**2 - 6*V2*V3 - 12*V2*V4 + V2*V5 + V4**2 + 9*V2 <= 0)) or ((((-V2 < 0) and (V1 + 2*V2 - 1 < 0)) or ((V2 < 0) and (-V1 - 2*V2 + 1 < 0))) and (V1*V2*V5 + V1*V4**2 + 2*V2**2*V5 - V2*V3**2 - 4*V2*V3*V4 - 2*V2*V4**2 + 6*V2*V3 + 12*V2*V4 - V2*V5 - V4**2 - 9*V2 <= 0))) and (((((-V1*V2 - V2**2 < 0) and (-2*V1 - 3*V2 + 1 < 0)) or ((V1*V2 + V2**2 < 0) and (2*V1 + 3*V2 - 1 < 0))) and (-2*V1**2*V2*V5 - 2*V1**2*V4**2 - 5*V1*V2**2*V5 + 2*V1*V2*V3**2 + 8*V1*V2*V3*V4 + 2*V1*V2*V4**2 - 3*V2**3*V5 + V2**2*V3**2 + 6*V2**2*V3*V4 + 3*V2**2*V4**2 - 12*V1*V2*V3 - 18*V1*V2*V4 + V1*V2*V5 + V1*V4**2 - 12*V2**2*V3 - 18*V2**2*V4 + V2**2*V5 + V2*V3**2 + 2*V2*V3*V4 + 2*V2*V4**2 + 9*V1*V2 + 9*V2**2 <= 0)) or ((((-V1*V2 - V2**2 < 0) and (2*V1 + 3*V2 - 1 < 0)) or ((V1*V2 + V2**2 < 0) and (-2*V1 - 3*V2 + 1 < 0))) and (2*V1**2*V2*V5 + 2*V1**2*V4**2 + 5*V1*V2**2*V5 - 2*V1*V2*V3**2 - 8*V1*V2*V3*V4 - 2*V1*V2*V4**2 + 3*V2**3*V5 - V2**2*V3**2 - 6*V2**2*V3*V4 - 3*V2**2*V4**2 + 12*V1*V2*V3 + 18*V1*V2*V4 - V1*V2*V5 - V1*V4**2 + 12*V2**2*V3 + 18*V2**2*V4 - V2**2*V5 - V2*V3**2 - 2*V2*V3*V4 - 2*V2*V4**2 - 9*V1*V2 - 9*V2**2 <= 0))) and (((((-V1**2*V2 - 3*V1*V2**2 - 2*V2**3 < 0) and (-3*V1 - 5*V2 + 1 < 0)) or ((V1**2*V2 + 3*V1*V2**2 + 2*V2**3 < 0) and (3*V1 + 5*V2 - 1 < 0))) and (-3*V1**3*V2*V5 - 3*V1**3*V4**2 - 14*V1**2*V2**2*V5 + 3*V1**2*V2*V3**2 + 12*V1**2*V2*V3*V4 - 4*V1**2*V2*V4**2 - 21*V1*V2**3*V5 + 8*V1*V2**2*V3**2 + 36*V1*V2**2*V3*V4 + 11*V1*V2**2*V4**2 - 10*V2**4*V5 + 3*V2**3*V3**2 + 20*V2**3*V3*V4 + 10*V2**3*V4**2 - 18*V1**2*V2*V3 - 30*V1**2*V2*V4 + V1**2*V2*V5 + V1**2*V4**2 - 54*V1*V2**2*V3 - 90*V1*V2**2*V4 + 3*V1*V2**2*V5 + 2*V1*V2*V3**2 + 6*V1*V2*V3*V4 + 8*V1*V2*V4**2 - 36*V2**3*V3 - 60*V2**3*V4 + 2*V2**3*V5 + 3*V2**2*V3**2 + 8*V2**2*V3*V4 + 8*V2**2*V4**2 + 9*V1**2*V2 + 27*V1*V2**2 + 18*V2**3 <= 0)) or ((((-V1**2*V2 - 3*V1*V2**2 - 2*V2**3 < 0) and (3*V1 + 5*V2 - 1 < 0)) or ((V1**2*V2 + 3*V1*V2**2 + 2*V2**3 < 0) and (-3*V1 - 5*V2 + 1 < 0))) and (3*V1**3*V2*V5 + 3*V1**3*V4**2 + 14*V1**2*V2**2*V5 - 3*V1**2*V2*V3**2 - 12*V1**2*V2*V3*V4 + 4*V1**2*V2*V4**2 + 21*V1*V2**3*V5 - 8*V1*V2**2*V3**2 - 36*V1*V2**2*V3*V4 - 11*V1*V2**2*V4**2 + 10*V2**4*V5 - 3*V2**3*V3**2 - 20*V2**3*V3*V4 - 10*V2**3*V4**2 + 18*V1**2*V2*V3 + 30*V1**2*V2*V4 - V1**2*V2*V5 - V1**2*V4**2 + 54*V1*V2**2*V3 + 90*V1*V2**2*V4 - 3*V1*V2**2*V5 - 2*V1*V2*V3**2 - 6*V1*V2*V3*V4 - 8*V1*V2*V4**2 + 36*V2**3*V3 + 60*V2**3*V4 - 2*V2**3*V5 - 3*V2**2*V3**2 - 8*V2**2*V3*V4 - 8*V2**2*V4**2 - 9*V1**2*V2 - 27*V1*V2**2 - 18*V2**3 <= 0))) and (((((-2*V1**3*V2 - 9*V1**2*V2**2 - 13*V1*V2**3 - 6*V2**4 < 0) and (-5*V1 - 8*V2 + 1 < 0)) or ((2*V1**3*V2 + 9*V1**2*V2**2 + 13*V1*V2**3 + 6*V2**4 < 0) and (5*V1 + 8*V2 - 1 < 0))) and (-10*V1**4*V2*V5 - 10*V1**4*V4**2 - 61*V1**3*V2**2*V5 + 10*V1**3*V2*V3**2 + 40*V1**3*V2*V3*V4 - 28*V1**3*V2*V4**2 - 137*V1**2*V2**3*V5 + 41*V1**2*V2**2*V3**2 + 178*V1**2*V2**2*V3*V4 + 17*V1**2*V2**2*V4**2 - 134*V1*V2**4*V5 + 48*V1*V2**3*V3**2 + 240*V1*V2**3*V3*V4 + 86*V1*V2**3*V4**2 - 48*V2**5*V5 + 14*V2**4*V3**2 + 96*V2**4*V3*V4 + 48*V2**4*V4**2 - 60*V1**3*V2*V3 - 96*V1**3*V2*V4 + 2*V1**3*V2*V5 + 2*V1**3*V4**2 - 270*V1**2*V2**2*V3 - 432*V1**2*V2**2*V4 + 9*V1**2*V2**2*V5 + 8*V1**2*V2*V3**2 + 24*V1**2*V2*V3*V4 + 28*V1**2*V2*V4**2 - 390*V1*V2**3*V3 - 624*V1*V2**3*V4 + 13*V1*V2**3*V5 + 24*V1*V2**2*V3**2 + 70*V1*V2**2*V3*V4 + 67*V1*V2**2*V4**2 - 180*V2**4*V3 - 288*V2**4*V4 + 6*V2**4*V5 + 17*V2**3*V3**2 + 48*V2**3*V3*V4 + 42*V2**3*V4**2 + 18*V1**3*V2 + 81*V1**2*V2**2 + 117*V1*V2**3 + 54*V2**4 <= 0)) or ((((-2*V1**3*V2 - 9*V1**2*V2**2 - 13*V1*V2**3 - 6*V2**4 < 0) and (5*V1 + 8*V2 - 1 < 0)) or ((2*V1**3*V2 + 9*V1**2*V2**2 + 13*V1*V2**3 + 6*V2**4 < 0) and (-5*V1 - 8*V2 + 1 < 0))) and (10*V1**4*V2*V5 + 10*V1**4*V4**2 + 61*V1**3*V2**2*V5 - 10*V1**3*V2*V3**2 - 40*V1**3*V2*V3*V4 + 28*V1**3*V2*V4**2 + 137*V1**2*V2**3*V5 - 41*V1**2*V2**2*V3**2 - 178*V1**2*V2**2*V3*V4 - 17*V1**2*V2**2*V4**2 + 134*V1*V2**4*V5 - 48*V1*V2**3*V3**2 - 240*V1*V2**3*V3*V4 - 86*V1*V2**3*V4**2 + 48*V2**5*V5 - 14*V2**4*V3**2 - 96*V2**4*V3*V4 - 48*V2**4*V4**2 + 60*V1**3*V2*V3 + 96*V1**3*V2*V4 - 2*V1**3*V2*V5 - 2*V1**3*V4**2 + 270*V1**2*V2**2*V3 + 432*V1**2*V2**2*V4 - 9*V1**2*V2**2*V5 - 8*V1**2*V2*V3**2 - 24*V1**2*V2*V3*V4 - 28*V1**2*V2*V4**2 + 390*V1*V2**3*V3 + 624*V1*V2**3*V4 - 13*V1*V2**3*V5 - 24*V1*V2**2*V3**2 - 70*V1*V2**2*V3*V4 - 67*V1*V2**2*V4**2 + 180*V2**4*V3 + 288*V2**4*V4 - 6*V2**4*V5 - 17*V2**3*V3**2 - 48*V2**3*V3*V4 - 42*V2**3*V4**2 - 18*V1**3*V2 - 81*V1**2*V2**2 - 117*V1*V2**3 - 54*V2**4 <= 0))) and (((((-6*V1**4*V2 - 37*V1**3*V2**2 - 84*V1**2*V2**3 - 83*V1*V2**4 - 30*V2**5 < 0) and (-8*V1 - 13*V2 + 1 < 0)) or ((6*V1**4*V2 + 37*V1**3*V2**2 + 84*V1**2*V2**3 + 83*V1*V2**4 + 30*V2**5 < 0) and (8*V1 + 13*V2 - 1 < 0))) and (48*V1**5*V2*V5 + 48*V1**5*V4**2 + 374*V1**4*V2**2*V5 - 48*V1**4*V2*V3**2 - 192*V1**4*V2*V3*V4 + 216*V1**4*V2*V4**2 + 1153*V1**3*V2**3*V5 - 278*V1**3*V2**2*V3**2 - 1180*V1**3*V2**2*V3*V4 + 147*V1**3*V2**2*V4**2 + 1756*V1**2*V2**4*V5 - 563*V1**2*V2**3*V3**2 - 2600*V1**2*V2**3*V3*V4 - 550*V1**2*V2**3*V4**2 + 1319*V1*V2**5*V5 - 456*V1*V2**4*V3**2 - 2412*V1*V2**4*V3*V4 - 929*V1*V2**4*V4**2 + 390*V2**6*V5 - 113*V2**5*V3**2 - 780*V2**5*V3*V4 - 390*V2**5*V4**2 + 288*V1**4*V2*V3 + 468*V1**4*V2*V4 - 6*V1**4*V2*V5 - 6*V1**4*V4**2 + 1776*V1**3*V2**2*V3 + 2886*V1**3*V2**2*V4 - 37*V1**3*V2**2*V5 - 42*V1**3*V2*V3**2 - 132*V1**3*V2*V3*V4 - 144*V1**3*V2*V4**2 + 4032*V1**2*V2**3*V3 + 6552*V1**2*V2**3*V4 - 84*V1**2*V2**3*V5 - 193*V1**2*V2**2*V3**2 - 600*V1**2*V2**2*V3*V4 - 566*V1**2*V2**2*V4**2 + 3984*V1*V2**4*V3 + 6474*V1*V2**4*V4 - 83*V1*V2**4*V5 - 288*V1*V2**3*V3**2 - 884*V1*V2**3*V3*V4 - 786*V1*V2**3*V4**2 + 1440*V2**5*V3 + 2340*V2**5*V4 - 30*V2**5*V5 - 139*V2**4*V3**2 - 420*V2**4*V3*V4 - 360*V2**4*V4**2 - 54*V1**4*V2 - 333*V1**3*V2**2 - 756*V1**2*V2**3 - 747*V1*V2**4 - 270*V2**5 < 0)) or ((((-6*V1**4*V2 - 37*V1**3*V2**2 - 84*V1**2*V2**3 - 83*V1*V2**4 - 30*V2**5 < 0) and (8*V1 + 13*V2 - 1 < 0)) or ((6*V1**4*V2 + 37*V1**3*V2**2 + 84*V1**2*V2**3 + 83*V1*V2**4 + 30*V2**5 < 0) and (-8*V1 - 13*V2 + 1 < 0))) and (-48*V1**5*V2*V5 - 48*V1**5*V4**2 - 374*V1**4*V2**2*V5 + 48*V1**4*V2*V3**2 + 192*V1**4*V2*V3*V4 - 216*V1**4*V2*V4**2 - 1153*V1**3*V2**3*V5 + 278*V1**3*V2**2*V3**2 + 1180*V1**3*V2**2*V3*V4 - 147*V1**3*V2**2*V4**2 - 1756*V1**2*V2**4*V5 + 563*V1**2*V2**3*V3**2 + 2600*V1**2*V2**3*V3*V4 + 550*V1**2*V2**3*V4**2 - 1319*V1*V2**5*V5 + 456*V1*V2**4*V3**2 + 2412*V1*V2**4*V3*V4 + 929*V1*V2**4*V4**2 - 390*V2**6*V5 + 113*V2**5*V3**2 + 780*V2**5*V3*V4 + 390*V2**5*V4**2 - 288*V1**4*V2*V3 - 468*V1**4*V2*V4 + 6*V1**4*V2*V5 + 6*V1**4*V4**2 - 1776*V1**3*V2**2*V3 - 2886*V1**3*V2**2*V4 + 37*V1**3*V2**2*V5 + 42*V1**3*V2*V3**2 + 132*V1**3*V2*V3*V4 + 144*V1**3*V2*V4**2 - 4032*V1**2*V2**3*V3 - 6552*V1**2*V2**3*V4 + 84*V1**2*V2**3*V5 + 193*V1**2*V2**2*V3**2 + 600*V1**2*V2**2*V3*V4 + 566*V1**2*V2**2*V4**2 - 3984*V1*V2**4*V3 - 6474*V1*V2**4*V4 + 83*V1*V2**4*V5 + 288*V1*V2**3*V3**2 + 884*V1*V2**3*V3*V4 + 786*V1*V2**3*V4**2 - 1440*V2**5*V3 - 2340*V2**5*V4 + 30*V2**5*V5 + 139*V2**4*V3**2 + 420*V2**4*V3*V4 + 360*V2**4*V4**2 + 54*V1**4*V2 + 333*V1**3*V2**2 + 756*V1**2*V2**3 + 747*V1*V2**4 + 270*V2**5 < 0))));
lisp << terpri(); prin2t "BEGIN_DNF"; flush() >>;
diag_result := rldnf diag_input;
lisp << terpri(); prin2t "DNF_DONE"; flush() >>;
write "DNF_ATOMS=", rlatnum diag_result;
 >>;
lisp prin2 end!* >>;
quit;

```
<!-- END EXACT FAILING SOURCE -->
