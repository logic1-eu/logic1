# Usage of Commit Tags in Logic1

Author: T. Sturm  
Last changed: 2026-10-03

## Suggested policy

The following policy is based on the AI-generated counts
[at the end of the document](#ai-genrated-counts-as-of-oct-1-2026).

### General format

Tags have the following syntax:

`<tags>     = <tag-list>": "`  
`<tag-list> = <tag> [", " <tag-list>] | <tag-list>", wip"`  
`<tag>      = <major>["/"<minor>]`

`wip` stands for work in progress. Each and every commit has tags. Commits
should be split in favor of accummulating tags whenever possible. There is
single space after the colon. The content of the mandatory subsequent commit
message starts with a capital letter.

### Admitted major tags

This list is open for discussion and extension. All tags used should be
documented here.

| Tag     |  meaning                                                          |
|---------|-------------------------------------------------------------------|
| `bench` |  benchmark sub-tree `logic1/benchmarks`                           |
| `build` |  project infrastructure `pyproject.toml`, `Makefile`, etc.        |
| `ci`    |  continuous integration                                           |
| `doc`   |  documentation sub-tree `logic1/doc`                              |
| `src`   |  source sub-tree `logic1/src`                                     |
| `test`  |  test files; will become subtree `logic1/tests`                   |

### Minor tags

- modules like `abc`, `firstorder`, `Complex`, `RCF`, etc.

### Formatting

Follow common conventions:

- Headlines of commit messages have at most 50 caracters
- Lines in the body of commit messages have at most 72 caracters

## AI-generated counts as of Oct 1, 2026

This list is based on Git commit subjects. A tag is a leading prefix followed
by `:`; comma-separated prefixes are counted as separate tags. For example,
`doc, src: ...` contributes one use each to `doc:` and `src:`.

There are **42 distinct tags**. Tags containing `/` are grouped by their first
component. Counts refer to the exact tag, not the group as a whole.

## `src`

| Tag | Commits |
| --- | ---: |
| `src:` | 129 |
| `src/Complex:` | 7 |
| `src/RCF:` | 42 |
| `src/RCF.simplify:` | 1 |
| `src/RFC:` | 1 |
| `src/Sets:` | 4 |
| `src/abc:` | 13 |
| `src/firstorder:` | 7 |
| `src/support:` | 2 |

## `doc`

| Tag | Commits |
| --- | ---: |
| `doc:` | 52 |
| `doc/Complex:` | 2 |
| `doc/RCF:` | 7 |
| `doc/tst:` | 1 |

## `ci`

| Tag | Commits |
| --- | ---: |
| `ci:` | 10 |
| `ci/conda:` | 1 |
| `ci/docs:` | 2 |
| `ci/pypi:` | 1 |
| `ci/workflow/pypi:` | 1 |

## `build`

| Tag | Commits |
| --- | ---: |
| `build:` | 8 |
| `build/Makefile:` | 1 |

## `test`

| Tag | Commits |
| --- | ---: |
| `test:` | 3 |
| `test/RCF:` | 8 |

## `bench`

| Tag | Commits |
| --- | ---: |
| `bench:` | 1 |
| `bench/RCF:` | 1 |

## `RCF`

| Tag | Commits |
| --- | ---: |
| `RCF:` | 5 |
| `RCF/term/sage:` | 1 |

## `Complex`

| Tag | Commits |
| --- | ---: |
| `Complex/simplify:` | 1 |

## Other unscoped tags

| Tag | Commits |
| --- | ---: |
| `Bugfix:` | 2 |
| `CHANGELOG:` | 1 |
| `Sets:` | 1 |
| `abc:` | 1 |
| `abc.QuantifierElimination:` | 1 |
| `conda:` | 4 |
| `firstorder:` | 1 |
| `fo:` | 3 |
| `pyproject:` | 1 |
| `refactor:` | 1 |
| `support:` | 1 |
| `testRCF:` | 1 |
| `th:` | 7 |
| `tst:` | 3 |
| `wip:` | 11 |

These are observed usages, not a standardized tag policy. Some entries are historical one-offs or apparent typos, such as `src/RFC:` and `testRCF:`.
