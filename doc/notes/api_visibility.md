# API visibility and compatibility

Discussion recorded on 2026-11-15 and deferred. No policy has been adopted.
The recommendations in the reply below are proposals, not decisions.
Local file links have been made repository-relative.

## User question

We are not at all consistent regarding private methods. Generally, there is
a problem that non-private methods are part of the API and changing their
contract requires a major revision. Another aspect is that classes can be
made private; what does this mean for their private and non-private methods?
Furthermore, private form whom? It could mean "should be used only within
the class" or "may be used within logic1 by the developers but external
users of the library. Are there best practices?

## Assistant reply

Yes. We need to distinguish **the library’s interface to external users**
from **a class’s interface to other Logic1 code**. My earlier explanation
did not separate these sufficiently.

Python does not provide distinct class-private and package-private access
levels. A leading underscore indicates a non-public implementation detail,
but does not specify exactly which collaborators may use it. Double
underscores perform name mangling, primarily to prevent collisions with
subclasses; they do not enforce privacy.
[Python documentation](https://docs.python.org/3/tutorial/classes.html#private-variables)

For Logic1, I would distinguish four categories:

| Category                    | Intended users                        | Compatibility obligation                          |
| --------------------------- | ------------------------------------- | ------------------------------------------------- |
| Public library API          | External callers                      | Preserve the published contract                   |
| Public subclass API         | Authors of external subclasses        | Preserve the documented extension contract        |
| Internal Logic1 interface   | Other Logic1 classes and modules      | Update internal callers together when changing it |
| Class implementation detail | The class’s implementation            | Other code should avoid depending on it           |

**An internal class can have methods without underscores.** PEP 8
explicitly treats an interface as internal when its containing class,
module, or package is internal. Thus, `_CartesianBnf.strict_dnf` would not
automatically become an external API merely because `strict_dnf` lacks an
underscore.
[PEP 8](https://peps.python.org/pep-0008/#public-and-internal-interfaces)

Within such a class, we could adopt this additional convention:

- `strict_dnf`: an operation available to collaborating Logic1 code.
- `conjoin_dnfs`: an extension point for Logic1’s subclasses.
- `_matrix_dnf`: an implementation helper of the class.

The last distinction would be **our project convention**, rather than a
Python access rule. Logic1 already has a comparable arrangement: the
internal class [_Range](../../logic1/theories/RCF/range.pyx#L152) provides
ordinary methods such as `intersection` alongside helpers such as
`_imul_core`.

External subclassing deserves particular attention. If we support external
subclasses, the conditions under which overridden methods are called
become part of the contract. Changing those calls can break compatibility
without changing a signature. Python’s own compatibility policy explicitly
recognizes this issue.
[PEP 387](https://peps.python.org/pep-0387/#backwards-compatibility-rules)

Regarding versions, **incompatible changes to the declared public API**
require a major version increase under SemVer after 1.0. Compatible changes
do not. SemVer treats `0.x` differently, and Logic1’s
[release policy](../../releases/v0.3.0.md) explicitly allows API evolution
between minor releases during that phase.
[SemVer specification](https://semver.org/spec/v2.0.0.html)

For our BNF classes, the present documentation leaves the boundary unclear.
It documents `CartesianBnf`, demonstrates its use, and explains subclassing.
Saying that a section “addresses implementers” does not distinguish Logic1
developers from external subclass authors. PEP 8 treats documented
interfaces as public unless explicitly designated internal or provisional;
`__all__` can also help declare a module’s public surface.
[PEP 8](https://peps.python.org/pep-0008/#public-and-internal-interfaces)

**My recommendation is to decide first whether external subclassing is
supported.** Unless we deliberately want that commitment, I would make
`dnf`, `cnf`, and `Simplification` the supported external interface and
explicitly designate the implementation classes as internal to Logic1.
Their methods could then retain readable names for internal collaboration,
with underscores reserved for class implementation helpers. Their
developer documentation should remain available, but state the
compatibility boundary clearly.
