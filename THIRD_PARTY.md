# Third-Party Software

## ASTER

Afresh's native statistics adapter includes ASTER headers and uses ASTER tree
and quartet-coloring implementations. ASTER is not vendored in this
repository. Obtain it from <https://github.com/chaoszhang/ASTER> and review
its license and notices before building or redistributing a combined work.

The ASTER checkout inspected during release preparation is based on commit
`34f962a1d9dfc4d598ed04aecd219822e61f091c` and carries a GNU Affero General
Public License version 3 notice. That local checkout had uncommitted changes,
so the commit alone does not identify every source file used during validation.

No ASTER source files or binaries are included here.

## Python Dependencies

The filtering workflow uses NumPy, SciPy, and DendroPy. Tests and runtime
requirements do not vendor these packages.
