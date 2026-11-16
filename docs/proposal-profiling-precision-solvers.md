# Proposal: MPI visibility, coarse-solve precision, extra solver libraries

Status: proposal (2026-11-16). Nothing built or uploaded. Scope: the dgns channel
recipes in this repo and how `dgns-solver` profiles and solves its pressure
coarse problem. Rules kept: stock upstream MFEM (build options fine, source
patches not), one environment, MPICH ABI, `pixi.lock` committed.

Legend: **[confirmed]** read in upstream source/docs (link given);
**[assumed]** plausible, not yet checked; **[measure]** needs a run.

## 1. MPI visibility inside MFEM and hypre

### Root cause is a GOTCHA bug, not `-fno-plt` itself

Caliper's `mpi` service wraps MPI with GOTCHA on Linux
(`CALIPER_MPIWRAP_USE_GOTCHA` whenever GOTCHA is available, PMPI otherwise;
Caliper `CMakeLists.txt` v2.15.0 lines 266-335) **[confirmed]**.

GOTCHA does rewrite GOT (`GLOB_DAT`) relocations, not only PLT ones:
`FOR_EACH_PLTREL` walks `DT_JMPREL` and then `DT_RELA`/`DT_REL`
(Caliper v2.15.0 `ext/gotcha/src/elf_ops.h:236-254`; same on GOTCHA `develop`;
added in GOTCHA's July 2023 "support function pointers" commits) **[confirmed]**.

But the choice between `DT_RELA` and `DT_REL` is made by `is_rela`, which
starts at 0 and is set **only** from the `DT_PLTREL` tag (`elf_ops.h`, the
`INIT_DYNAMIC` macro) **[confirmed]**. A library linked entirely with `-fno-plt`
has no PLT, hence no `DT_PLTREL`. `is_rela` stays 0, GOTCHA walks `DT_REL`
(absent on x86-64), and every `GLOB_DAT` entry is skipped. A library with at
least one PLT entry works. This matches the foil session: dgns built with
`-fplt` became visible **[assumed: consistent explanation, not yet
reproduced]**.

Check on turin16 (5 min), before anything else:

```sh
readelf -d $CONDA_PREFIX/lib/libHYPRE.so $CONDA_PREFIX/lib/libmfem.so | grep -E 'PLTREL|JMPREL'
readelf -r $CONDA_PREFIX/lib/libHYPRE.so | grep -E 'MPI_(Isend|Allreduce)'   # expect R_X86_64_GLOB_DAT
```

If there are no `PLTREL`/`JMPREL` lines and the MPI symbols are `GLOB_DAT`,
the diagnosis holds.

### Options

| Option | Cost | Gain |
|---|---|---|
| **A. Patch GOTCHA in our `caliper` recipe** (`is_rela` also true when `DT_RELA` is present; ~1-3 lines) + upstream PR to LLNL/GOTCHA (and Caliper's `ext/gotcha`) | Caliper rebuild only; recipe already carries one upstream-backport patch; drop it when upstream releases | MPI inside MFEM, hypre and dgns visible with no other rebuilds; dgns can drop `DGNS_PLT=ON` |
| B. Rebuild `mfem-dgns` with `-fplt` | Easy, our recipe | MFEM visible; hypre still not |
| C. Own `hypre` recipe with `-fplt` | Ongoing: track conda-forge's hypre (3.2.1, plain CMake, `recipe/build.sh`), MPI variants, `run_exports`; everything linking hypre must pin ours | hypre visible; enables hypre build options (see below) |
| D. PMPI tool via `LD_PRELOAD` (mpiP, or Caliper built with `WITH_GOTCHA=OFF`) | No rebuild of our stack; Linux only; mpiP not on our channel **[assumed: needs packaging or spack]** | All MPI visible (dynamic symbol interposition wins over GOT binding regardless of `-fno-plt`); no Caliper-region attribution with mpiP |

Linker options do not help. The GOT binding is already resolved through normal
symbol lookup, so it's GOTCHA's table walk that misses, not the binding.

**Recommendation: A.** D is the fallback if the readelf check does not
confirm the diagnosis.

### Do we need regions inside MFEM/hypre?

Not to start with. With A, MPI time nests under whatever dgns region is open.
A dgns region around the coarse BoomerAMG setup/solve gives the coarse share
and its MPI time per level of the p/h hierarchy.

hypre has native profiling, all build-time and off in conda-forge's build
(hypre v3.0.0 `src/CMakeLists.txt`) **[confirmed]**:
- `HYPRE_ENABLE_CALIPER`: Caliper annotations inside hypre
  (`src/utilities/caliper_instrumentation.h`), giving a per-AMG-level breakdown.
- `HYPRE_ENABLE_TIMING`: hypre's own timers.
- Runtime, no rebuild: `HYPRE_BoomerAMGSetPrintLevel` prints setup statistics
  (levels, rows/nnz per level, operator/grid complexity) **[confirmed]**.
  This is the first look at a coarse problem with tiny per-rank sizes.

Per-level hypre regions would justify option C later, not now.

## 2. Single or mixed precision for the coarse AMG

hypre facts:
- Single precision is a compile-time build (`HYPRE_SINGLE`, `HYPRE_Real` =
  float for the whole library) **[confirmed, hypre docs]**.
- hypre 3.x adds `HYPRE_ENABLE_MIXED_PRECISION`. Every function `Foo` gets
  `Foo_flt`/`Foo_dbl`/`Foo_long_dbl` variants, and the plain API follows
  `HYPRE_SetGlobalPrecision` (one global runtime precision); real arguments
  become `void *` / `long double` in the prototypes
  ([hypre mixed precision](https://hypre.readthedocs.io/en/latest/ch-mprecision.html))
  **[confirmed]**. A PETSc build of 3.0.0 with this option failed
  ([petsc#1828](https://gitlab.com/petsc/petsc/-/issues/1828)).
- MFEM's `HypreParMatrix`/`HypreBoomerAMG` assume `HYPRE_Real` == `real_t`
  (double for us) **[assumed: matches MFEM's single-precision build
  requirement; not re-checked in 4.10 source]**. A float coarse AMG therefore
  bypasses MFEM's wrapper. dgns would assemble the coarse CSR, convert it to
  float and call `_flt` hypre functions directly. A second, float-only hypre
  in the same process clashes on symbols, so it's not an option.

### What past runs already measured

- **PR philipnickel/dgns#162 (2026-11-04, DTU, TGV 3D moving mesh,
  Intel MPI).** Strong scaling broke down because of the AMG coarse solve.
  Its cost at many ranks is the cost per BoomerAMG V-cycle on small
  distributed levels: 0.6-0.8 ms per level at 64 ranks across nodes. Waiting
  from imbalance and the CG reductions were ruled out. Fixes, now the
  defaults (`configs/linear_solver/default.yaml`):
  - `amg_relax_type` 6: hybrid symmetric GS, so fewer inner coarse-CG
    iterations.
  - `amg_aggressive_levels` 0.
  - `amg_agglomeration_size` 2000: `HYPRE_BoomerAMGSetSeqThreshold`
    (`multigrid.cpp:834`), so levels with a few rows per rank are solved
    gathered.

  Kept as experimental, off by default:
  - `coarse_gather_ranks`: dgns-side `R A R^T` onto group leaders.
  - Non-Galerkin.
  - The additive cycles. Additive coarse correction about doubled the outer
    iterations, and mult-additive + non-Galerkin segfaults in hypre.

  After the fixes, Q3 strong-scaling efficiency against 16 ranks:

  | Mesh | Efficiency |
  |---|---|
  | 64³, 16³ seed | 0.95 at 256 ranks |
  | 32³, 8³ seed | 0.85 at 128 ranks |
  | 32³, 32³ seed (no h-levels: AMG on the full Q1 mesh) | 0.65 at 128 ranks |
- **dgns-results `scaling/strong_tgv` range times** (`cycle.coarse` against
  the pressure preconditioner, 15 steps):
  - 2026-10-28, AMG on the fine Q1 mesh: the coarse share grows from 1% at
    16 ranks to **57% at 256 ranks** (32³) and 25% (64³).
  - 2026-10-29, with h-levels down to a 512/4096-DOF Q1 coarse level: the
    coarse share is ≤5% at every rank count (3.6% at 256 ranks on 32³).

  The preprint's scaling section uses the h-level setup; there, pressure
  and reassembly fall behind first, not the coarse solve.
- **Coarse share in other recent profiles** (session transcripts): about
  10-15% on the cylinder, KCS and turin180 free-surface runs; 23% on the
  steady cavity.
- **Per-level partitioning design** (`hodgles`
  `docs/design/per-level-partitioning.md`, PR #240/#245). The 3D hill at 150
  ranks has ~470 seed cells per rank, and "the coarse Q1 h-levels and the
  AMG set the latency floor". Its stage A3 (agglomeration: Q1 levels below
  a grain redistributed onto `K_l` ranks, `gather_ranks` as the coarsest
  case) is the planned structural fix.

So AMG did limit strong scaling. The cause measured was latency per level
across nodes, and h-levels plus `SeqThreshold` already removed most of it in
the TGV. What remains is the latency floor of the matrix-free Q1 h-levels and
the AMG on wall-refined seeds at high rank counts.

### Does single precision cut communication?

Only bandwidth. It halves the bytes per message but not the number of
messages or the latency of each. The measured cost (0.6-0.8 ms per AMG level
at 64 ranks across nodes) is per-message latency on levels with a few rows
per rank. Those messages are a few hundred bytes and the Allreduces carry
scalars, so halving them saves next to nothing. Float would help only the
upper AMG levels, which are bandwidth-bound and scale.

Next steps for the coarse solve, in order (dgns-side or BoomerAMG options,
stock MFEM):
1. Measure the current defaults at scale on a wall-refined case (hill or
   channel, 150-460 ranks), with the coarse region split into AMG setup,
   AMG levels and the Q1 h-levels. Once MFEM and hypre are visible (section
   1), MPI time nests under each. Try `coarse_gather_ranks` = 16/20
   (one rank per node) together with `amg_agglomeration_size`.
2. Per-level agglomeration (per-level-partitioning A3): it also removes the
   floor of the matrix-free Q1 h-levels, which no AMG setting can.
3. Precision (mixed hypre + direct `_flt` calls) only if step 1 shows a
   bandwidth- or flop-bound coarse solve after 1-2. Not expected from the
   data.

## 3. Additional solver libraries

On conda-forge today: PETSc, Trilinos, SuperLU_DIST, MUMPS. Not on
conda-forge: Ginkgo, AmgX, STRUMPACK (checked via feedstock repos,
2026-11-16) **[confirmed]**. Each MFEM `MFEM_USE_*` option is a build flag (allowed), but
it adds a run dependency to every solver environment.

| Library | Helps with | Packaging cost (MPICH ABI) | Verdict |
|---|---|---|---|
| PETSc (`MFEM_USE_PETSC`) | Coarse: GAMG, PCTelescope (agglomeration), BoomerAMG via PETSc; Newton-Krylov: SNES (line search, JFNK) for steady drivers | conda-forge has MPI-variant builds; must use the same hypre. Check its hypre pin matches 3.2 **[assumed]** | Best single addition if steady/Newton work grows; for the coarse solve, dgns already has `coarse_gather_ranks` and plans A3, which cover PCTelescope's job |
| SuperLU_DIST / MUMPS | Exact direct coarse solve on an agglomerated level; hypre can call SuperLU_DIST itself (`HYPRE_USING_DSUPERLU`, own hypre build) | On conda-forge | Only for small, already agglomerated coarse levels |
| Ginkgo (`MFEM_USE_GINKGO`) | GPU later; mixed precision by design; its AMG (PGM) is weaker than BoomerAMG for our needs **[assumed]** | Own recipe (CMake, moderate) | Revisit with GPU work |
| AmgX | GPU AMG only | NVIDIA-only, own recipe | GPU later only |
| Trilinos/MueLu | Coarse AMG alternative, Belos/NOX | Very heavy | No |
| STRUMPACK | Low-rank direct/preconditioner | Own recipe | No |

## 4. Order and what to measure

1. **Confirm the GOTCHA diagnosis** (readelf + a 20-line MPI test lib built
   with `-fno-plt`, Caliper `mpi-report`). Patch `caliper` (option A), test in
   the recipe (a `-fno-plt` consumer whose MPI calls must appear), open the
   upstream PR. Then drop `-fplt` from dgns if A works.
2. **Measure** a wall-refined case (hill or channel, 150-460 DTU ranks) on
   the current defaults (PR #162), with `cycle.coarse` split into AMG
   setup/levels and the Q1 h-levels. Record MPI calls and bytes per region,
   and the BoomerAMG `PrintLevel` statistics (`amg_print_level`). Rerun with
   `coarse_gather_ranks` per node. Decide from this:
   - coarse plus Q1 h-levels <10% of the V-cycle at the target rank count:
     stop;
   - latency-bound (many small messages): per-level agglomeration
     (per-level-partitioning A3);
   - bandwidth- or flop-bound AMG levels: consider mixed precision.
3. **Libraries**: PETSc only when a steady/Newton driver needs SNES or 2.2
   proves too much code; Ginkgo/AmgX when GPU work starts.

Packages touched by step 1: `caliper` only. No builds or uploads without
explicit OK.

## Sources

- Caliper v2.15.0 `CMakeLists.txt`, `ext/gotcha/src/elf_ops.h`:
  https://github.com/LLNL/Caliper/tree/v2.15.0
- GOTCHA `src/elf_ops.h`, `src/gotcha.c` (develop), releases:
  https://github.com/LLNL/GOTCHA
- hypre v3.0.0 `src/CMakeLists.txt`, `src/parcsr_ls/HYPRE_parcsr_ls.h`:
  https://github.com/hypre-space/hypre/tree/v3.0.0
- hypre mixed precision: https://hypre.readthedocs.io/en/latest/ch-mprecision.html
- PETSc + hypre 3.0 mixed precision build issue: https://gitlab.com/petsc/petsc/-/issues/1828
- conda-forge hypre recipe: https://github.com/conda-forge/hypre-feedstock/tree/main/recipe
- Past measurements: PR philipnickel/dgns#162 (description: DTU tables and
  job paths under `/work3/s214960/dgns/develop/scaling/`);
  `dgns-results/scaling/strong_tgv/` run logs (`[hodgles range times]`);
  preprint `Chapters/04_Results/performance.tex`; `hodgles`
  `docs/design/per-level-partitioning.md`.
