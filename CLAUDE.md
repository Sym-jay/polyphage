# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Multi-objective computational design of *Ideonella sakaiensis* PETase (IsPETase)
variants for PET degradation. ProteinMPNN proposes sequences against a fixed
backbone with the catalytic triad frozen, ThermoMPNN predicts the stability cost
of each mutation, and NSGA-II searches that space, returning a Pareto front.

It is a **research repository, not a product**. The output is ranked candidate
sequences, never verified enzymes. Nothing here measures catalysis. The hard part
is not making the search run — it is establishing that a result means anything,
which is why roughly half the code is controls, nulls and statistics.

## Commands

```bash
python -m pytest tests/ -q                                    # full suite
python -m pytest tests/test_control.py -q                      # one file
python -m pytest tests/test_replicate.py::test_wilson_matches_published_value -q   # one test
python build_notebook.py                                       # regenerate the notebook
```

No linter, formatter or CI is configured. The test suite needs **no GPU and no
ProteinMPNN** — see *Testing conventions* below.

The real pipeline runs in Colab on a T4; `torch` and the two upstream model
repos are cloned at runtime and are gitignored, never vendored.

## Architecture

Data flows in one direction, and `Target` is the spine:

```
encoding.Target ──> scoring (ProteinMPNN NLL) ──┐
       │                                        ├──> optimize (NSGA-II) ──> validate ──> control ──> replicate
       └──────────> stability (ThermoMPNN ddG) ─┘
```

- **`encoding.py`** — `Target` is the single source of truth for index mapping,
  and every other module defers to it. `load_target` is a hard gate: it refuses
  to build a `Target` unless Biopython and ProteinMPNN agree on the sequence
  residue by residue, and unless residues 160/206/237 really are Ser/Asp/His.
- **`scoring.py`** — `BatchScorer` (fast, batched, the one used in the loop) and
  `SubprocessScorer` (slow, ProteinMPNN's own code path, the reference).
  `check_equivalence` must pass before trusting the fast one.
- **`optimize.py`** — NSGA-II over integer vectors. Objectives are pluggable
  `ObjectiveSpec`s and are **all minimized**; negate inside `fn` to maximize.
- **`validate.py`** — turns a pymoo result into `VariantReport`s and compares
  against the FAST-PETase / DuraPETase tables.
- **`control.py`** — three randomization nulls (see below).
- **`replicate.py`** — recovery rate across seeds, with Wilson intervals, exact
  binomial tails and Holm correction. This produces the reportable claim.
- **`stability.py`** — ThermoMPNN ddG, cached as a site-saturation table so
  scoring a variant in the loop is dictionary lookups, not model calls.

## Invariants that have already caused bugs

Each of these cost real debugging time. They fail silently, which is why they
are listed rather than left to be rediscovered.

**Two numbering systems, and they differ by 29.** The design target uses PDB
numbering (6EQE chain A is residues 29–293); sequences and optimizer vectors are
0-based. ThermoMPNN's CSV emits 0-based indices. Always map through
`target.residue_numbers`. Reading one as the other shifted every ddG by 29
residues while still covering ~89% of positions numerically — enough to pass a
coverage check. `StabilityOracle._parse` now verifies the wildtype letter on
every row so a mismatch raises immediately.

**pymoo passes operators `random_state`, never `seed`.** Reading
`kwargs.get("seed")` returns `None`, `default_rng(None)` draws from OS entropy,
and runs become irreproducible while still logging the seed they were given.
`optimize._rng` handles this; `tests/test_reproducibility.py` fails if it
regresses.

**ProteinMPNN scores depend on a random decoding order**, with a spread of
~0.02 nats on this protein — comparable to the effect of one or two mutations.
Scores are averaged over 8 fixed-seed orders so the objective is deterministic.
Never compare unseeded single-draw scores.

**The ddG sign convention is verified, not assumed.** If it were inverted the
optimizer would minimize toward the *least* stable variants and still produce a
tidy front; nothing else would notice. Run
`StabilityOracle.check_sign_convention()` before any three-objective search.

**Choose the null that matches what the search was told.** Using the wrong one
is wrong in both directions:

| Null | Controls for | Use with |
|---|---|---|
| `uniform_aa_sampler` | chance | always, as the floor |
| `mpnn_prior_sampler` | ProteinMPNN's prior | two-objective runs |
| `ddg_prior_sampler` | ThermoMPNN's prior | three-objective runs |

This matters concretely: ThermoMPNN already ranks T140D at percentile 18 of
5035 substitutions (`results/thermompnn_validation.md`), so a ddG-driven run
that "finds" T140D against the uniform null alone is crediting the optimizer
with what the stability model supplied.

**Summed ddG ignores epistasis.** ThermoMPNN is trained on single point mutants;
adding per-mutation ddG across a 9-mutation variant generally overestimates
stabilization. It ranks candidates — it is not ground truth.

## The notebook is generated

`Polyphage.ipynb` is built by `build_notebook.py`. Hand-edits are discarded on
the next build. Edit the modules or the builder, then regenerate and commit.

**Never use Colab's *Save a copy to GitHub* on it.** Commit `c78ce43` did
exactly that and reverted the notebook on `main` to a pre-fix version, deleting
cells merged an hour earlier while the `.py` files stayed correct.

Cell 1 clones `polyphage` from GitHub at `POLYPHAGE_REF` rather than embedding
the modules, which is what made that failure invisible. The consequence worth
knowing: **what runs in Colab is whatever is pushed**, so a change has to be
committed before it can be tested there. To test a branch, change
`POLYPHAGE_REF` in Cell 1 — but merged branches are deleted on this repository,
so once a PR lands, `"main"` is the only ref that resolves.

**Replacing the files on disk is not enough to change what runs.** Python serves
whatever it imported first, so a fresh clone is silently ignored by a session
that has already imported the package. Cell 1 now purges `sys.modules` for this
reason, and any cell that swaps the code by hand must do the same:

```python
for m in [m for m in sys.modules if m.startswith("polyphage")]:
    del sys.modules[m]
```

This failure mode has cost several debugging rounds. It presents as a fix that
"didn't work" — the error message comes from code that is no longer on disk.

## Testing conventions

Tests run on synthetic `Target`s and stubbed models — no GPU, no ProteinMPNN, no
ThermoMPNN. `_parse` is exercised against hand-written CSVs; the prior samplers
against a stub supplying `position_log_probs` or a `.table`.

So when adding anything that depends on a model, **take a pluggable callable**
rather than the model itself, the way `sample_random_variants` takes an
`aa_sampler`. That is what keeps the nulls testable, and it is why a third null
could be added without touching the code paths around it.

## How results get claimed

The README carries a Status table distinguishing *implemented* from *actually
run*; most of this pipeline is the former. Keep it honest when that changes.

Results go in `results/` as records carrying their raw data and the code SHA
that produced them (`results/thermompnn_validation.md` is the template). A
number whose code version is unknown cannot go in a paper.

Do not state a finding more strongly than its control supports. A single run
recovering a known mutation is an anecdote; the reportable form is a recovery
rate across independent seeds with a confidence interval and an explicit null,
which is what `replicate.py` exists to produce.
