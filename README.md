# Polyphage

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sym-jay/polyphage/blob/main/Polyphage.ipynb)

Multi-objective computational design of *Ideonella sakaiensis* PETase
(IsPETase) variants for PET-plastic degradation.

ProteinMPNN proposes sequences against a fixed backbone with the catalytic triad
frozen; ThermoMPNN predicts the stability cost of each mutation; NSGA-II searches
that space under competing objectives and returns a Pareto front rather than a
single "best" sequence.

The problem being targeted: wild-type IsPETase denatures around 46–48 °C, but PET
only becomes accessible to enzymes near its glass transition at 65–70 °C. The
natural enzyme falls apart before the plastic is soft enough to chew.

## Results so far

### T140D does not reproduce

Earlier versions of this README led with a 9-mutation variant containing
**T140D**, an exact DuraPETase mutation, recovered by a run that had no access
to that paper. That result is withdrawn.

The run that produced it predated the reproducibility fix, when the optimizer's
operators drew from OS entropy instead of the run seed. Re-running the same
configuration under the fixed code — `pop_size=40, n_gen=25, max_mutations=10,
seed=1` — produces an entirely different front and **no exact literature
matches at all**. None of the three reproducible runs now logged contains one.

So T140D came from a single unrepeatable draw. This is the correct outcome of
fixing the bug, not a failure of the method: whether the search *can* recover
T140D is now a question about how often it does so across seeds, which is what
`replicate.py` measures and what has not yet been run to completion.

The older `runs.jsonl` entries have been replaced rather than kept alongside,
because an unreproducible run log is worse than none — it invites exactly the
claim that was just withdrawn.

### D186H: the first recovery with a claim behind it

A 20-seed replication of the three-objective search
([`results/replication_20seeds.md`](results/replication_20seeds.md)):

**D186H — a FAST-PETase thermostability mutation (Lu et al. 2022) — is recovered
in 2 of 20 independent runs, above chance under every null tested.** The search
had no access to the paper identifying it.

| Null | p (Holm) | Chance/run | Enrichment |
|---|---|---|---|
| Uniform | 0.0030 | 0.0018 | 1.0× |
| ProteinMPNN prior | 0.0318 | 0.0060 | 3.3× |
| ddG prior, residue | 0.0077 | 0.0029 | 1.6× |
| **ddG prior, residue + position** | **0.0339** | 0.0062 | 3.4× |

The last row is the strongest control available: ThermoMPNN steering both which
positions are mutated and which residue goes there, matching what the
three-objective search was actually told. Four other candidate mutations
appeared once each and none clears correction.

In aggregate, 5 of 20 seeds recovered some exact literature mutation (25%, 95%
CI [11%, 47%]) against a 2.5% chance rate — p = 1.1 × 10⁻⁴.

**What this does not say.** One mutation of fourteen, at a 10% recovery rate
with a wide interval [3%, 30%]. Recovering a published mutation shows the search
finds real signal; it says nothing about the other mutations on these fronts,
and nothing at all about catalysis. Full limits in
[`results/replication_20seeds.md`](results/replication_20seeds.md).

### The third objective fixes the degenerate front

The two-objective search was never a real trade-off, as this README has said
from the start. Adding ddG confirms it:

| | 2 objectives | 3 objectives (+ ddG) |
|---|---|---|
| Variants on the front | 3 | **16** |
| Mean pairwise Hamming | 4.7 | **9.3** |
| Mutations in *every* variant | 6 | **1** |
| ddG range | — | −3.41 to −1.17 |

The two-objective front is one converged lineage: six of its mutations appear in
all three variants. With ddG in play the front spans 4–10 mutations with only
A179V universal, and the ddG values are live rather than the constant zero a
broken table produced earlier.

That is the pipeline working as designed, and it is the configuration worth
replicating. Both runs took ~139 s on a T4.

Full run log: [`results/runs.jsonl`](results/runs.jsonl). ThermoMPNN's zero-shot
validation, which *is* a defensible result, is in
[`results/thermompnn_validation.md`](results/thermompnn_validation.md).

## Status

| Component | State |
|---|---|
| Residue mapping + triad gate | Working, gated, tested |
| ProteinMPNN batch scorer | Working, checked against the reference path |
| NSGA-II search | Working; **reproducible as of this branch** |
| Randomization control (uniform) | **Run** on the 20-seed replication |
| Randomization control (ProteinMPNN prior) | **Run** on the 20-seed replication |
| Randomization control (ThermoMPNN ddG prior, residue) | **Run** — D186H survives |
| Randomization control (ddG prior, residue + position) | **Run** — D186H survives |
| Replication across seeds | **Done** — 20 seeds, see [`results/replication_20seeds.md`](results/replication_20seeds.md) |
| ThermoMPNN ddG | **Working, verified end to end** (100% coverage, sign confirmed) |
| ThermoMPNN zero-shot validation | **Done** — see [`results/thermompnn_validation.md`](results/thermompnn_validation.md) |
| Activity / docking screen | Not started |
| Wet-lab validation | Not started |

Run the tests with `python -m pytest tests/ -q`. They need neither a GPU nor
ProteinMPNN: both the target and the scoring model are synthetic, because what
is under test is the search's RNG plumbing and the ddG residue mapping.

## Reference structures

| What | PDB | Note |
|---|---|---|
| Wild-type IsPETase | `6EQE` | 0.92 Å, Austin et al. 2018 PNAS. The design target. |
| Wild-type IsPETase (apo) | `5YFE` | 1.39 Å alternative |
| Wild-type + MHET analog | `5XG0` | 1.58 Å, for later docking validation |
| FAST-PETase | `7SH6` | Lu et al. 2022 Nature. Benchmark only, not a design input. |

Catalytic triad: **Ser160** (nucleophile), **His237** (general base),
**Asp206** (orients His237). Frozen in every design.

Earlier notes in this project cited `6EQM` and `7VVE`. Both were wrong —
`7VVE` is not FAST-PETase at all. Use the table above.

## Layout

```
polyphage/
  encoding.py    vector <-> sequence <-> PDB residue numbers; triad gate
  scoring.py     ProteinMPNN foldability; batched + subprocess reference
  optimize.py    NSGA-II problem, categorical operators, objective specs
  validate.py    hotspot overlap, front reporting, run log, plotting
  baselines.py   the no-search control: greedy ddG stacking
  control.py     randomization controls: uniform and ProteinMPNN-prior nulls
  replicate.py   recovery rate across seeds, with CIs and Holm correction
  stability.py   ThermoMPNN ddG via a cached site-saturation table
tests/             regression tests, 33 of them, none needing a GPU
build_notebook.py  generates Polyphage.ipynb
results/           run log and figures
```

**`Polyphage.ipynb` is generated.** Edit the modules, then run
`python3 build_notebook.py` to regenerate it. Hand-edits to the notebook are
discarded by the next build.

> **Never use Colab's *Save a copy to GitHub* on this notebook.** It commits your
> in-session copy over the repository's. Commit `c78ce43` did exactly that: it
> reverted `Polyphage.ipynb` on `main` to a pre-fix version, removing cells that
> had been merged an hour earlier, while the repository's `.py` files stayed
> correct. The result was a notebook that ran stale code with nothing saying so,
> and several debugging rounds spent on symptoms of the drift.

The notebook no longer embeds the modules as `%%writefile` cells, which is what
made that failure invisible. Cell 1 clones `polyphage` from GitHub at
`POLYPHAGE_REF` (a branch, tag or commit) and prints the SHA it got beside the
SHA the notebook was built from. **What runs is whatever is pushed** — so a
change has to be committed before it can be tested, which is the right
constraint for results that have to be reproducible. To test a feature branch,
change `POLYPHAGE_REF` in Cell 1.

## Usage

Click the Colab badge above, set Runtime → T4 GPU, and run top to bottom.
Cells 4 and 6 are gates: if either raises, stop rather than continuing, because
everything downstream depends on the residue mapping being correct.

Locally:

```python
from polyphage import *

target = load_target("inputs/design_target/6EQE.pdb", "outputs/parsed_6EQE.jsonl")
batch  = BatchScorer(mpnn_dir=".", pdb_path="inputs/design_target/6EQE.pdb")

objectives = [foldability_objective(batch), novelty_objective(target)]
res = run(target, objectives, pop_size=40, n_gen=25, max_mutations=10)
print_front(report_front(target, res, [o.name for o in objectives]))
```

## Design decisions

**Categorical operators, not continuous ones.** Amino acid identity is
categorical. pymoo's default SBX crossover and polynomial mutation assume an
ordered continuous variable, under which "index 5 (F) lies between 4 (E) and
6 (G)" — biochemically meaningless, and it biases the search toward the middle of
the alphabet. `optimize.py` uses uniform crossover and random-reset mutation.

**Mutations can revert.** Without reversion, mutation count only ratchets upward
and the low-mutation end of the front collapses — exactly the end that matters,
since FAST-PETase is 5 mutations.

**Scores are averaged over fixed decoding orders.** ProteinMPNN scores a sequence
under a random autoregressive decoding order, with a spread of ~0.02 nats on this
protein — comparable to the score change from one or two mutations. Averaging over
8 fixed-seed orders makes the objective deterministic and cuts that noise.
Otherwise NSGA-II ranks candidates by dice.

**Stability lookups, not model calls.** ThermoMPNN's site-saturation scan runs
once on the wild-type and is cached as a table, so scoring a variant inside the
loop is a handful of lookups.

**The seed has to actually reach the operators.** pymoo passes operators a
seeded generator under `random_state`; it has never passed one under `seed`.
Reading `kwargs.get("seed")` returned `None`, `np.random.default_rng(None)` then
drew from OS entropy, and every run was irreproducible while still recording
`"seed": 1` in the log. `tests/test_reproducibility.py` fails if this regresses.

## Two bugs worth knowing about

Both were live in the code that produced the results above, and both were the
kind that fail quietly rather than loudly.

**Irreproducible runs.** Described just above. Consequence: no previously logged
run can be regenerated, and `repeat_seeds_summary` was not varying what its name
claims. Fixed on this branch.

**ThermoMPNN residue numbering.** ThermoMPNN's SSM output indexes positions
0-based along the sequence; the design target uses PDB numbering, which on 6EQE
starts at residue 29. Reading one as the other shifted every ddG by 29 residues
— and because ~89% of positions still overlapped numerically, the coverage check
passed. The parser now maps through `target.residue_numbers` and verifies the
wildtype letter on every row, so a mismatch raises instead of silently poisoning
the objective. Any ddG number produced before this fix should be discarded.

A related trap, not a bug but worth stating: nothing in the pipeline would
notice if ThermoMPNN's ddG sign were inverted — the optimizer would simply
minimize toward the *least* stable variants. `StabilityOracle.check_sign_convention()`
tests the sign against the known heat-stable mutations and should be run before
any three-objective search.

## What a reportable claim looks like

A single run recovering T140D is an anecdote. The reportable form is:

> The method recovers T140D in k of n independent runs (95% CI [lo, hi]),
> against a chance rate of p per run.

`replicate.py` produces exactly that table. Three things make it defensible:

**Confidence intervals, Wilson not normal.** With 20–50 seeds and counts often
at 0 or n, the normal approximation produces intervals outside [0, 1]. Wilson
does not.

**Correction for multiple testing.** There are 14 known mutations to recover.
Testing each and reporting the smallest p-value is cheating; `holm_adjust`
controls the family-wise error rate without assuming the tests are independent,
which they are not — mutations co-occur within a variant.

**Three nulls, and which one is honest depends on what the search was told.**

| Null | Question it answers | Use when |
|---|---|---|
| Uniform random mutation | Did the search beat chance? | Always, as the floor |
| ProteinMPNN's prior | Did the optimizer find it, or did ProteinMPNN already favour it? | Two-objective runs |
| ThermoMPNN's ddG prior | …or did ThermoMPNN already favour it? | Three-objective runs |

The prior nulls keep position choice uniform — the optimizer isn't told which
positions to change either — but draw the replacement residue from the model's
own distribution. A mutation that survives one was located by the search rather
than handed over.

The third is not a formality. ThermoMPNN's zero-shot scan already ranks 12 of
the 14 known mutations in the top quartile, **T140D among them at percentile
18**. Reporting a three-objective result against the uniform null alone would
credit the optimizer with what the stability model supplied. Equally, running
the ddG null against a *two*-objective front is wrong in the other direction —
penalising the search for a prior it never had.

`replicate.py` also reports **front diversity** — mean pairwise Hamming distance
within a Pareto front. This project's run 2 returned two variants sharing seven
of nine mutations: one converged lineage presented as two findings. The metric
makes that visible instead of leaving it to be noticed by eye.

## Limitations

Stated plainly, because they bound what this project can claim:

- **No wet-lab validation.** This produces ranked candidates, not verified enzymes.
- **ProteinMPNN's score measures foldability, not function.** Nothing in the
  current objectives measures catalysis. These variants are not shown to degrade
  PET faster, or at all.
- **ddG is additive here.** ThermoMPNN is trained on single point mutants; summing
  per-mutation ddG across a 9-mutation variant ignores epistasis and generally
  overestimates stabilisation. It is a screening heuristic for ranking, not a
  ground truth.
- **Stability and activity can trade off.** Rigidifying a protein can stiffen the
  active site it needs for chemistry.
- **Beating wild-type on ProteinMPNN NLL is expected, not a finding.** Natural
  enzymes sit off the inverse-folding optimum because evolution trades stability
  against function and regulation. A lower NLL than wild-type is a known
  property of these models, so it cannot carry a result on its own.
- **`novelty_objective` is not a real objective.** Maximising mutation count is
  not a property anyone wants in an enzyme; it exists to give the front a second
  axis. Until ddG or activity replaces it, the two-objective front is closer to
  a diversity sweep than a trade-off.

## Roadmap

1. ~~Re-run everything under the reproducibility fix.~~ **Done** — and it
   withdrew the T140D result. See *Results so far*.
2. ~~Three-objective runs, after `check_sign_convention()` passes.~~ **Done** —
   the ddG objective is live and the front is no longer degenerate.
3. ~~Replication across seeds.~~ **Done** at 20 seeds — D186H recovered 2/20,
   significant against the uniform and ProteinMPNN-prior nulls.
4. ~~The ddG-prior null on that replication.~~ **Done** — D186H survives at
   p = 0.0077.
5. ~~The ddG null with position weighting.~~ **Done** — D186H survives at
   p = 0.0339. All controls are now run.
6. **The greedy-ddG baseline.** Implemented in `baselines.py`, not yet run, and
   it should be run before anything is written up. ThermoMPNN ranks N233K inside
   its top ten substitutions, so stacking the best positions recovers a known
   mutation deterministically — against the search's 25% of runs. The search has
   to win on foldability instead, and that has not been measured.
7. **Extend to 50 seeds**, to tighten the 10% recovery rate's [3%, 30%]
   interval. ~2 h on a T4; `retest` re-scores against every null for free.
8. Activity screen on top candidates: ESMFold → triad geometry check → PET
   oligomer docking (AutoDock Vina / GNINA), measuring Ser160 OG to ester
   carbonyl distance. This is the part that speaks to actual plastic degradation.
9. FastAPI + Docker serving layer, so other researchers can supply their own
   constraints and get ranked sequences back.

## How this was built

The pipeline was developed with AI assistance (Claude Code, Anthropic). That
covered module design and implementation, the ProteinMPNN and NSGA-II
integration, the batched scoring path, and the validation tooling. Commits carry
a `Co-Authored-By: Claude Opus 5.5` trailer.

The scientific direction, the choice of objectives, the structures, and the
interpretation of results are the author's. Several factual errors introduced
early on — notably the wrong PDB accession codes — were caught by verifying
identifiers against RCSB and the primary literature rather than trusting
generated text. That verification step is part of the method, not an aside.

## References

- Austin et al. 2018, *PNAS* — IsPETase structure, 6EQE
- Lu et al. 2022, *Nature* — FAST-PETase (S121E/D186H/R224Q/N233K/R280A), 7SH6
- Cui et al. 2021, *Nature Communications* — DuraPETase
- Dauparas et al. 2022, *Science* — [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)
- Dieckhaus et al. 2024, *PNAS* — [ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN)
- Tsuboyama et al. 2023, *Nature* — Megascale, ThermoMPNN's training set.
  331 natural + 148 designed domains of **40–72 aa**, which is why IsPETase
  (265 residues) cannot be in it. See
  [`results/thermompnn_validation.md`](results/thermompnn_validation.md#is-this-really-zero-shot)
- Deb et al. 2002, *IEEE Trans. Evol. Comput.* — NSGA-II
