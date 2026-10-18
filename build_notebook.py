"""Generate Polyphage.ipynb, which clones its own code from this repository.

Run: python3 build_notebook.py

The notebook used to embed every module as a ``%%writefile`` cell, on the theory
that this made it self-contained. It did the opposite. The notebook carried a
*copy* of the code that drifted from the repository with nothing announcing the
drift, and Colab's "save to GitHub" eventually pushed a stale copy over ``main``
(commit c78ce43), silently reverting the modules for anyone who ran it while the
repository's own ``.py`` files were perfectly correct. Several debugging rounds
were spent on symptoms of that drift rather than on real bugs.

So Cell 1 now clones ``polyphage`` from GitHub at a ref you choose. Consequences
worth knowing:

* The code that runs is whatever is pushed, so a change must be committed before
  it can be tested. That is a constraint, and it is the right one for work whose
  results have to be reproducible.
* ``POLYPHAGE_REF`` in Cell 1 selects a branch, tag or commit, so testing a
  feature branch means editing one string.
* A Colab save-back can no longer poison the code. At worst it makes the
  orchestration cells stale, which Cell 1's SHA comparison reports.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
REPO = "https://github.com/Sym-jay/polyphage"

# This script is deliberately idempotent: running it twice on an unchanged tree
# produces a byte-identical notebook. An earlier version stamped Cell 1 with the
# build-time HEAD, which meant every rebuild dirtied Polyphage.ipynb whether or
# not anything had actually changed -- spurious diffs on every commit, and a
# `git status` nobody could trust after a rebuild. The provenance that matters
# is the SHA Cell 1 prints at *runtime* from the clone it actually ran.


def md(text):
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    }


def code(text):
    lines = text.strip("\n").split("\n")
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [l + "\n" for l in lines[:-1]] + [lines[-1]],
    }


cells = [
    md("""
# Polyphage

Multi-objective design of IsPETase variants: ProteinMPNN proposes sequences
against a fixed backbone with the catalytic triad frozen, NSGA-II searches that
space under competing objectives and returns a Pareto front.

**Runtime: T4 GPU** (Runtime > Change runtime type > T4 GPU). Free tier is enough.

Run cells in order. Cell 4 and Cell 6 are gates -- if either raises, stop and fix
it rather than continuing.

Reference structures: `6EQE` = wild-type IsPETase (the design target),
`7SH6` = FAST-PETase (benchmark only). Earlier `6EQM`/`7VVE` were wrong;
`7VVE` is not FAST-PETase.
"""),
    md("""
## Cell 1 — Environment

`POLYPHAGE_REF` selects which commit of the project to run: a branch, a tag or a
SHA. The code is cloned from GitHub rather than written out by this notebook, so
what runs is always what is pushed — change a module, push it, re-run this cell.

Re-running this cell is enough to pick up new code: it re-clones *and* drops the
already-imported modules, so Python does not keep serving the version it loaded
first. Note that merged branches are deleted on this repository, so once a
PR lands, `"main"` is the only ref that will resolve.

The cell prints the SHA it cloned. **Record that SHA with any result you
report** — it is the only thing that identifies which code produced a number
after the fact, and a result whose code version is unknown cannot go in a paper.
"""),
    code(f"""
POLYPHAGE_REF = "main"   # branch, tag or commit to run

!git clone -q https://github.com/dauparas/ProteinMPNN.git
%cd ProteinMPNN
!pip install -q biopython pymoo pandas matplotlib

import subprocess, sys

!rm -rf /content/pp_src /content/ProteinMPNN/polyphage
# Full clone, then checkout: `clone --branch` rejects a bare commit SHA, and
# POLYPHAGE_REF is documented to accept one.
subprocess.run(["git", "clone", "-q", "{REPO}", "/content/pp_src"], check=True)
subprocess.run(
    ["git", "-C", "/content/pp_src", "checkout", "-q", POLYPHAGE_REF], check=True
)
!cp -r /content/pp_src/polyphage /content/ProteinMPNN/polyphage
!mkdir -p outputs inputs/pdbs inputs/design_target

sys.path.insert(0, '/content/ProteinMPNN')

# Drop any already-imported copy. Re-running this cell replaces the files on
# disk, but Python keeps whatever it imported first, so without this the fresh
# clone is ignored and the session silently keeps running the old code. That
# has cost several debugging rounds on this project; it is two lines.
for _m in [_m for _m in sys.modules if _m.startswith('polyphage')]:
    del sys.modules[_m]

sha = subprocess.run(
    ["git", "-C", "/content/pp_src", "rev-parse", "--short", "HEAD"],
    capture_output=True, text=True,
).stdout.strip()
print(f"polyphage code : {{sha}}  (ref: {{POLYPHAGE_REF}})")
print()
print(f"Record `polyphage @ {{sha}}` with any result from this session. It is")
print("the only thing that makes a run identifiable after the fact.")
print("ready")
"""),
    md("""
> **Do not use Colab's *Save a copy to GitHub* on this notebook.** It commits
> your in-session copy over the repository's, which is how commit `c78ce43`
> reverted the notebook on `main`. The notebook is generated: edit the modules
> or `build_notebook.py`, run `python3 build_notebook.py`, and commit that.
"""),
    md("## Cell 3 — Fetch and parse the structure"),
    code("""
import urllib.request, shutil

# 6EQE = wild-type IsPETase (0.92 A).  7SH6 = FAST-PETase (reference only).
for pdb in ["6EQE", "7SH6"]:
    urllib.request.urlretrieve(f"https://files.rcsb.org/download/{pdb}.pdb",
                               f"inputs/pdbs/{pdb}.pdb")
    print("downloaded", pdb)

shutil.copy("inputs/pdbs/6EQE.pdb", "inputs/design_target/6EQE.pdb")
"""),
    code("""
!python helper_scripts/parse_multiple_chains.py \\
  --input_path=inputs/design_target/ \\
  --output_path=outputs/parsed_6EQE.jsonl

!python helper_scripts/make_fixed_positions_dict.py \\
  --input_path=outputs/parsed_6EQE.jsonl \\
  --output_path=outputs/fixed_6EQE.jsonl \\
  --chain_list "A" \\
  --position_list "160 206 237"
"""),
    md("""
## Cell 4 — Load the target (GATE)

`load_target` raises if the triad is not SER/ASP/HIS at 160/206/237, or if
Biopython and ProteinMPNN disagree about the sequence residue by residue.
If it raises, **stop** -- every residue number downstream would be wrong.
"""),
    code("""
from polyphage import load_target

target = load_target(
    pdb_path="inputs/design_target/6EQE.pdb",
    parsed_jsonl="outputs/parsed_6EQE.jsonl",
    chain="A",
)
print("length:", len(target.wildtype_seq))
print("modeled residues:", target.residue_numbers[0], "-", target.residue_numbers[-1])
print("mutable positions:", target.n_mutable)
print("frozen (triad):", [(rn, target.wildtype_seq[i])
                          for rn, i in zip(target.fixed_residue_numbers, target.fixed_idx)])
"""),
    md("""
## Cell 5 — Scorers

`BatchScorer` loads the model and featurizes the backbone once, then scores a
whole population per forward pass. `SubprocessScorer` is ProteinMPNN's own
`--score_only` path -- slow, but it is the reference the fast one is checked against.
"""),
    code("""
from polyphage import BatchScorer, SubprocessScorer, check_equivalence

batch = BatchScorer(mpnn_dir=".", pdb_path="inputs/design_target/6EQE.pdb", chain="A")
ref   = SubprocessScorer(mpnn_dir=".",
                         parsed_jsonl="outputs/parsed_6EQE.jsonl",
                         fixed_jsonl="outputs/fixed_6EQE.jsonl")
print("scorers built")
"""),
    md("""
## Cell 6 — Equivalence check (GATE)

The one check that matters before scaling. If this raises, the `tied_featurize`
indices in `BatchScorer.__init__` need re-checking against this clone's
`protein_mpnn_utils.py` -- ProteinMPNN's return tuple is positional, so an
upgrade can reshuffle it silently.
"""),
    code("""
check_equivalence(batch, ref, target.wildtype_seq)
"""),
    md("Time the speedup while you are here -- worth a line in the report."),
    code("""
import time
t0 = time.time(); ref.score(target.wildtype_seq);        t_sub = time.time() - t0
t0 = time.time(); batch.score([target.wildtype_seq]*8);  t_bat = time.time() - t0
print(f"subprocess, 1 seq: {t_sub:.1f}s   batch, 8 seqs: {t_bat:.2f}s")
"""),
    md("""
## Cell 7 — Small run

Confirm the loop before spending compute. What to look at, in order:

1. **`n_mut` per variant** -- must be <= `max_mutations` and mostly single digits.
   Anything in the hundreds means the constraint or the sampling is not taking effect.
2. **`known hotspot positions hit`** -- overlap with FAST-PETase / DuraPETase positions.
3. **`EXACT literature mutations`** -- same position *and* same amino acid. Rare, and
   the single most citable thing a run can produce.
"""),
    code("""
import time
from polyphage import (foldability_objective, novelty_objective, run,
                       report_front, print_front, log_run, summarize)

objectives = [foldability_objective(batch), novelty_objective(target)]
names = [o.name for o in objectives]

config = dict(pop_size=12, n_gen=5, max_mutations=15, seed=1)
t0 = time.time()
res = run(target, objectives, **config)
wall = time.time() - t0

reports = report_front(target, res, names)
print_front(reports)
print(summarize(reports))
log_run(config, reports, wall)
"""),
    md("""
### Cell 7b — Reproducibility (GATE)

The same seed and config must give the same front. This was broken until the
`fix/reproducibility-and-thermompnn` branch: the operators read a `seed` key
pymoo never passes, fell back to OS entropy, and every run differed while still
logging `"seed": 1`. Any result produced before that fix cannot be regenerated.

Cheap to check, so check it rather than assuming — and note that a stale
`polyphage` module cached by Colab is the usual reason this fails after an edit
(Runtime > Restart session).
"""),
    code("""
a = run(target, objectives, pop_size=12, n_gen=5, max_mutations=15,
        seed=7, verbose=False)
b = run(target, objectives, pop_size=12, n_gen=5, max_mutations=15,
        seed=7, verbose=False)

import numpy as np
fa, fb = np.atleast_2d(a.F), np.atleast_2d(b.F)
assert fa.shape == fb.shape, f"same seed gave {fa.shape[0]} vs {fb.shape[0]} variants"
np.testing.assert_allclose(np.sort(fa, axis=0), np.sort(fb, axis=0))
print(f"reproducible: two seed=7 runs both gave {fa.shape[0]} identical variants")
"""),
    md("""
## Cell 8 — Plot the front

A front that is one point, or a dead-straight line, means the two objectives are
not actually in conflict -- fix that before adding more.
"""),
    code("""
from polyphage import plot_front

wt_objs = {
    "mpnn_nll": float(batch.score([target.wildtype_seq])[0]),
    "neg_n_mutations": 0.0,
}
fig, ax = plot_front(reports, wildtype_objectives=wt_objs)
fig.savefig("pareto_front.png", dpi=150)
"""),
    md("""
## Cell 9 — Scale up

Only after Cells 7 and 8 look sane.

`max_mutations=10` on purpose: same regime as DuraPETase (10) and FAST-PETase (5),
so the comparison is like-for-like.
"""),
    code("""
config = dict(pop_size=40, n_gen=25, max_mutations=10, seed=1)
t0 = time.time()
res = run(target, objectives, **config)
wall = time.time() - t0

reports = report_front(target, res, names)
print_front(reports)
log_run(config, reports, wall)
"""),
    md("""
## Cell 10 — Save the run log

Colab wipes the filesystem. Download this every session: the run-to-run
progression is the spine of the report and cannot be reconstructed later.
"""),
    code("""
from google.colab import files
files.download('runs.jsonl')
files.download('pareto_front.png')
"""),
    md("""
---

# Part 2 — Is it real, and is it stable?

Two questions the run above cannot answer:

1. **Was T140D luck?** 14 known hotspot positions among 262 mutable ones means a
   9-mutation variant hits *some* hotspot about 39% of the time by chance, so
   position-level overlap proves almost nothing. An *exact* position-and-residue
   match runs at about 2.5%. Cell 11 measures both against an explicit null.
2. **Does any of this survive heat?** The ProteinMPNN score says "this sequence
   fits this backbone". It says nothing about temperature. Cells 12-15 add a
   real stability objective.
"""),
    md("""
## Cell 11 — Randomization control (the validity check)

Generates thousands of random variants with the same mutation count, and asks
how often chance alone reproduces what the search found.

Read the `p` values: `p < 0.05` means chance reproduces the result less than 5%
of the time. The exact-match p-value is the one that matters -- position-level
overlap is weak evidence on its own.
"""),
    code("""
from polyphage import run_control

control = run_control(target, reports, n_samples=5000)
"""),
    md("""
## Cell 12 — Install ThermoMPNN

[ThermoMPNN](https://github.com/Kuhlman-Lab/ThermoMPNN) (Kuhlman Lab, MIT licence)
predicts **ddG** -- the change in folding free energy when you mutate a residue.
**Negative ddG = more stable.** It is built on ProteinMPNN's own embeddings, so
it shares the backbone already loaded here.
"""),
    code("""
%cd /content
!git clone https://github.com/Kuhlman-Lab/ThermoMPNN.git
!pip install -q pytorch-lightning omegaconf
%cd /content/ProteinMPNN
print("cloned")
"""),
    md("""
## Cell 13 — Confirm the interface (do not skip)

The ThermoMPNN README does not document `custom_inference.py`, and this project
has already lost time to confidently-guessed identifiers that did not exist.
Two things trip this up, both already handled in `StabilityOracle.build`:

* The script is in `analysis/`, not at the repo root.
* ThermoMPNN ships its own `datasets.py` defining `Mutation`, which collides
  with the installed HuggingFace `datasets` package. `PYTHONPATH` must carry the
  repo root or the import fails.

Expected flags: `--pdb --chain --model_path --out_dir`. If this checkout differs,
pass the extras through `extra_args` in Cell 14.
"""),
    code("""
!find /content/ThermoMPNN -name "custom_inference.py"
!ls /content/ThermoMPNN/models
!PYTHONPATH=/content/ThermoMPNN python \\
    /content/ThermoMPNN/analysis/custom_inference.py -h
"""),
    md("""
## Cell 14 — Build the ddG table (once)

`custom_inference.py` runs *site-saturation mutagenesis*: every position against
every amino acid, in one pass. Cached as a table, so scoring a variant inside the
optimiser is just lookups -- no model calls in the loop.
"""),
    code("""
from polyphage import StabilityOracle

oracle = StabilityOracle(
    thermompnn_dir="/content/ThermoMPNN",
    pdb_path="inputs/design_target/6EQE.pdb",
    target=target,
    chain="A",
)
oracle.build()   # pass extra_args=[...] if Cell 13 showed different flags

print("\\nMost stabilising single mutations ThermoMPNN predicts:")
for rn, aa, ddg in oracle.best_stabilising(15):
    wt = target.wildtype_seq[target.residue_numbers.index(rn)]
    print(f"  {wt}{rn}{aa}  ddG={ddg:+.3f}")
"""),
    md("""
Worth a look before optimising: do any of those top single mutations coincide
with the FAST-PETase set (S121E, D186H, R224Q, N233K, R280A) or the DuraPETase
set (117, 119, 140, 159, 165, 168, 180, 188, 214, 280)?

**They do, strongly.** On 6EQE, ThermoMPNN places 12 of the 14 known mutations
in the top quartile of 5035 scored substitutions (mean percentile 16.6 against a
null of 50; binomial p < 0.001 on the below-median count). N233K ranks at
percentile 0. Full numbers in `results/thermompnn_validation.md`.

That cuts both ways. It validates ThermoMPNN as an objective — the model is not
noise, it independently rediscovers what two experimental groups found. But it
also means **T140D sits at percentile 18 in ThermoMPNN's own ranking**, so a
three-objective run that recovers T140D has shown little on its own. Cell 16c
is the control that accounts for this.
"""),
    code("""
from polyphage import FAST_PETASE, DURA_PETASE

top = oracle.best_stabilising(100)
for rn, aa, ddg in top:
    tags = []
    if FAST_PETASE.get(rn) == aa: tags.append("FAST-PETase")
    if DURA_PETASE.get(rn) == aa: tags.append("DuraPETase")
    if tags:
        print(f"  {rn}{aa}  ddG={ddg:+.3f}   <-- {', '.join(tags)}")
"""),
    md("""
### Cell 14c — Sign convention (GATE)

Nothing downstream would notice if ThermoMPNN's ddG sign were inverted: the
optimiser would quietly minimise toward the *least* stable variants it can find
and still report a tidy Pareto front.

This checks the sign against ground truth. FAST-PETase and DuraPETase mutations
were selected experimentally for thermostability, so under the documented
convention (negative = stabilising) they should sit below the median of all
scored mutations. **If this prints `FLIPPED?`, stop** and negate ddG in
`_parse` before running Cell 15.
"""),
    code("""
sign = oracle.check_sign_convention()
assert sign["verdict"] != "FLIPPED?", (
    "ThermoMPNN ddG sign looks inverted -- fix _parse before optimising"
)
"""),
    md("""
## Cell 15 — Three-objective run

Now the search optimises:

- `mpnn_nll` -- does it fold (lower better)
- `ddg` -- does it survive heat (lower better, negative = stabilising)
- `neg_n_mutations` -- how far from wildtype (fewer mutations is easier to test)

These genuinely conflict, which is the point: stabilising mutations often cost
foldability, and both usually cost mutation count. The Pareto front is the map of
those trade-offs.
"""),
    code("""
import time
from polyphage import foldability_objective, novelty_objective, stability_objective
from polyphage import run, report_front, print_front, log_run

objectives = [
    foldability_objective(batch),
    stability_objective(oracle),
    novelty_objective(target),
]
names = [o.name for o in objectives]

config = dict(pop_size=40, n_gen=25, max_mutations=10, seed=1)
t0 = time.time()
res = run(target, objectives, **config)
wall = time.time() - t0

reports = report_front(target, res, names)
print_front(reports)
log_run({**config, "objectives": names}, reports, wall)
"""),
    md("""
## Cell 16 — Control the new front

Two nulls, and they answer different questions. Report both.
"""),
    code("""
from polyphage import mpnn_prior_sampler, run_control

# Null 1: uniform random mutation. "Did the search beat chance?"
control = run_control(target, reports, n_samples=5000, label="uniform")
"""),
    md("""
### Cell 16b — The stricter null

The uniform null cannot answer the question a reviewer asks next: **did the
optimiser find this mutation, or does ProteinMPNN's prior already put mass on
it?** T140D only counts if the search located it. If the model assigns aspartate
high probability at 140 from the backbone alone, any method sampling from the
model would stumble on it, and the search deserves no credit.

This null keeps position selection uniform — the optimiser is not told which
positions to change either — but draws the replacement residue from the model's
own conditional distribution. A mutation that survives *this* was found by the
search, not handed to it.
"""),
    code("""
strict = mpnn_prior_sampler(target, batch)
control_strict = run_control(
    target, reports, n_samples=5000, aa_sampler=strict, label="ProteinMPNN-prior"
)
"""),
    md("""
### Cell 16c — The ddG null (only for three-objective runs)

Which null is honest depends on what the search was told. The ProteinMPNN-prior
null above is correct for a run whose only model is ProteinMPNN. Once ddG is an
objective, the search is being steered by ThermoMPNN, and **ThermoMPNN's prior
is what needs controlling for** — it already ranks T140D at percentile 18.

Run this instead of Cell 16b when `reports` came from the three-objective run.
Running it on a two-objective front would be wrong in the other direction:
penalising the search for a prior it never had access to.
"""),
    code("""
from polyphage import ddg_position_weights, ddg_prior_sampler

# A prior null has two halves. The sampler answers "given this position, which
# residue?"; the weights answer "which positions at all?". A ddG-driven search
# is told both, so a null that models only the first is too weak -- on 6EQE it
# enriched D186H just 1.6x despite ThermoMPNN ranking it at percentile 1.
ddg_null = ddg_prior_sampler(target, oracle)
ddg_pos = ddg_position_weights(target, oracle)

control_ddg = run_control(
    target, reports, n_samples=5000, aa_sampler=ddg_null,
    label="ThermoMPNN-ddG-prior (residue only)"
)
"""),
    md("""
## Cell 17 — Replication (this is the result)

One run recovering a known mutation is an anecdote. The reportable claim is
"the method recovers this mutation in k of n independent runs, 95% CI [lo, hi],
against a chance rate of p".

**This is the cell that produces the paper's headline number.** Budget for it:
20 seeds at the Cell 15 configuration is roughly 20 x 2.5 min on a T4. Start
with `SEEDS = range(1, 6)` to confirm the plumbing, then go to 20-50.

Watch the front-diversity line. If it warns, the fronts are near-duplicate
lineages and the variant count overstates how many distinct solutions the
search actually found — which is exactly what happened in this project's run 2,
where a two-member front shared seven of nine mutations.
"""),
    code("""
from polyphage import null_rate_per_run, print_replication, replicate

SEEDS = range(1, 21)   # start at range(1, 6) to smoke-test the plumbing

# Chance rate per run, measured rather than assumed. Match n_mutations to the
# mutation count of the variants you are actually comparing against.
null_rates = null_rate_per_run(target, n_mutations=9, n_samples=20000)

result = replicate(
    target, objectives, seeds=SEEDS,
    pop_size=40, n_gen=25, max_mutations=10,
    null_rate_per_run=null_rates,
)
print_replication(result)
"""),
    md("""
Run the same thing against the stricter null to get the second column of the
table. A mutation significant under the uniform null but not under this one was
supplied by ProteinMPNN's prior, not discovered by the search — and saying so
plainly is far stronger than being asked about it in review.
"""),
    code("""
from polyphage import print_replication, retest

# Score the SAME sweep against every null. retest() reuses the runs already in
# hand, so this costs seconds rather than another 45 minutes of search.
nulls = {
    "uniform": dict(),
    "ProteinMPNN-prior": dict(aa_sampler=strict),
    "ddG-prior (residue)": dict(aa_sampler=ddg_null),
    "ddG-prior (residue+position)": dict(aa_sampler=ddg_null, position_weights=ddg_pos),
}

for label, kw in nulls.items():
    rates = null_rate_per_run(target, n_mutations=9, n_samples=20000, **kw)
    scored = retest(result, rates)
    survivors = [r.mutation for r in scored.recoveries
                 if r.p_adjusted is not None and r.p_adjusted < 0.05]
    print(f"{label:<30} survives: {', '.join(survivors) or 'nothing'}")
"""),
    md("""
The last row is the one to report. A mutation that survives **residue+position**
was located by the search; one that survives only the weaker nulls was handed
over by ThermoMPNN's prior, and saying so is far stronger than being asked about
it in review.
"""),
    code("""
# Keep the sweep. Colab wipes its filesystem, and this cost ~45 minutes.
from polyphage import save_replication

save_replication(result, "replication.json")
from google.colab import files; files.download("replication.json")
"""),
    md("""
---

## What this does and does not show

**Does:** produce variants that a structure model considers foldable and a
stability model considers more heat-tolerant than wildtype, and recover mutations
that were independently confirmed in the lab.

**Does not:** prove any of these degrade PET faster, or at all. Nothing here
measures catalysis. ddG predicts folding stability, not activity, and the two
often trade against each other -- rigidifying a protein can stiffen the active
site it needs for chemistry.

## What is still missing for the full goal

The remaining gap is *activity* -- does it actually cut plastic. That needs a
second stage, because it requires a 3D structure per candidate rather than a
sequence:

1. **Fold** the top candidates with ESMFold or ColabFold.
2. **Check the triad survived** -- Ser160 OG to His237 NE2 should stay near 3 A.
   A variant that scores well but distorts the triad is dead on arrival.
3. **Dock** a PET oligomer (MHET/BHET analog) with AutoDock Vina or GNINA, and
   measure the distance from Ser160's oxygen to the ester carbonyl carbon it has
   to attack.

This is a screen over the top ~20 candidates, not something that can run inside
the loop -- folding each candidate takes seconds to minutes. That is the next
build.
"""),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 0,
    "metadata": {
        "colab": {"provenance": [], "name": "Polyphage.ipynb"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

out = HERE / "Polyphage.ipynb"
out.write_text(json.dumps(notebook, indent=1))
print(f"wrote {out}  ({len(cells)} cells)")
