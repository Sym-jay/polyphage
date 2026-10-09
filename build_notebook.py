"""Generate Polyphage.ipynb with the polyphage package embedded as %%writefile cells.

Run: python3 build_notebook.py
Then upload the resulting Polyphage.ipynb to Colab (File > Upload notebook).

Embedding the modules means the notebook is self-contained: no Drive mount, no
GitHub clone, nothing to keep in sync by hand. Re-run this script after editing
any module to regenerate the notebook.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
MODULES = [
    "__init__.py", "encoding.py", "scoring.py", "optimize.py",
    "validate.py", "control.py", "stability.py",
]


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
    md("## Cell 1 — Environment"),
    code("""
!git clone https://github.com/dauparas/ProteinMPNN.git
%cd ProteinMPNN
!pip install -q biopython pymoo matplotlib
!mkdir -p polyphage outputs inputs/pdbs inputs/design_target

import sys
sys.path.insert(0, '/content/ProteinMPNN')
print('ready')
"""),
    md("""
## Cell 2 — Write the polyphage package

These cells write the project modules to disk. Nothing to upload or mount.
"""),
]

for name in MODULES:
    source = (HERE / "polyphage" / name).read_text().rstrip("\n")
    cells.append(code(f"%%writefile polyphage/{name}\n{source}"))

cells += [
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

1. **Was T140D luck?** 15 known hotspot positions among 262 mutable ones means a
   9-mutation variant hits *some* hotspot about 40% of the time by chance.
   Cell 11 measures this against an explicit null instead of guessing.
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
## Cell 13 — Discover the interface (do not skip)

The ThermoMPNN README does not document `custom_inference.py`'s flags, and this
project has already lost time to a confidently-guessed identifier that did not
exist. Print the real help text and read it before running anything.

If the flags differ from `--pdb` / `--chain` / `--out`, pass the right ones
through `extra_args` in the next cell, or tell me and I will patch
`StabilityOracle.build`.
"""),
    code("""
!python /content/ThermoMPNN/custom_inference.py -h
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
set (117, 119, 140, 159, 165, 168, 180, 188, 214, 280)? If ThermoMPNN already
ranks known-good mutations highly, say so in the report -- it means the search
starts from a well-informed prior.
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
## Cell 16 — Control the new front, then repeat across seeds

One run recovering a known mutation could be luck in the search itself, separate
from luck in the null. Several seeds recovering it is a property of the method.
"""),
    code("""
control = run_control(target, reports, n_samples=5000)
"""),
    code("""
from polyphage import repeat_seeds_summary

per_seed = {}
for seed in [1, 2, 3, 4, 5]:
    r = run(target, objectives, pop_size=40, n_gen=25, max_mutations=10,
            seed=seed, verbose=False)
    per_seed[seed] = report_front(target, r, names)
    print(f"seed {seed}: {len(per_seed[seed])} variants")

summary = repeat_seeds_summary(per_seed)
print("\\nrecovery across seeds:", summary["recovery_rate"])
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
