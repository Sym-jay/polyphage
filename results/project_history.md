# What changed, in order

A record of every result this project has produced, including the ones that were
withdrawn. Read it as a sequence: each entry exists because the one before it
turned out to be wrong, or incomplete, or untested.

---

## The starting point

| | |
|---|---|
| Protein | IsPETase, from *Ideonella sakaiensis* |
| Structure | **6EQE**, 0.92 Å (Austin et al. 2018, *PNAS*) |
| Chain | 265 modelled residues, PDB numbering **29–293** |
| Catalytic triad | **Ser160** (attacks), **His237** (deprotonates), **Asp206** (orients His237) |
| Frozen positions | those three, in every design |
| Mutable positions | 262 |
| Wild-type foldability | `mpnn_nll` **1.3779** — lower means the model likes it more |

**The problem.** PET only softens enough to attack near 65 °C. The enzyme falls
apart at 46–48 °C. It dies roughly twenty degrees before its substrate becomes
workable. Close that gap and you have something useful.

**The method.** ProteinMPNN scores whether a mutated sequence still fits the
backbone. ThermoMPNN predicts the stability change of each substitution. NSGA-II
searches sequence space under both, returning a Pareto front rather than one
answer.

**The validation idea.** Two laboratories already found heat-stable PETase
variants experimentally and published the mutations. The pipeline has no access
to those papers. So: hide the answers, run the search, and measure how often it
recovers them.

---

## 1. Run 1, pre-fix — the smoke test

12 individuals, 5 generations, 7.4 s. Six variants.

Every point on the front was worse than wild type on both axes, and the front
was nearly a straight line. By this project's own criterion that means the two
objectives are not in conflict and one of them is doing no work. It was never a
result; it was a check that the loop ran.

---

## 2. Run 2, pre-fix — the result that was withdrawn

40 individuals, 25 generations, 141 s.

```
n_mut=9    nll 1.3151    N37Q R53S T77K K95A T116A T140D Q228A R260F H293V
n_mut=10   nll 1.3303    N37Q R53S T77K K95A T116A Q228A R260F T270N S290A H293I
```

Both score better than wild type (1.3779), and the first contains **T140D** —
one of the ten mutations in DuraPETase, an experimentally confirmed heat-stable
variant. The search had never seen that paper.

This looked like the finding. It was not.

**Why it died.** The NSGA-II operators read `kwargs.get("seed")`, a key pymoo
has never passed. It returned `None`, `np.random.default_rng(None)` drew from
operating-system entropy, and every run was random while still recording
`"seed": 1` in the log. Two runs at the same seed returned fronts of 8 and 6
variants.

So this front could never be regenerated — not by us, not by anyone. **Withdrawn.**

---

## 3. Run 2, post-fix — the same configuration, a different answer

Same `pop_size=40, n_gen=25, max_mutations=10, seed=1`, now reproducible.

```
n_mut=8    nll 1.3248    A65S T88G T110V S124A Q182M R224S A240T D283K
```

Not one mutation in common with the withdrawn result. No literature match
anywhere on the front. This is the evidence that entry 2 was an artifact of the
bug rather than a finding, and it is why the withdrawal stands.

---

## 4. Run 3, post-fix — three objectives, and a real trade-off

Adding predicted stability as a third objective changed the shape of the search:

| | two objectives | three objectives |
|---|---|---|
| Variants on the front | 3 | **16** |
| Mean pairwise difference | 4.7 | **9.3** |
| Mutations shared by *every* variant | 6 | **1** |
| ddG range | — | −1.17 to −3.41 |

The two-objective front was one converged lineage wearing three faces. With
stability in play the front spans 4–10 mutations with only A179V universal.

The two ends:

```
best folding      nll 1.3607   ddG -1.172   A40D L117S M161L A179V Q182V N212S N233T S290Y
best stability    nll 1.3973   ddG -3.413   R53M Q91I T116C L117S S125R A179V A183N L216R E274W
```

---

## 5. Twenty seeds — the replication

One run recovering a mutation is an anecdote. Twenty independent runs give a
rate with an interval.

**D186H**, a confirmed FAST-PETase mutation, was recovered in seeds 6 and 11.

| Null model | What it concedes | p (Holm) |
|---|---|---|
| Uniform | nothing — pure chance | 0.0030 |
| ProteinMPNN prior | the folding model already liked it | 0.0318 |
| ThermoMPNN ddG prior, residue | the stability model already liked it | 0.0077 |
| ThermoMPNN ddG prior, residue **and** position | …and already liked that position | **0.0339** |

Four escalating controls, each granting more ground to the alternative
explanation. D186H survived all four. Four other candidates appeared once each
and cleared nothing.

Separately and independently: ThermoMPNN, with no search at all, ranks **12 of
the 14** known stabilising mutations in the top quartile of 5,035 possible
substitutions (p < 0.001). Contamination is ruled out on a size argument — its
training set caps at 72-residue domains, and IsPETase is 265.

---

## 6. The greedy baseline — and the result that reframed the project

The control nobody runs: what happens with **no search at all**?

Rank ThermoMPNN's table, keep the best substitution at each position, take the
top nine. One pass over a table that was computed anyway.

```
9 mutations   ddG -11.971   nll 1.3601   recovers FAST:233K
A33V T110I Q127L L216R N225T N233K R260W C273F D283R
```

| | foldability | stability | known mutations |
|---|---|---|---|
| **Greedy, 9 mutations** | **1.3601** | **−11.971** | **N233K** |
| Search, best foldability | 1.3607 | −1.172 | none |
| Search, best stability | 1.3973 | −3.413 | none |
| Wild type | 1.3779 | 0 | — |

**The baseline dominates all 16 front members** — better on both objectives, by
the same dominance test the search uses to build its own front. It recovers a
known mutation every single run; the search manages one in four.

The defence drafted in advance — that greedy stacking would sacrifice
foldability — did not survive the data. ProteinMPNN scores the greedy stack
**better than wild type**.

---

## The three bugs, each of which failed silently

| Bug | What it looked like | What it cost |
|---|---|---|
| Seed never reached pymoo's operators | runs logged `"seed": 1` and were random | every pre-fix result; T140D withdrawn |
| ThermoMPNN's 0-based index read as a PDB residue number | 88.9% coverage, which looked fine | every ddG attached 29 residues away |
| ddG sign never verified | a tidy Pareto front either way | the search could have been optimising toward the *least* stable variants |

None of these produced an error. Each produced a plausible, confident, wrong
number. That is why the controls exist.

---

## Where it stands

**Solid.** ThermoMPNN's zero-shot enrichment. The reproducibility fix. The
four-null ladder. The greedy baseline. The validation methodology is the
contribution, and it is what caught everything above.

**Not solid.** Any claim that the multi-objective search contributes anything.

**The open question.** Every objective so far measures folding or heat. None
measures whether the enzyme can still cut PET. The greedy baseline stacks nine
aggressive mutations including N233, adjacent to the catalytic His237 — so it
may be winning on paper by destroying the thing nobody is measuring. The
catalytic geometry screen (`polyphage/geometry.py`) exists to settle that, and
has not yet been run.
