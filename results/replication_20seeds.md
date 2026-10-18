# Replication across 20 seeds, three-objective search

**Date:** 2026-10-18
**Code:** `polyphage` @ `c8d19f3`
**Config:** `pop_size=40, n_gen=25, max_mutations=10`, seeds 1–20
**Objectives:** `mpnn_nll` (ProteinMPNN), `ddg` (ThermoMPNN), `neg_n_mutations`
**Cost:** ~140 s per seed, ~46 min total on a T4

> **Status: complete.** D186H survives all four nulls, including the strongest
> one available — ThermoMPNN's ddG prior weighting both residue choice and
> position choice. No further control is outstanding. Read *Limits* before
> deciding how much weight this carries.

## Search behaviour

| | |
|---|---|
| Front size | mean 12.2, range 3–21 |
| Front diversity (mean pairwise Hamming) | 7.1 |
| Seeds recovering any exact literature mutation | 5/20 = 25%, 95% CI [11%, 47%] |

No diversity warning: fronts are genuinely distinct solutions rather than one
converged lineage, unlike the two-objective runs.

Aggregate against the uniform null: 5 of 20 seeds against a 2.51% per-run chance
rate gives **p = 1.1 × 10⁻⁴**. The equivalent figure under a prior null has not
been measured.

## Per-mutation recovery

Chance rates measured, not derived, at 20 000 null draws per sampler. `p` is
Holm-adjusted across the five mutations tested. The rightmost column is
ThermoMPNN's own zero-shot percentile from
[`thermompnn_validation.md`](thermompnn_validation.md) — lower means ThermoMPNN
already ranks it as more stabilising.

### Uniform null — "did the search beat chance?"

| Mutation | Found | Chance | p (Holm) | Verdict | ThermoMPNN pct |
|---|---|---|---|---|---|
| **FAST:186H** | 2/20 | 0.0018 | **0.0030** | survives | 1 |
| FAST:233K | 1/20 | 0.0014 | 0.1027 | no | 0 |
| Dura:168R | 1/20 | 0.0013 | 0.1027 | no | 19 |
| Dura:140D | 1/20 | 0.0015 | 0.1027 | no | 18 |
| Dura:165A | 1/20 | 0.0022 | 0.1027 | no | 39 |

### ProteinMPNN-prior null — "or did ProteinMPNN already favour it?"

| Mutation | Found | Chance | p (Holm) | Verdict |
|---|---|---|---|---|
| **FAST:186H** | 2/20 | 0.0060 | **0.0318** | survives |
| Dura:168R | 1/20 | 0.0043 | 0.3303 | no |
| FAST:233K | 1/20 | 0.0239 | 1.0000 | no |
| Dura:140D | 1/20 | 0.0239 | 1.0000 | no |
| Dura:165A | 1/20 | 0.0256 | 1.0000 | no |

### ThermoMPNN ddG-prior null — "or did the stability model?"

The control that matters for a three-objective run, since ddG is what steered
the search.

| Mutation | Found | Chance | p (Holm) | Verdict | Enrichment vs uniform |
|---|---|---|---|---|---|
| **FAST:186H** | 2/20 | 0.0029 | **0.0080** | survives | 1.6× |
| FAST:233K | 1/20 | 0.0042 | 0.2484 | no | 3.0× |
| Dura:168R | 1/20 | 0.0036 | 0.2484 | no | 2.8× |
| Dura:140D | 1/20 | 0.0032 | 0.2484 | no | 2.1× |
| Dura:165A | 1/20 | 0.0048 | 0.2484 | no | 2.2× |

### ThermoMPNN ddG prior, residue **and** position

The strongest null available: ThermoMPNN steers both which positions get
mutated and which residue goes there, matching what the three-objective search
was actually told.

| Mutation | Found | p (Holm) | Verdict |
|---|---|---|---|
| **FAST:186H** | 2/20 | **0.0339** | survives |
| Dura:165A | 1/20 | 0.1416 | no |
| Dura:168R | 1/20 | 0.1551 | no |
| Dura:140D | 1/20 | 0.1551 | no |
| FAST:233K | 1/20 | 0.2295 | no |

### Summary — D186H across all four nulls

| Null | p (Holm) | Implied chance/run | Enrichment vs uniform |
|---|---|---|---|
| Uniform | 0.0030 | 0.0018 | 1.0× |
| ProteinMPNN prior | 0.0318 | 0.0060 | 3.3× |
| ddG prior, residue only | 0.0077 | 0.0029 | 1.6× |
| **ddG prior, residue + position** | **0.0339** | 0.0062 | 3.4× |

**D186H survives every null, including the strongest.** Recovered in seeds 6 and
11; seed 6 also produced N233K.

Adding position weighting doubled the enrichment, 1.6× → 3.4×, which confirms
that the residue-only null was understating the comparison. It did not change
the verdict.

## The claim this supports

> A multi-objective structure-based search recovers D186H — an experimentally
> validated FAST-PETase thermostability mutation (Lu et al. 2022, *Nature*) — in
> 2 of 20 independent runs (10%, 95% CI [3%, 30%]), above chance under four
> progressively stricter nulls, including one matched to the stability model
> that steered the search (p = 0.034, Holm-corrected across five candidate
> mutations). The search had no access to the publication identifying it.

## Limits

State these alongside the claim; they bound it hard.

- **One mutation of fourteen.** Four other candidates appeared once each and
  none clears correction. The method recovers *some* known biology above
  chance, not the known set.
- **A 10% recovery rate with a wide interval.** [3%, 30%] at n = 20. Fifty seeds
  would tighten it; the point estimate may move.
- **Recovery is not design.** Rediscovering a published mutation says the search
  finds real signal. It says nothing about the *other* mutations on these
  fronts, which remain unvalidated candidates.
- **Nothing here measures catalysis.** ddG predicts folding stability. These
  variants are not shown to degrade PET faster, or at all.
- **One protein, one backbone.** 6EQE only. Whether this generalises is untested.
- **The null uses `n_mutations=9` while fronts span 4–10.** Chance rises with
  mutation count, so this is conservative rather than wrong — but it should be
  matched per variant before publication.

## Reading

**D186H is the only mutation with any claim.** Recovered in 2 of 20 independent
runs against a 0.18% chance rate, and still significant after controlling for
ProteinMPNN's prior and correcting across five tests.

**T140D remains withdrawn.** One run in twenty, and ProteinMPNN's prior alone
produces it at 2.4% per run — a 38% chance of at least one appearance across 20
runs. p = 1.0. This is consistent with the earlier finding that the original
T140D result was an artifact of unseeded RNG.

**Single-run recoveries carry no weight.** Four of the five mutations appeared
once. At n = 20 a single appearance cannot clear a corrected threshold against
any reasonable null, which is the point of running 20 seeds rather than one.

## The position half of the prior

*Resolved — kept because it explains why the final null differs from the third.*

The residue-only ddG null enriched D186H just **1.6×** over uniform, far less
than expected for a substitution ThermoMPNN ranks at percentile 1 of 5035. That
gap exposed a weakness in the control rather than a fact about the data.

`ddg_prior_sampler` normalises *within* a position: given that 186 is being
mutated, how likely is histidine? A global percentile says nothing about that —
if several substitutions at 186 score well, histidine is unremarkable among
them. So the null was modelling only half of what a ddG-driven search is told.
The other half is **which positions are worth touching at all**, and ThermoMPNN
answers that too.

`ddg_position_weights` supplies it, weighting each position by the best
stabilising substitution available there. The strongest honest null uses both:

```python
rates = null_rate_per_run(
    target, n_mutations=9, n_samples=20000,
    aa_sampler=ddg_prior_sampler(target, oracle),
    position_weights=ddg_position_weights(target, oracle),
)
print_replication(retest(result, rates))   # reuses the sweep, seconds not minutes
```

Running it raised D186H's implied chance rate from 0.0029 to 0.0062 — the
enrichment roughly doubled, confirming that position choice was the missing
half. D186H survived regardless, at p = 0.0339.

## Reproducing

```python
result = replicate(target, objectives, seeds=range(1, 21),
                   pop_size=40, n_gen=25, max_mutations=10,
                   null_rate_per_run=null_rate_per_run(target, n_mutations=9,
                                                       n_samples=20000))
save_replication(result, "replication.json")       # survives a dead runtime
print_replication(retest(result, other_null_rates))  # score against any null
```
