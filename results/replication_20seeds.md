# Replication across 20 seeds, three-objective search

**Date:** 2026-10-18
**Code:** `polyphage` @ `c8d19f3`
**Config:** `pop_size=40, n_gen=25, max_mutations=10`, seeds 1–20
**Objectives:** `mpnn_nll` (ProteinMPNN), `ddg` (ThermoMPNN), `neg_n_mutations`
**Cost:** ~140 s per seed, ~46 min total on a T4

> **Status: incomplete.** The decisive control — the ThermoMPNN ddG-prior null —
> has not been run. See *What is still missing*. Do not cite the verdicts below
> as final.

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

## What is still missing

**This was a three-objective run, so the honest control is ThermoMPNN's ddG
prior, not ProteinMPNN's.** The ddG objective is what steered the search, and
`ddg_prior_sampler` exists precisely for this comparison. It has not been run.

The ThermoMPNN percentile column is why this matters. D186H sits at **percentile
1** and N233K at **percentile 0** of 5035 scored substitutions — the two
mutations recovered most are the two ThermoMPNN ranks highest. A ddG-weighted
null may well reproduce them at a rate that removes the D186H result.

Two outcomes are live, and both are reportable:

1. D186H survives the ddG null → the search contributes signal beyond the
   stability model's prior.
2. D186H does not → the recoveries are ThermoMPNN's prior being ferried through
   the search, which is a clean negative result about where the signal lives.

Until that cell runs, neither can be claimed.

**Minor:** the null rates use `n_mutations=9` while the front spans 4–10
mutations. Chance rises with mutation count, so this overstates the chance rate
for smaller variants and makes the test conservative rather than wrong. Worth
matching per variant before publication.

## Reproducing

```python
result = replicate(target, objectives, seeds=range(1, 21),
                   pop_size=40, n_gen=25, max_mutations=10,
                   null_rate_per_run=null_rate_per_run(target, n_mutations=9,
                                                       n_samples=20000))
save_replication(result, "replication.json")       # survives a dead runtime
print_replication(retest(result, other_null_rates))  # score against any null
```
