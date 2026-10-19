# The greedy baseline dominates the search

**Date:** 2026-10-19
**Code:** `polyphage` @ `70e5d56`
**Compared against:** run 3 of [`runs.jsonl`](runs.jsonl), the three-objective
Pareto front (16 variants, `pop_size=40, n_gen=25, max_mutations=10, seed=1`)

> **This is a negative result about the method, and it is the most important
> thing in this repository.** The search, as configured, adds nothing over a
> heuristic that costs no compute at all.

## What the baseline is

Take ThermoMPNN's site-saturation table. At each position keep only the single
most stabilising substitution. Rank positions by that value. Take the top nine.
No search, no optimiser, deterministic, one pass over a table that was computed
anyway.

```
9 mutations: A33V T110I Q127L L216R N225T N233K R260W C273F D283R
summed ddG : -11.971
mpnn_nll   : 1.3601   (wild type 1.3779 -- better by 0.0177)
recovers   : FAST:233K
```

## The comparison

| | Foldability (`mpnn_nll`, lower better) | Stability (`ddg`, lower better) | Known mutations |
|---|---|---|---|
| **Greedy baseline, 9 mutations** | **1.3601** | **−11.971** | **N233K** |
| Search, best foldability (8 mut) | 1.3607 | −1.172 | none |
| Search, best stability (9 mut) | 1.3973 | −3.413 | none |
| Wild type | 1.3779 | 0 | — |

**The baseline dominates 16 of the 16 front members** — better or equal on every
objective and strictly better on at least one, which is the same dominance test
the search itself uses to build its front. Against the front's most stabilising
variant it is **3.5× more stabilising** while also folding better.

Like for like at nine mutations, the two front members are at ddG −1.434 and
−3.413; the baseline is at −11.971 with a better NLL than both.

The front also recovered no exact literature mutation in this run. The baseline
recovered one, deterministically.

## The defence that failed

The argument for the search was that greedy stacking optimises ddG *alone*, so
it has no reason to produce a sequence that still folds. On this data that is
simply not what happens: ProteinMPNN scores the greedy stack **better than wild
type** and marginally better than the best variant the search found. The
foldability objective did not need defending, because the greedy set did not
sacrifice it.

## Why the search probably loses

Stated as hypotheses, not findings:

- **The budget is tiny.** 40 individuals x 25 generations is 1,000 evaluations
  over a 262-position, 20-letter categorical space. That is not a search so much
  as a sample.
- **The operators are local.** Sampling starts near wild type and mutation
  touches roughly two genes per individual per generation. Reaching a specific
  nine-position combination from there in 25 generations is implausible.
- **`novelty_objective` rewards mutation count regardless of quality**, so some
  of the front's spread is bought with mutations that do nothing.

Each is testable. A larger budget, a greedy-seeded initial population, or
dropping the novelty objective would each change the answer, and none has been
tried.

## Caveats, in both directions

**Against the baseline.** Summed ddG assumes no epistasis, and ThermoMPNN is
trained on single point mutants. A −11.971 stack of nine "best" substitutions is
very likely a large overestimate, and the baseline exploits exactly the regime
where the additive assumption is least trustworthy. The search shares the same
flaw but pushes it less hard.

**Also against the baseline.** The greedy set includes N233, adjacent to the
catalytic His237. Nothing here measures catalysis, so an active-site disruption
would not show up in either method's objectives.

**Against the search.** None of the above rescues it. Both methods optimise the
same objectives under the same assumptions, and on those objectives the baseline
wins outright.

## What this does and does not change

**Unaffected:** the ThermoMPNN zero-shot result
([`thermompnn_validation.md`](thermompnn_validation.md)) — that measures the
stability model, not the search. The reproducibility fix, the four-null ladder
and this baseline are the methodological contribution, and this result is what
they were built to find.

**Weakened:** every claim that the multi-objective search contributes. D186H's
2-of-20 recovery ([`replication_20seeds.md`](replication_20seeds.md)) is still
true as measured, but it now sits beside a baseline that recovers a known
mutation *every single time* for free.

**The honest headline is no longer "the search recovers known biology."** It is
that a trivial ranking of a stability model's own output beats a multi-objective
evolutionary search on every axis the search was given, and that almost nobody
checks.

## Reproducing

```python
from polyphage import compare_to_front, print_comparison
comparison = compare_to_front(target, oracle, reports, scorer=batch, n_mutations=9)
print_comparison(comparison)
```
