# ThermoMPNN zero-shot recovery of known thermostabilising mutations

**Date:** 2026-10-18
**Code:** `polyphage` @ `3c544c7`
**Structure:** 6EQE chain A, 265 modelled residues (29–293), catalytic triad frozen
**Model:** ThermoMPNN `thermoMPNN_default.pt`, site-saturation scan via
`analysis/custom_inference.py`

## Result

ThermoMPNN's site-saturation scan ranks the experimentally confirmed
thermostabilising mutations of FAST-PETase and DuraPETase far above chance,
with no access to the papers that identified them.

| Statistic | Observed | Null | p |
|---|---|---|---|
| Known mutations below median ddG | 13 / 14 | 7 / 14 | 0.00092 (exact binomial) |
| Mean percentile | 16.6 | 50.0 | < 2.5e-6 (400k Monte Carlo draws) |
| In top quartile | 12 / 14 | 3.5 / 14 | — |
| In top decile | 6 / 14 | 1.4 / 14 | — |

Table size: **5035 entries** = 265 residues × 19 non-wildtype substitutions,
covering **100%** of mutable positions. Median ddG across all scored
substitutions: **+1.058** (most substitutions are destabilising, as expected).

## Raw data

Percentile is the mutation's rank within all 5035 scored substitutions; lower is
more stabilising. Negative ddG is stabilising — confirmed, not assumed, by this
table (see *Sign convention* below).

| Mutation | Source | ddG | Percentile |
|---|---|---|---|
| N233K | FAST-PETase | −1.198 | 0 |
| S180I | DuraPETase | −0.859 | 1 |
| D186H | FAST-PETase | −0.512 | 1 |
| S214H | DuraPETase | −0.373 | 3 |
| R280A | both | −0.069 | 8 |
| W117F | DuraPETase | −0.016 | 10 |
| S121E | FAST-PETase | +0.022 | 11 |
| **T140D** | DuraPETase | **+0.208** | **18** |
| A168R | DuraPETase | +0.238 | 19 |
| S188Q | DuraPETase | +0.265 | 20 |
| Q119Y | DuraPETase | +0.268 | 21 |
| R224Q | FAST-PETase | +0.378 | 25 |
| S165A | DuraPETase | +0.713 | 39 |
| S159H | DuraPETase | +1.250 | 57 |

Most stabilising substitutions overall (top 15): N225T (−1.861), L216R (−1.714),
Q127L/I/M/V, N225S, R260W/Y, **N233K (−1.198)**, N233R, N233A, A33V, C273F,
D283R. Note the clustering: Q127, N233, N225 and R260 each appear with multiple
favourable substitutions, suggesting position-level hotspots rather than
isolated point effects.

## Sign convention

Verified, not assumed. `StabilityOracle.check_sign_convention()` reports
**consistent**: 93% of known-stabilising mutations fall below the median, which
would be inverted if the sign were flipped. This matters because nothing else in
the pipeline would have caught an inverted sign — the optimizer would have
minimised toward the *least* stable variants while still producing a tidy Pareto
front.

## What this means for the project

**It validates ThermoMPNN as an objective.** The stability model is not noise:
it independently rediscovers what two experimental groups found. That is the
precondition for the three-objective search being worth running at all.

**It raises the bar for claiming the search works.** T140D sits at percentile 18
in ThermoMPNN's own ranking. A three-objective run that recovers T140D has
therefore demonstrated little on its own — the ddG objective was already
pointing at it. Any ddG-driven result must be tested against
`ddg_prior_sampler`, the null that samples replacement residues weighted by
ThermoMPNN's predictions, not against the uniform null alone.

The two-objective run that originally recovered T140D had no access to
ThermoMPNN, so its correct control would have been `mpnn_prior_sampler` — but
that run has since been **withdrawn**: it predated the reproducibility fix, and
re-running its configuration under the fixed code produces no exact literature
match at all. The percentile figures above stand regardless, because they
describe ThermoMPNN's ranking rather than anything the search did.

## Caveat worth reporting

S159H is the single known mutation ThermoMPNN ranks as destabilising (+1.250,
percentile 57). DuraPETase's ten mutations were selected as a *set*, so
individual ddG predictions need not reflect a residue's contribution in
combination — which is the same epistasis limitation that makes the additive ddG
sum a screening heuristic rather than a ground truth.

## Reproducing

Open the notebook, set `POLYPHAGE_REF`, run Cells 1–4, then 12 → 13 → 14 → 14b
→ 14c. Cell 14 must print `100.0% of mutable positions covered`; 88.9% means the
residue-numbering fix is absent and the table is misaligned by 29 residues.
