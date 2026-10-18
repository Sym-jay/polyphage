"""Replication across seeds: turn a single anecdote into a recovery rate.

One run recovering T140D is not a result. It is one draw from a stochastic
search, and the only honest way to describe it is "the method recovers this
mutation in k of n independent runs, 95% CI [lo, hi]". That is what this module
produces.

Two things it measures that a single run cannot:

* **Recovery rate per known mutation**, with a Wilson confidence interval and a
  binomial test against the chance rate from ``control.py``. Wilson rather than
  the normal approximation because n is small (20-50 seeds) and k is often 0 or
  n, where the normal interval runs outside [0, 1] and is simply wrong.
* **Front diversity.** Run 2 of this project returned a two-member front whose
  members shared seven of nine mutations -- one converged lineage reported as
  two findings. Mean pairwise Hamming distance within a front makes that visible
  instead of leaving it to be noticed by eye.

Multiple testing matters here: there are 14 known mutations to recover, so
testing each one separately and reporting the smallest p-value would be
cheating. ``holm_adjust`` applies a Holm-Bonferroni correction, which controls
the family-wise error rate without assuming the tests are independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .encoding import hamming
from .validate import VariantReport, report_front

Z_95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Correct at the boundaries, unlike the normal approximation: k=0 gives
    [0, something], not a negative lower bound, which matters because most
    individual mutations will be recovered in zero runs.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return (max(0.0, center - half), min(1.0, center + half))


def binomial_tail(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p). One-sided, exact.

    Used to ask: if each run had only the chance probability ``p`` of recovering
    this mutation, how often would k or more of n runs recover it anyway?
    """
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0 if k > 0 else 1.0
    if p >= 1.0:
        return 1.0
    return float(
        sum(
            math.comb(n, i) * p**i * (1 - p) ** (n - i)
            for i in range(k, n + 1)
        )
    )


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni step-down adjustment.

    Controls the family-wise error rate across the mutations tested, without
    assuming independence between them (Bonferroni's assumption, which fails
    here because mutations co-occur within a variant).
    """
    if not pvalues:
        return {}
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (name, p) in enumerate(items):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)  # enforce monotonicity
        adjusted[name] = running
    return adjusted


@dataclass
class MutationRecovery:
    """How often one known mutation was recovered across independent runs."""

    mutation: str
    n_found: int
    n_seeds: int
    null_rate: float | None = None
    p_value: float | None = None
    p_adjusted: float | None = None

    @property
    def rate(self) -> float:
        return self.n_found / self.n_seeds if self.n_seeds else 0.0

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(self.n_found, self.n_seeds)


@dataclass
class ReplicationResult:
    seeds: list[int]
    front_sizes: list[int]
    front_diversity: list[float]
    recoveries: list[MutationRecovery]
    seeds_with_any_exact: int
    per_seed_matches: dict[int, list[str]] = field(default_factory=dict)

    @property
    def n_seeds(self) -> int:
        return len(self.seeds)


def front_diversity(target, reports: list[VariantReport]) -> float:
    """Mean pairwise Hamming distance between members of one Pareto front.

    Near zero means the population converged to a single lineage and the front's
    members are near-duplicates, so the number of variants on it overstates how
    many distinct solutions were found. A front with fewer than two members has
    no diversity to measure and returns 0.0.
    """
    seqs = [r.sequence for r in reports]
    if len(seqs) < 2:
        return 0.0
    dists = [
        sum(a != b for a, b in zip(seqs[i], seqs[j]))
        for i in range(len(seqs))
        for j in range(i + 1, len(seqs))
    ]
    return float(np.mean(dists))


def replicate(
    target,
    objectives,
    seeds,
    null_rate_per_run: dict[str, float] | None = None,
    **run_kwargs,
) -> ReplicationResult:
    """Run the search once per seed and aggregate what it recovered.

    ``objectives`` is reused across runs, so any scorer it closes over must be
    stateless with respect to the search -- which ``BatchScorer`` is, since it
    only reads the backbone it featurized at construction.

    ``null_rate_per_run`` optionally maps a mutation label to the probability
    that a single *chance* run recovers it, as measured by ``control.py``. When
    supplied, each mutation gets an exact binomial p-value and a Holm-adjusted
    one. Without it the recovery rates stand alone, which is still far more than
    a single run can support.
    """
    from .optimize import run as run_search

    names = [o.name for o in objectives]
    seeds = list(seeds)

    sizes, diversity, per_seed = [], [], {}
    for seed in seeds:
        res = run_search(target, objectives, seed=seed, verbose=False, **run_kwargs)
        reports = report_front(target, res, names)
        sizes.append(len(reports))
        diversity.append(front_diversity(target, reports))
        per_seed[seed] = sorted({m for r in reports for m in r.exact_matches})

    found_counts: dict[str, int] = {}
    for matches in per_seed.values():
        for m in matches:
            found_counts[m] = found_counts.get(m, 0) + 1

    n = len(seeds)
    recoveries = [
        MutationRecovery(mutation=m, n_found=k, n_seeds=n)
        for m, k in sorted(found_counts.items(), key=lambda kv: -kv[1])
    ]

    if null_rate_per_run:
        raw = {}
        for rec in recoveries:
            p0 = null_rate_per_run.get(rec.mutation)
            if p0 is None:
                continue
            rec.null_rate = p0
            rec.p_value = binomial_tail(rec.n_found, n, p0)
            raw[rec.mutation] = rec.p_value
        for rec in recoveries:
            if rec.mutation in raw:
                rec.p_adjusted = holm_adjust(raw)[rec.mutation]

    return ReplicationResult(
        seeds=seeds,
        front_sizes=sizes,
        front_diversity=diversity,
        recoveries=recoveries,
        seeds_with_any_exact=sum(1 for v in per_seed.values() if v),
        per_seed_matches=per_seed,
    )


def print_replication(result: ReplicationResult) -> None:
    """The table that belongs in the report."""
    n = result.n_seeds
    print(f"Replication across {n} independent seeds\n")

    sizes = np.array(result.front_sizes)
    div = np.array(result.front_diversity)
    print(f"  front size      : mean {sizes.mean():.1f}  range [{sizes.min()}, {sizes.max()}]")
    print(f"  front diversity : mean pairwise Hamming {div.mean():.1f}")
    if div.mean() < 2:
        print("    WARNING -- fronts are near-duplicate lineages, so the number of")
        print("    variants on a front overstates how many distinct solutions exist.")

    lo, hi = wilson_interval(result.seeds_with_any_exact, n)
    print(
        f"\n  seeds recovering any exact literature mutation: "
        f"{result.seeds_with_any_exact}/{n} "
        f"({result.seeds_with_any_exact / n:.0%}, 95% CI [{lo:.0%}, {hi:.0%}])"
    )

    if not result.recoveries:
        print("\n  No exact literature mutations recovered in any run.")
        print("  Report this plainly; it is a result, not a failed run.")
        return

    print(f"\n  {'mutation':<16}{'found':>8}{'rate':>8}{'95% CI':>18}{'p (Holm)':>12}")
    for rec in result.recoveries:
        lo, hi = rec.ci
        ci = f"[{lo:.2f}, {hi:.2f}]"
        p = "-" if rec.p_adjusted is None else f"{rec.p_adjusted:.4f}"
        print(
            f"  {rec.mutation:<16}{rec.n_found:>4}/{rec.n_seeds:<3}"
            f"{rec.rate:>8.2f}{ci:>18}{p:>12}"
        )

    if any(r.p_adjusted is not None for r in result.recoveries):
        print("\n  p is Holm-adjusted across the mutations tested. A mutation")
        print("  recovered in a minority of runs is still a real signal if its")
        print("  adjusted p is small -- the comparison is against chance, not 50%.")
