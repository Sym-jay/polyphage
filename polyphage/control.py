"""Randomization control: is the hotspot overlap better than chance?

Run 2 recovered T140D, an exact DuraPETase mutation. That is only evidence if
random mutations of the same size do not recover it nearly as often. With 15
known hotspot positions among ~262 mutable ones, a 9-mutation variant lands on
*some* hotspot position maybe 40% of the time by luck, so the position-level
overlap is weak evidence on its own. Matching the exact amino acid is much
rarer. This module measures both against an explicit null instead of guessing.

The null used here is uniform random mutation: same number of mutations, random
positions, random replacement amino acids. That is the honest baseline for the
claim "the search found these, not chance". A stricter null -- random variants
resampled from ProteinMPNN's own high-likelihood region -- would test the
stronger claim "the *optimizer* found these, not just the model's prior", and
is worth adding once this one is reported.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoding import AA_ALPHABET
from .validate import DURA_PETASE, FAST_PETASE, KNOWN_HOTSPOTS, VariantReport


@dataclass
class NullResult:
    n_mutations: int
    n_samples: int
    observed_positions: int
    observed_exact: int
    null_positions_mean: float
    null_exact_mean: float
    p_positions: float
    p_exact: float

    def __str__(self) -> str:
        return (
            f"{self.n_mutations} mutations, {self.n_samples} random draws\n"
            f"  hotspot positions hit : observed {self.observed_positions}, "
            f"chance {self.null_positions_mean:.2f}  ->  p = {self.p_positions:.4f}\n"
            f"  exact literature hits : observed {self.observed_exact}, "
            f"chance {self.null_exact_mean:.3f}  ->  p = {self.p_exact:.4f}"
        )


def _exact_tables() -> dict[int, str]:
    """Position -> amino acid, for every experimentally confirmed mutation."""
    combined = dict(DURA_PETASE)
    combined.update(FAST_PETASE)
    return combined


def sample_random_variants(
    target, n_mutations: int, n_samples: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Draw random variants; return (hotspot position hits, exact matches) counts.

    Mutations are drawn without replacement over mutable positions, and the
    replacement amino acid is never the wildtype one -- otherwise a "mutation"
    could silently be a no-op and the null would be too easy to beat.
    """
    exact = _exact_tables()
    mutable_resnums = np.array([target.resnum(i) for i in target.mutable_idx])
    wt_aa = np.array([target.wildtype_seq[i] for i in target.mutable_idx])

    hotspot_mask = np.array([rn in KNOWN_HOTSPOTS for rn in mutable_resnums])

    pos_hits = np.zeros(n_samples, dtype=int)
    exact_hits = np.zeros(n_samples, dtype=int)

    n_mutable = len(mutable_resnums)
    for s in range(n_samples):
        picks = rng.choice(n_mutable, size=n_mutations, replace=False)
        pos_hits[s] = int(hotspot_mask[picks].sum())

        n_exact = 0
        for p in picks:
            choices = [a for a in AA_ALPHABET if a != wt_aa[p]]
            new_aa = choices[rng.integers(len(choices))]
            if exact.get(int(mutable_resnums[p])) == new_aa:
                n_exact += 1
        exact_hits[s] = n_exact

    return pos_hits, exact_hits


def compare_to_null(
    target,
    report: VariantReport,
    n_samples: int = 5000,
    seed: int = 0,
) -> NullResult:
    """Empirical p-values for one variant's hotspot overlap."""
    rng = np.random.default_rng(seed)
    k = report.n_mutations
    pos_hits, exact_hits = sample_random_variants(target, k, n_samples, rng)

    obs_pos = len(report.hotspot_positions)
    obs_exact = len(report.exact_matches)

    # One-sided: how often does chance do at least as well as observed?
    # +1 in numerator and denominator keeps p away from exactly 0, which an
    # empirical estimate can never justify.
    p_pos = (np.sum(pos_hits >= obs_pos) + 1) / (n_samples + 1)
    p_exact = (np.sum(exact_hits >= obs_exact) + 1) / (n_samples + 1)

    return NullResult(
        n_mutations=k,
        n_samples=n_samples,
        observed_positions=obs_pos,
        observed_exact=obs_exact,
        null_positions_mean=float(pos_hits.mean()),
        null_exact_mean=float(exact_hits.mean()),
        p_positions=float(p_pos),
        p_exact=float(p_exact),
    )


def run_control(
    target, reports: list[VariantReport], n_samples: int = 5000, seed: int = 0
) -> list[NullResult]:
    """Control every variant on the front, and print a verdict."""
    results = [compare_to_null(target, r, n_samples, seed) for r in reports]

    print(f"Randomization control ({n_samples} random variants per mutation count)\n")
    for r, res in zip(reports, results):
        print(f"[{r.mutation_str}]")
        print(res)
        print()

    best_exact = min((r.p_exact for r in results), default=1.0)
    any_exact = any(r.observed_exact > 0 for r in results)

    if any_exact and best_exact < 0.05:
        print(
            f"VERDICT: exact literature matches are unlikely by chance "
            f"(best p = {best_exact:.4f}). This is a defensible result."
        )
    elif any_exact:
        print(
            f"VERDICT: exact matches found, but chance alone reproduces them at "
            f"p = {best_exact:.4f}. Report the match, but do not claim the method "
            "beat chance on this evidence -- run more generations or more seeds."
        )
    else:
        print(
            "VERDICT: no exact literature matches on this front. Position-level "
            "overlap alone is weak evidence; do not lead the report with it."
        )

    return results


def repeat_seeds_summary(per_seed_reports: dict[int, list[VariantReport]]) -> dict:
    """Aggregate hotspot recovery across several optimizer seeds.

    A single run recovering T140D could be luck in the search itself, separate
    from luck in the null. Running the same config under several seeds and
    reporting how many recover it is the cheapest way to show the result is a
    property of the method and not of seed 1.
    """
    found = {}
    for seed, reports in per_seed_reports.items():
        matches = {m for r in reports for m in r.exact_matches}
        found[seed] = sorted(matches)

    all_matches = sorted({m for ms in found.values() for m in ms})
    recovery = {
        m: sum(1 for ms in found.values() if m in ms) / len(found)
        for m in all_matches
    }
    return {"per_seed": found, "recovery_rate": recovery, "n_seeds": len(found)}
