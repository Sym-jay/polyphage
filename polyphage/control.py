"""Randomization control: is the hotspot overlap better than chance?

Run 2 recovered T140D, an exact DuraPETase mutation. That is only evidence if
random mutations of the same size do not recover it nearly as often. There are
**14** known hotspot positions among ~262 mutable ones -- FAST-PETase's 5 and
DuraPETase's 10, which share position 280 -- so a 9-mutation variant lands on
*some* hotspot position about 39% of the time by luck. Position-level overlap
is therefore close to worthless as evidence.

Matching the exact amino acid is far rarer: 14 target (position, residue) pairs
out of 262 x 19 possibilities puts a single mutation at ~0.28%, and a
9-mutation variant at roughly 2.5% of scoring any exact hit. That is the number
worth reporting. This module measures both against an explicit null rather than
relying on the estimate above.

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
from .validate import KNOWN_HOTSPOTS, VariantReport, exact_match_labels


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


def uniform_aa_sampler(target):
    """The default null's proposal: any amino acid but the wildtype one.

    Excluding wildtype matters -- otherwise a "mutation" could silently be a
    no-op, the null variants would carry fewer real mutations than the observed
    one, and the null would be too easy to beat.
    """
    wt_aa = [target.wildtype_seq[i] for i in target.mutable_idx]

    def sample(p: int, rng: np.random.Generator) -> str:
        choices = [a for a in AA_ALPHABET if a != wt_aa[p]]
        return choices[rng.integers(len(choices))]

    return sample


def mpnn_prior_sampler(target, scorer, temperature: float = 1.0):
    """A stricter null: replacement residues drawn from ProteinMPNN's own prior.

    The uniform null answers "did the search beat chance". It cannot answer the
    question a reviewer asks next: **did the optimizer find this mutation, or
    does ProteinMPNN's prior already put mass on it?** T140D is only interesting
    if the search located it; if the model assigns aspartate high probability at
    140 from the backbone alone, then any method that samples from the model
    would stumble on it and the search deserves no credit.

    This proposal keeps position selection uniform -- the optimizer is not told
    which positions to change either -- but draws the replacement residue from
    the model's conditional distribution at that position. A mutation that
    survives *this* null was found by the search, not handed to it.

    ``temperature`` > 1 flattens the model's distribution, < 1 sharpens it. 1.0
    uses the model as-is, which is the comparison worth reporting.
    """
    logp = scorer.position_log_probs(target.wildtype_seq)  # (L, 21)
    wt_aa = [target.wildtype_seq[i] for i in target.mutable_idx]

    # Restrict to the 20 real amino acids in this module's alphabet order, drop
    # the wildtype residue, and renormalise.
    from .scoring import MPNN_ALPHABET

    cols = [MPNN_ALPHABET.index(a) for a in AA_ALPHABET]
    tables: list[tuple[list[str], np.ndarray]] = []
    for p, seq_idx in enumerate(target.mutable_idx):
        row = np.asarray(logp[seq_idx])[cols] / temperature
        probs = np.exp(row - row.max())
        keep = [j for j, a in enumerate(AA_ALPHABET) if a != wt_aa[p]]
        probs = probs[keep]
        total = probs.sum()
        probs = probs / total if total > 0 else np.full(len(keep), 1 / len(keep))
        tables.append(([AA_ALPHABET[j] for j in keep], probs))

    def sample(p: int, rng: np.random.Generator) -> str:
        choices, probs = tables[p]
        return choices[rng.choice(len(choices), p=probs)]

    return sample


def sample_random_variants(
    target,
    n_mutations: int,
    n_samples: int,
    rng: np.random.Generator,
    aa_sampler=None,
) -> tuple[np.ndarray, np.ndarray, list[list[str]]]:
    """Draw null variants.

    Returns (hotspot position hits, exact-match counts, exact-match labels per
    sample). Positions are drawn without replacement over mutable positions;
    the replacement residue comes from ``aa_sampler``, which defaults to the
    uniform non-wildtype proposal. Pass ``mpnn_prior_sampler`` for the stricter
    model-prior null.
    """
    if aa_sampler is None:
        aa_sampler = uniform_aa_sampler(target)

    mutable_resnums = np.array([target.resnum(i) for i in target.mutable_idx])
    hotspot_mask = np.array([rn in KNOWN_HOTSPOTS for rn in mutable_resnums])

    pos_hits = np.zeros(n_samples, dtype=int)
    exact_hits = np.zeros(n_samples, dtype=int)
    labels_per_sample: list[list[str]] = []

    n_mutable = len(mutable_resnums)
    for s in range(n_samples):
        picks = rng.choice(n_mutable, size=n_mutations, replace=False)
        pos_hits[s] = int(hotspot_mask[picks].sum())

        labels: list[str] = []
        for p in picks:
            new_aa = aa_sampler(int(p), rng)
            labels.extend(exact_match_labels(int(mutable_resnums[p]), new_aa))
        labels = sorted(set(labels))
        labels_per_sample.append(labels)
        exact_hits[s] = len(labels)

    return pos_hits, exact_hits, labels_per_sample


def null_rate_per_run(
    target,
    n_mutations: int,
    n_samples: int = 5000,
    seed: int = 0,
    aa_sampler=None,
) -> dict[str, float]:
    """Per-mutation probability that one *chance* variant contains it.

    Feeds ``replicate.replicate``, which turns these into exact binomial
    p-values on an observed recovery rate across seeds. Measured rather than
    derived, so it stays correct when the proposal is not uniform.
    """
    rng = np.random.default_rng(seed)
    _, _, labels = sample_random_variants(
        target, n_mutations, n_samples, rng, aa_sampler=aa_sampler
    )
    counts: dict[str, int] = {}
    for sample_labels in labels:
        for label in sample_labels:
            counts[label] = counts.get(label, 0) + 1
    return {k: v / n_samples for k, v in counts.items()}


def compare_to_null(
    target,
    report: VariantReport,
    n_samples: int = 5000,
    seed: int = 0,
    aa_sampler=None,
) -> NullResult:
    """Empirical p-values for one variant's hotspot overlap.

    ``aa_sampler`` selects the null. Default is the uniform non-wildtype
    proposal; pass ``mpnn_prior_sampler(target, scorer)`` for the stricter
    model-prior null, and report both.
    """
    rng = np.random.default_rng(seed)
    k = report.n_mutations
    pos_hits, exact_hits, _ = sample_random_variants(
        target, k, n_samples, rng, aa_sampler=aa_sampler
    )

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
    target,
    reports: list[VariantReport],
    n_samples: int = 5000,
    seed: int = 0,
    aa_sampler=None,
    label: str = "uniform",
) -> list[NullResult]:
    """Control every variant on the front, and print a verdict."""
    results = [
        compare_to_null(target, r, n_samples, seed, aa_sampler=aa_sampler)
        for r in reports
    ]

    print(
        f"Randomization control, {label} null "
        f"({n_samples} random variants per mutation count)\n"
    )
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
