"""The baselines a search has to beat to have been worth running.

The first question anyone should ask about an optimizer is what happens without
it. For this pipeline the answer is uncomfortable: ThermoMPNN's site-saturation
table already ranks N233K -- a FAST-PETase mutation -- inside its top ten
substitutions. Stacking the best-scoring positions recovers a known mutation
**deterministically**, in no compute at all, while the search recovers something
in 25% of runs.

On recovery rate alone, the baseline wins and the search is expensive
decoration. That comparison belongs in the paper whether or not it flatters the
method, because a reviewer will make it in the first paragraph.

The defence, and the claim actually worth testing, is that greedy stacking
optimises ddG *alone*. It has no reason to produce a sequence that still folds,
and summed ddG ignores epistasis, so a tall stack of individually-stabilising
mutations can be a protein that does not fold at all. The search balances ddG
against ProteinMPNN foldability and mutation count. If the greedy variant scores
badly on NLL where the Pareto front does not, the search is buying something
real -- and that is a stronger result than recovery rate.

``compare_to_front`` sets that comparison up; it does not decide it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .encoding import AA_ALPHABET, format_mutations, mutations
from .validate import exact_match_labels


@dataclass
class BaselineVariant:
    """A variant built without any search, for comparison against the front."""

    sequence: str
    n_mutations: int
    mutation_str: str
    exact_matches: list[str] = field(default_factory=list)
    objectives: dict[str, float] = field(default_factory=dict)
    total_ddg: float = 0.0


def greedy_ddg_variant(target, oracle, n_mutations: int = 9) -> BaselineVariant:
    """Stack the most stabilising substitutions ThermoMPNN predicts.

    One substitution per position -- the best available there -- then the
    ``n_mutations`` positions whose best substitution is most stabilising. Per
    position matters: ThermoMPNN's raw top-N list repeats positions (Q127 appears
    four times on 6EQE), and a variant cannot carry three residues at one site.

    This is the no-search control. It is deterministic, costs nothing, and the
    search has to beat it on *some* axis to have been worth running.
    """
    best_at_position: list[tuple[int, int, str, float]] = []
    for p, seq_idx in enumerate(target.mutable_idx):
        resnum = target.resnum(seq_idx)
        wt = target.wildtype_seq[seq_idx]
        options = [
            (aa, oracle.table[(resnum, aa)])
            for aa in AA_ALPHABET
            if aa != wt and (resnum, aa) in oracle.table
        ]
        if not options:
            continue
        aa, ddg = min(options, key=lambda kv: kv[1])
        best_at_position.append((seq_idx, resnum, aa, ddg))

    best_at_position.sort(key=lambda t: t[3])
    chosen = best_at_position[:n_mutations]

    seq = list(target.wildtype_seq)
    for seq_idx, _resnum, aa, _ddg in chosen:
        seq[seq_idx] = aa
    sequence = "".join(seq)

    matches: list[str] = []
    for resnum, _wt, mut in mutations(target, sequence):
        matches.extend(exact_match_labels(resnum, mut))

    return BaselineVariant(
        sequence=sequence,
        n_mutations=len(chosen),
        mutation_str=format_mutations(target, sequence),
        exact_matches=sorted(set(matches)),
        total_ddg=float(sum(t[3] for t in chosen)),
    )


def compare_to_front(
    target,
    oracle,
    reports,
    scorer=None,
    n_mutations: int = 9,
) -> dict:
    """Put the greedy baseline next to a Pareto front, on every axis.

    ``scorer`` is optional. Without it the comparison covers recovery and ddG;
    with a ``BatchScorer`` it also covers foldability, which is the axis where
    the search can actually win. Run it with one.
    """
    baseline = greedy_ddg_variant(target, oracle, n_mutations)

    if scorer is not None:
        baseline.objectives["mpnn_nll"] = float(scorer.score([baseline.sequence])[0])
        baseline.objectives["wildtype_nll"] = float(
            scorer.score([target.wildtype_seq])[0]
        )

    front_matches = sorted({m for r in reports for m in r.exact_matches})
    comparable = [
        r for r in reports
        if abs(r.n_mutations - baseline.n_mutations) <= 2
    ]

    return {
        "baseline": baseline,
        "baseline_matches": baseline.exact_matches,
        "front_matches": front_matches,
        "front_size": len(reports),
        "comparable_front_members": len(comparable),
    }


def print_comparison(comparison: dict) -> None:
    """Print it the way it has to appear in the paper: baseline first."""
    b = comparison["baseline"]

    print("Greedy ddG baseline -- no search, deterministic\n")
    print(f"  {b.n_mutations} mutations: {b.mutation_str}")
    print(f"  summed ddG : {b.total_ddg:+.3f}")
    if "mpnn_nll" in b.objectives:
        wt = b.objectives["wildtype_nll"]
        nll = b.objectives["mpnn_nll"]
        print(f"  mpnn_nll   : {nll:.4f}  (wildtype {wt:.4f}, "
              f"{'worse' if nll > wt else 'better'} by {abs(nll - wt):.4f})")
    print(f"  recovers   : {', '.join(b.exact_matches) or 'nothing'}")

    print(f"\nSearch front ({comparison['front_size']} variants, "
          f"{comparison['comparable_front_members']} at a comparable size)")
    print(f"  recovers   : {', '.join(comparison['front_matches']) or 'nothing'}")

    print()
    if b.exact_matches and not comparison["front_matches"]:
        print("  The baseline recovers what the search does not. Report that.")
    elif b.exact_matches:
        print("  Both recover known mutations, so recovery rate alone does not")
        print("  separate them. The claim has to rest on the other objectives:")
        print("  does the front hold foldability where the greedy stack loses it?")
    else:
        print("  The search recovers what greedy stacking does not, which is")
        print("  the cleanest version of this result.")
