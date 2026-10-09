"""Validation and reporting: does the search find what the literature already knows?

Hotspot overlap is the project's main internal validity check. It is not proof
the designs work -- it is evidence the search is finding biologically real
signal rather than noise, which is exactly what a purely computational project
can honestly claim.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .encoding import format_mutations, hamming, mutations

# Lu et al. 2022, Nature -- FAST-PETase, 5 mutations vs wildtype IsPETase.
FAST_PETASE = {121: "E", 186: "H", 224: "Q", 233: "K", 280: "A"}

# Cui et al. 2021 -- DuraPETase, 10 mutations.
DURA_PETASE = {
    117: "F", 119: "Y", 140: "D", 159: "H", 165: "A",
    168: "R", 180: "I", 188: "Q", 214: "H", 280: "A",
}

KNOWN_HOTSPOTS = set(FAST_PETASE) | set(DURA_PETASE)


@dataclass
class VariantReport:
    sequence: str
    objectives: dict[str, float]
    n_mutations: int
    mutation_str: str
    hotspot_positions: list[int] = field(default_factory=list)
    exact_matches: list[str] = field(default_factory=list)


def report_front(target, res, objective_names: list[str]) -> list[VariantReport]:
    """Turn a pymoo result into readable, checkable variant records."""
    from .encoding import decode

    X = np.atleast_2d(res.X)
    F = np.atleast_2d(res.F)
    reports = []

    for x, f in zip(X, F):
        seq = decode(target, x)
        muts = mutations(target, seq)
        positions = {rn for rn, _, _ in muts}

        exact = []
        for rn, _, mt in muts:
            for name, table in (("FAST", FAST_PETASE), ("Dura", DURA_PETASE)):
                if table.get(rn) == mt:
                    exact.append(f"{name}:{rn}{mt}")

        reports.append(
            VariantReport(
                sequence=seq,
                objectives=dict(zip(objective_names, (float(v) for v in f))),
                n_mutations=hamming(target, seq),
                mutation_str=format_mutations(target, seq),
                hotspot_positions=sorted(positions & KNOWN_HOTSPOTS),
                exact_matches=sorted(set(exact)),
            )
        )

    return sorted(reports, key=lambda r: tuple(r.objectives.values()))


def print_front(reports: list[VariantReport], max_mutations_shown: int = 20) -> None:
    print(f"{len(reports)} Pareto-optimal variants\n")
    for i, r in enumerate(reports, 1):
        objs = "  ".join(f"{k}={v:.4f}" for k, v in r.objectives.items())
        print(f"[{i}] {objs}  n_mut={r.n_mutations}")
        shown = r.mutation_str.split()
        if len(shown) > max_mutations_shown:
            print(f"    {' '.join(shown[:max_mutations_shown])} ... (+{len(shown) - max_mutations_shown} more)")
        else:
            print(f"    {r.mutation_str or '(wildtype)'}")
        if r.hotspot_positions:
            print(f"    known hotspot positions hit: {r.hotspot_positions}")
        if r.exact_matches:
            print(f"    EXACT literature mutations: {r.exact_matches}")
        print()


def summarize(reports: list[VariantReport]) -> dict:
    """One-line-per-run metrics. This is what goes in the run log."""
    n_muts = [r.n_mutations for r in reports]
    hit_positions = sorted({p for r in reports for p in r.hotspot_positions})
    exact = sorted({m for r in reports for m in r.exact_matches})
    return {
        "n_variants": len(reports),
        "n_mutations_min": min(n_muts) if n_muts else 0,
        "n_mutations_max": max(n_muts) if n_muts else 0,
        "hotspot_positions_hit": hit_positions,
        "n_hotspot_positions_hit": len(hit_positions),
        "exact_literature_mutations": exact,
    }


def log_run(
    config: dict,
    reports: list[VariantReport],
    wall_seconds: float,
    path: str | Path = "runs.jsonl",
) -> dict:
    """Append a run record. The progression across runs is the report narrative;
    it cannot be reconstructed later if runs are not logged as they happen."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": round(wall_seconds, 1),
        "config": config,
        "results": summarize(reports),
        "variants": [
            {"mutations": r.mutation_str, "objectives": r.objectives, "n_mut": r.n_mutations}
            for r in reports
        ],
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def plot_front(reports: list[VariantReport], wildtype_objectives: dict | None = None):
    """Scatter the first two objectives.

    A front that is a single point or a straight line means the objectives are
    not in genuine conflict and one of them is doing no work.
    """
    import matplotlib.pyplot as plt

    names = list(reports[0].objectives)
    if len(names) < 2:
        raise ValueError("need at least 2 objectives to plot a front")

    xs = [r.objectives[names[0]] for r in reports]
    ys = [r.objectives[names[1]] for r in reports]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(xs, ys, s=60, label="Pareto front", zorder=3)
    for r, x, y in zip(reports, xs, ys):
        ax.annotate(f"{r.n_mutations}", (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8)

    if wildtype_objectives:
        ax.scatter([wildtype_objectives[names[0]]], [wildtype_objectives[names[1]]],
                   marker="*", s=220, color="crimson", label="wildtype", zorder=4)

    ax.set_xlabel(names[0])
    ax.set_ylabel(names[1])
    ax.set_title("Polyphage Pareto front (labels = mutation count)")
    ax.legend()
    fig.tight_layout()
    return fig, ax
