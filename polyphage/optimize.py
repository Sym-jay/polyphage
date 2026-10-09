"""NSGA-II over PETase sequences.

Design notes worth defending in the report:

* Amino-acid identity is **categorical**, not ordinal. pymoo's default SBX
  crossover and polynomial mutation assume a continuous, ordered variable --
  under them "index 5 (F) is between 4 (E) and 6 (G)", which is biochemically
  meaningless and biases the search toward whatever happens to sit mid-alphabet.
  This module uses uniform crossover and random-reset mutation instead.
* Mutation can also *revert* a position to wildtype. Without that, mutation
  count only ever ratchets upward and the low-mutation end of the Pareto front
  collapses.
* A hard constraint caps Hamming distance from wildtype, so candidates stay in
  the experimentally testable regime (real variants: 5-10 mutations) no matter
  how long the run goes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.mutation import Mutation
from pymoo.core.problem import Problem
from pymoo.core.sampling import Sampling
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.optimize import minimize

from .encoding import decode, hamming, wildtype_vector

N_AA = 20


class WildtypeBiasedSampling(Sampling):
    """Seed the population at wildtype, then perturb a handful of positions."""

    def __init__(self, target, min_mut: int = 1, max_mut: int = 8):
        super().__init__()
        self.target = target
        self.min_mut = min_mut
        self.max_mut = max_mut

    def _do(self, problem, n_samples, **kwargs):
        wt = wildtype_vector(self.target)
        X = np.tile(wt, (n_samples, 1))
        rng = np.random.default_rng(kwargs.get("seed"))
        for row in X:
            k = rng.integers(self.min_mut, self.max_mut + 1)
            where = rng.choice(len(row), size=k, replace=False)
            row[where] = rng.integers(0, N_AA, size=k)
        return X


class RandomResetMutation(Mutation):
    """Per-gene categorical mutation.

    With probability ``prob`` a gene is touched. A touched gene reverts to
    wildtype with probability ``revert_prob``, otherwise it is resampled
    uniformly over the 20 amino acids.
    """

    def __init__(self, target, prob: float = 0.01, revert_prob: float = 0.5):
        super().__init__()
        self.target = target
        self.gene_prob = prob
        self.revert_prob = revert_prob

    def _do(self, problem, X, **kwargs):
        X = X.astype(int).copy()
        wt = wildtype_vector(self.target)
        rng = np.random.default_rng(kwargs.get("seed"))

        touched = rng.random(X.shape) < self.gene_prob
        revert = touched & (rng.random(X.shape) < self.revert_prob)
        resample = touched & ~revert

        X[revert] = np.broadcast_to(wt, X.shape)[revert]
        X[resample] = rng.integers(0, N_AA, size=int(resample.sum()))
        return X


@dataclass
class ObjectiveSpec:
    """One scalar objective. ``fn`` takes a list of sequences, returns an array.

    All objectives are *minimized*; negate inside ``fn`` for things you want to
    maximize (as novelty does).
    """

    name: str
    fn: callable


class PETaseProblem(Problem):
    def __init__(self, target, objectives: list[ObjectiveSpec], max_mutations: int = 15):
        self.target = target
        self.objectives = objectives
        self.max_mutations = max_mutations
        super().__init__(
            n_var=target.n_mutable,
            n_obj=len(objectives),
            n_ieq_constr=1,
            xl=0,
            xu=N_AA - 1,
            vtype=int,
        )

    def _evaluate(self, X, out, *args, **kwargs):
        seqs = [decode(self.target, x) for x in np.atleast_2d(X)]
        out["F"] = np.column_stack([spec.fn(seqs) for spec in self.objectives])
        dists = np.array([hamming(self.target, s) for s in seqs], dtype=float)
        # g <= 0 is feasible
        out["G"] = (dists - self.max_mutations).reshape(-1, 1)


def foldability_objective(scorer) -> ObjectiveSpec:
    """ProteinMPNN negative log-likelihood. Lower is better, so minimize directly."""
    return ObjectiveSpec("mpnn_nll", lambda seqs: np.asarray(scorer.score(seqs), dtype=float))


def novelty_objective(target) -> ObjectiveSpec:
    """Mutation count. Maximized, hence negated.

    This is a weak objective -- a proxy for "explores further from wildtype",
    not a property anyone wants in an enzyme. It exists to give the front a
    second axis until a real one (pLDDT, ddG, docking) replaces it.
    """
    return ObjectiveSpec(
        "neg_n_mutations",
        lambda seqs: -np.array([hamming(target, s) for s in seqs], dtype=float),
    )


def run(
    target,
    objectives: list[ObjectiveSpec],
    pop_size: int = 8,
    n_gen: int = 3,
    max_mutations: int = 15,
    mutation_prob: float | None = None,
    seed: int = 1,
    verbose: bool = True,
):
    """Run NSGA-II. Returns the pymoo result object.

    ``mutation_prob`` defaults to roughly two touched genes per individual per
    generation, which keeps the search local on a ~265-residue protein.
    """
    if mutation_prob is None:
        mutation_prob = 2.0 / target.n_mutable

    problem = PETaseProblem(target, objectives, max_mutations=max_mutations)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=WildtypeBiasedSampling(target, max_mut=min(8, max_mutations)),
        crossover=UniformCrossover(prob=0.9),
        mutation=RandomResetMutation(target, prob=mutation_prob),
        eliminate_duplicates=True,
    )
    return minimize(
        problem, algorithm, ("n_gen", n_gen), seed=seed, verbose=verbose,
        save_history=True,
    )
