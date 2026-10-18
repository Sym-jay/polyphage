"""The randomization control and the stricter model-prior null.

``mpnn_prior_sampler`` normally needs ProteinMPNN and a GPU. It only consumes
``scorer.position_log_probs(seq) -> (L, 21)``, so a stub supplies that array and
the sampling logic is tested directly: the point under test is whether the null
draws from the distribution it was given, not whether ProteinMPNN is correct.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.control import (
    mpnn_prior_sampler,
    null_rate_per_run,
    sample_random_variants,
    uniform_aa_sampler,
)
from polyphage.encoding import AA_ALPHABET, Target
from polyphage.scoring import MPNN_ALPHABET
from polyphage.validate import DURA_PETASE, exact_match_labels

SEQ_LEN = 60
FIRST_RESNUM = 29


def make_target() -> Target:
    """A target whose residue numbers cover DuraPETase position 140."""
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=SEQ_LEN))
    # Place residue 140 inside the span so 140D is reachable.
    resnums = list(range(100, 100 + SEQ_LEN))
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq=seq, residue_numbers=resnums,
        fixed_residue_numbers=(), fixed_idx=[],
        mutable_idx=list(range(SEQ_LEN)),
    )


class StubScorer:
    """Returns a chosen (L, 21) log-prob array, like BatchScorer would."""

    def __init__(self, log_probs):
        self._lp = log_probs

    def position_log_probs(self, seq):
        return self._lp


def uniform_log_probs(target):
    return np.full((len(target.wildtype_seq), len(MPNN_ALPHABET)), np.log(1 / 21))


def peaked_log_probs(target, seq_idx: int, aa: str, mass: float = 0.95):
    """Uniform everywhere, except one position where the model strongly prefers `aa`.

    The remaining mass is spread over the other 20 slots, so `aa` really does
    hold `mass` of the distribution -- a flat baseline of 1/21 would leave the
    other residues summing to ~0.86 and swamp it.
    """
    lp = uniform_log_probs(target)
    rest = (1.0 - mass) / (len(MPNN_ALPHABET) - 1)
    lp[seq_idx, :] = np.log(rest)
    lp[seq_idx, MPNN_ALPHABET.index(aa)] = np.log(mass)
    return lp


def test_uniform_sampler_never_returns_wildtype():
    """A no-op 'mutation' would make the null easier to beat than it is."""
    target = make_target()
    sample = uniform_aa_sampler(target)
    rng = np.random.default_rng(0)
    for p in range(len(target.mutable_idx)):
        wt = target.wildtype_seq[target.mutable_idx[p]]
        for _ in range(20):
            assert sample(p, rng) != wt


def test_mpnn_prior_sampler_never_returns_wildtype():
    target = make_target()
    scorer = StubScorer(uniform_log_probs(target))
    sample = mpnn_prior_sampler(target, scorer)
    rng = np.random.default_rng(0)
    for p in range(0, len(target.mutable_idx), 7):
        wt = target.wildtype_seq[target.mutable_idx[p]]
        for _ in range(20):
            assert sample(p, rng) != wt


def test_mpnn_prior_sampler_follows_the_models_distribution():
    """A residue the model loves must dominate the null's draws.

    This is what makes the stricter null meaningful: if ProteinMPNN already
    puts its mass on aspartate at 140, the null recovers T140D constantly and
    the search gets no credit for it.
    """
    target = make_target()
    pos_140 = target.residue_numbers.index(140)
    assert target.wildtype_seq[pos_140] != "D"  # otherwise there is nothing to find
    lp = peaked_log_probs(target, pos_140, "D")

    sample = mpnn_prior_sampler(target, StubScorer(lp))
    p = target.mutable_idx.index(pos_140)
    rng = np.random.default_rng(0)
    draws = [sample(p, rng) for _ in range(400)]
    assert draws.count("D") / len(draws) > 0.8


def test_strict_null_recovers_a_favoured_mutation_far_more_often():
    """The uniform and model-prior nulls must give different answers."""
    target = make_target()
    pos_140 = target.residue_numbers.index(140)
    lp = peaked_log_probs(target, pos_140, "D")

    uniform = null_rate_per_run(target, n_mutations=9, n_samples=800, seed=1)
    strict = null_rate_per_run(
        target, n_mutations=9, n_samples=800, seed=1,
        aa_sampler=mpnn_prior_sampler(target, StubScorer(lp)),
    )

    label = "Dura:140D"
    assert DURA_PETASE[140] == "D"
    assert strict.get(label, 0) > 5 * uniform.get(label, 1e-9)


def test_sample_random_variants_labels_match_the_front_reporter():
    """The control must score the same event the observed front reports."""
    target = make_target()
    rng = np.random.default_rng(0)
    _, exact_hits, labels = sample_random_variants(target, 9, 200, rng)

    assert len(labels) == 200
    assert all(len(ls) == h for ls, h in zip(labels, exact_hits))
    for sample_labels in labels:
        for label in sample_labels:
            prefix, rest = label.split(":")
            resnum, aa = int(rest[:-1]), rest[-1]
            assert label in exact_match_labels(resnum, aa)


def test_null_rate_is_a_probability_per_run():
    target = make_target()
    rates = null_rate_per_run(target, n_mutations=9, n_samples=500, seed=0)
    assert all(0.0 <= v <= 1.0 for v in rates.values())


def test_more_mutations_raise_the_chance_rate():
    target = make_target()
    few = null_rate_per_run(target, n_mutations=2, n_samples=4000, seed=0)
    many = null_rate_per_run(target, n_mutations=20, n_samples=4000, seed=0)
    assert sum(many.values()) > sum(few.values())
