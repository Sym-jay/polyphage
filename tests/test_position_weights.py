"""Position weighting: the half of the prior null the samplers miss.

``ddg_prior_sampler`` answers "given this position, which residue?".
``ddg_position_weights`` answers "which positions at all?". A ddG-driven search
is told both, so a null that only models the first is too weak -- which is not
hypothetical: it enriched D186H by 1.6x despite ThermoMPNN ranking it at
percentile 1 of 5035 substitutions.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.control import (
    ddg_position_weights,
    ddg_prior_sampler,
    null_rate_per_run,
    sample_random_variants,
)
from polyphage.encoding import AA_ALPHABET, Target

SEQ_LEN = 60


class StubOracle:
    def __init__(self, table):
        self.table = table


def make_target() -> Target:
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=SEQ_LEN))
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq=seq,
        residue_numbers=list(range(100, 100 + SEQ_LEN)),
        fixed_residue_numbers=(), fixed_idx=[],
        mutable_idx=list(range(SEQ_LEN)),
    )


def flat_table(target, value=0.0):
    return {(rn, aa): value
            for rn in target.residue_numbers for aa in AA_ALPHABET}


def test_weights_are_a_probability_distribution():
    target = make_target()
    w = ddg_position_weights(target, StubOracle(flat_table(target)))
    assert len(w) == len(target.mutable_idx)
    assert w.sum() == pytest.approx(1.0)
    assert (w >= 0).all()


def test_a_flat_table_gives_uniform_weights():
    target = make_target()
    w = ddg_position_weights(target, StubOracle(flat_table(target)))
    assert w.max() - w.min() < 1e-12


def test_a_position_with_a_stabilising_option_is_favoured():
    target = make_target()
    table = flat_table(target)
    table[(140, "D")] = -4.0            # one strongly stabilising option
    w = ddg_position_weights(target, StubOracle(table))

    idx = target.mutable_idx.index(target.residue_numbers.index(140))
    assert w[idx] > 10 * np.median(w)


def test_only_the_best_option_at_a_position_counts():
    """A position is worth targeting if *any* substitution there is good."""
    target = make_target()
    good = flat_table(target)
    good[(140, "D")] = -4.0

    many = flat_table(target)
    many[(140, "D")] = -4.0
    for aa in "EFGHIK":
        many[(140, aa)] = -3.9          # several near-equal options

    i = target.mutable_idx.index(target.residue_numbers.index(140))
    w_one = ddg_position_weights(target, StubOracle(good))[i]
    w_many = ddg_position_weights(target, StubOracle(many))[i]
    assert w_one == pytest.approx(w_many, rel=0.02)


def test_destabilising_positions_are_suppressed():
    target = make_target()
    table = flat_table(target)
    for aa in AA_ALPHABET:
        table[(140, aa)] = +4.0         # nothing good available here
    w = ddg_position_weights(target, StubOracle(table))

    idx = target.mutable_idx.index(target.residue_numbers.index(140))
    assert w[idx] < np.median(w) / 10


def test_temperature_flattens_the_weights():
    target = make_target()
    table = flat_table(target)
    table[(140, "D")] = -3.0

    sharp = ddg_position_weights(target, StubOracle(table), temperature=0.5)
    flat = ddg_position_weights(target, StubOracle(table), temperature=20.0)
    assert sharp.max() > flat.max()


def test_weighted_sampling_actually_shifts_the_positions_drawn():
    target = make_target()
    table = flat_table(target)
    table[(140, "D")] = -5.0
    w = ddg_position_weights(target, StubOracle(table))

    rng = np.random.default_rng(0)
    idx = target.mutable_idx.index(target.residue_numbers.index(140))

    hits = 0
    for _ in range(300):
        picks = rng.choice(len(w), size=9, replace=False, p=w)
        hits += idx in picks
    assert hits / 300 > 0.5             # uniform would give ~9/60 = 0.15


def test_both_halves_together_beat_either_alone():
    """The point of the whole exercise: position + residue > residue only."""
    target = make_target()
    table = flat_table(target)
    table[(140, "D")] = -5.0
    oracle = StubOracle(table)

    kw = dict(n_mutations=9, n_samples=1500, seed=1)
    uniform = null_rate_per_run(target, **kw)
    aa_only = null_rate_per_run(
        target, aa_sampler=ddg_prior_sampler(target, oracle), **kw)
    both = null_rate_per_run(
        target, aa_sampler=ddg_prior_sampler(target, oracle),
        position_weights=ddg_position_weights(target, oracle), **kw)

    label = "Dura:140D"
    assert both.get(label, 0) > aa_only.get(label, 0) > uniform.get(label, 0)


def test_position_weights_default_to_uniform():
    """Omitting them must not silently change an existing null."""
    target = make_target()
    rng_a = np.random.default_rng(3)
    rng_b = np.random.default_rng(3)
    a = sample_random_variants(target, 9, 200, rng_a)[0]
    b = sample_random_variants(target, 9, 200, rng_b, position_weights=None)[0]
    assert (a == b).all()
