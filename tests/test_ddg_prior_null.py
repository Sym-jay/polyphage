"""The ddG-weighted null must follow ThermoMPNN's predictions.

If it does not, a three-objective result gets compared against the wrong
baseline and the optimizer is credited with what the stability model supplied.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.control import ddg_prior_sampler, null_rate_per_run
from polyphage.encoding import AA_ALPHABET, Target
from polyphage.validate import DURA_PETASE

SEQ_LEN = 60


class StubOracle:
    """Just the `.table` that ddg_prior_sampler reads."""

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
    return {
        (rn, aa): value
        for rn in target.residue_numbers
        for aa in AA_ALPHABET
    }


def test_sampler_never_returns_wildtype():
    target = make_target()
    sample = ddg_prior_sampler(target, StubOracle(flat_table(target)))
    rng = np.random.default_rng(0)
    for p in range(0, SEQ_LEN, 7):
        wt = target.wildtype_seq[target.mutable_idx[p]]
        for _ in range(20):
            assert sample(p, rng) != wt


def test_a_flat_table_reduces_to_uniform():
    """With no ddG signal the null must not secretly prefer anything."""
    target = make_target()
    sample = ddg_prior_sampler(target, StubOracle(flat_table(target)))
    rng = np.random.default_rng(0)
    draws = [sample(0, rng) for _ in range(4000)]
    counts = np.array([draws.count(a) for a in set(draws)], dtype=float)
    # 19 choices, 4000 draws -> ~210 each; allow generous slack.
    assert counts.min() > 120 and counts.max() < 320


def test_strongly_stabilising_residue_dominates():
    """Negative ddG is stabilising, so it must be the favoured draw."""
    target = make_target()
    table = flat_table(target, value=0.0)
    pos140 = target.residue_numbers.index(140)
    assert target.wildtype_seq[pos140] != "D"
    table[(140, "D")] = -5.0  # strongly stabilising

    sample = ddg_prior_sampler(target, StubOracle(table))
    p = target.mutable_idx.index(pos140)
    rng = np.random.default_rng(0)
    draws = [sample(p, rng) for _ in range(500)]
    assert draws.count("D") / len(draws) > 0.8


def test_sign_is_not_inverted():
    """A destabilising residue must be drawn far less often, not more."""
    target = make_target()
    table = flat_table(target, value=0.0)
    pos140 = target.residue_numbers.index(140)
    table[(140, "D")] = +5.0  # strongly DEstabilising

    sample = ddg_prior_sampler(target, StubOracle(table))
    p = target.mutable_idx.index(pos140)
    rng = np.random.default_rng(0)
    draws = [sample(p, rng) for _ in range(500)]
    assert draws.count("D") / len(draws) < 0.02


def test_temperature_flattens_the_distribution():
    target = make_target()
    table = flat_table(target, value=0.0)
    pos140 = target.residue_numbers.index(140)
    table[(140, "D")] = -3.0

    p = target.mutable_idx.index(pos140)
    sharp = ddg_prior_sampler(target, StubOracle(table), temperature=0.5)
    flat = ddg_prior_sampler(target, StubOracle(table), temperature=20.0)

    rng = np.random.default_rng(0)
    sharp_rate = [sharp(p, rng) for _ in range(500)].count("D") / 500
    flat_rate = [flat(p, rng) for _ in range(500)].count("D") / 500
    assert sharp_rate > flat_rate


def test_ddg_null_recovers_a_favoured_mutation_more_than_uniform():
    """The whole point: the two nulls must disagree about T140D."""
    target = make_target()
    table = flat_table(target, value=0.0)
    table[(140, "D")] = -4.0
    assert DURA_PETASE[140] == "D"

    uniform = null_rate_per_run(target, n_mutations=9, n_samples=800, seed=1)
    ddg = null_rate_per_run(
        target, n_mutations=9, n_samples=800, seed=1,
        aa_sampler=ddg_prior_sampler(target, StubOracle(table)),
    )
    assert ddg.get("Dura:140D", 0) > 5 * uniform.get("Dura:140D", 1e-9)


def test_missing_table_entries_are_treated_as_neutral():
    """A sparse table must not crash or silently favour unscored residues."""
    target = make_target()
    sample = ddg_prior_sampler(target, StubOracle({(140, "D"): -2.0}))
    rng = np.random.default_rng(0)
    p = target.mutable_idx.index(target.residue_numbers.index(140))
    draws = [sample(p, rng) for _ in range(300)]
    assert draws.count("D") / len(draws) > 0.2   # favoured
    assert len(set(draws)) > 1                   # but not the only option
