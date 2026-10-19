"""The no-search control.

If greedy ddG stacking recovers known mutations as well as the search does, the
search has to justify itself on another axis. These tests pin down that the
baseline is built correctly, so the comparison is fair in both directions.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.baselines import compare_to_front, greedy_ddg_variant
from polyphage.encoding import AA_ALPHABET, Target, mutations
from polyphage.validate import FAST_PETASE, VariantReport

SEQ_LEN = 60


class StubOracle:
    def __init__(self, table):
        self.table = table


def make_target() -> Target:
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=SEQ_LEN))
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq=seq,
        residue_numbers=list(range(200, 200 + SEQ_LEN)),   # spans 233
        fixed_residue_numbers=(), fixed_idx=[],
        mutable_idx=list(range(SEQ_LEN)),
    )


def flat_table(target, value=0.0):
    return {(rn, aa): value
            for rn in target.residue_numbers for aa in AA_ALPHABET}


def test_takes_the_requested_number_of_mutations():
    target = make_target()
    rng = np.random.default_rng(1)
    table = {(rn, aa): float(rng.normal())
             for rn in target.residue_numbers for aa in AA_ALPHABET}

    for k in (1, 5, 9, 15):
        assert greedy_ddg_variant(target, StubOracle(table), k).n_mutations == k


def test_never_picks_the_same_position_twice():
    """ThermoMPNN's raw top-N repeats positions; a variant cannot."""
    target = make_target()
    table = flat_table(target, 0.0)
    # One position with five excellent options -- a naive top-N would take them all.
    for aa, ddg in zip("DEFGH", [-9.0, -8.9, -8.8, -8.7, -8.6]):
        table[(233, aa)] = ddg

    v = greedy_ddg_variant(target, StubOracle(table), n_mutations=5)
    positions = [rn for rn, _, _ in mutations(target, v.sequence)]
    assert len(positions) == len(set(positions)) == 5


def test_picks_the_best_substitution_at_each_chosen_position():
    target = make_target()
    table = flat_table(target, 0.0)
    table[(233, "K")] = -5.0
    table[(233, "W")] = -1.0          # same position, worse

    v = greedy_ddg_variant(target, StubOracle(table), n_mutations=1)
    assert v.mutation_str.endswith("233K")


def test_never_mutates_to_wildtype():
    target = make_target()
    table = flat_table(target, 0.0)
    wt_at_233 = target.wildtype_seq[target.residue_numbers.index(233)]
    table[(233, wt_at_233)] = -99.0   # a "no-op" that must not be chosen

    v = greedy_ddg_variant(target, StubOracle(table), n_mutations=3)
    assert v.n_mutations == 3
    # every chosen mutation must actually change the residue
    for resnum, wt, mut in mutations(target, v.sequence):
        assert wt != mut
    # and position 233 must not have been picked at all, since its only
    # attractive option was a no-op
    assert 233 not in [rn for rn, _, _ in mutations(target, v.sequence)]


def test_it_finds_a_known_mutation_the_way_thermompnn_really_does():
    """The uncomfortable case: no search, and a literature hit falls out."""
    target = make_target()
    table = flat_table(target, 0.0)
    assert FAST_PETASE[233] == "K"
    table[(233, "K")] = -5.0

    v = greedy_ddg_variant(target, StubOracle(table), n_mutations=9)
    assert "FAST:233K" in v.exact_matches


def test_it_is_deterministic():
    target = make_target()
    rng = np.random.default_rng(2)
    table = {(rn, aa): float(rng.normal())
             for rn in target.residue_numbers for aa in AA_ALPHABET}
    o = StubOracle(table)
    assert (greedy_ddg_variant(target, o, 9).sequence
            == greedy_ddg_variant(target, o, 9).sequence)


def test_summed_ddg_is_the_sum_of_what_was_chosen():
    target = make_target()
    table = flat_table(target, 0.0)
    for rn, ddg in [(233, -3.0), (234, -2.0), (235, -1.0)]:
        table[(rn, "W")] = ddg

    v = greedy_ddg_variant(target, StubOracle(table), n_mutations=3)
    assert v.total_ddg == pytest.approx(-6.0)


def test_comparison_reports_both_sides():
    target = make_target()
    table = flat_table(target, 0.0)
    table[(233, "K")] = -5.0

    reports = [VariantReport(sequence=target.wildtype_seq, objectives={},
                             n_mutations=0, mutation_str="",
                             exact_matches=["Dura:140D"])]
    out = compare_to_front(target, StubOracle(table), reports)

    assert "FAST:233K" in out["baseline_matches"]
    assert out["front_matches"] == ["Dura:140D"]
    assert out["front_size"] == 1


def test_comparison_without_a_scorer_omits_foldability():
    target = make_target()
    out = compare_to_front(target, StubOracle(flat_table(target)), [])
    assert "mpnn_nll" not in out["baseline"].objectives
