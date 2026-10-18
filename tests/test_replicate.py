"""Statistics behind the recovery-rate claim.

These are the numbers a reviewer will check, so they are tested against
closed-form values rather than against whatever the code happens to return.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from polyphage.encoding import AA_ALPHABET, Target
from polyphage.replicate import (
    binomial_tail,
    front_diversity,
    holm_adjust,
    wilson_interval,
)
from polyphage.validate import VariantReport


def test_wilson_stays_inside_unit_interval_at_the_boundaries():
    """Where the normal approximation goes negative, Wilson must not."""
    lo, hi = wilson_interval(0, 20)
    assert lo == 0.0 and 0 < hi < 1

    lo, hi = wilson_interval(20, 20)
    assert hi == 1.0 and 0 < lo < 1


def test_wilson_matches_published_value():
    """k=2, n=20 -> approximately [0.028, 0.301] at 95%."""
    lo, hi = wilson_interval(2, 20)
    assert lo == pytest.approx(0.028, abs=0.002)
    assert hi == pytest.approx(0.301, abs=0.002)


def test_wilson_interval_narrows_with_more_seeds():
    narrow = wilson_interval(10, 100)
    wide = wilson_interval(1, 10)
    assert (narrow[1] - narrow[0]) < (wide[1] - wide[0])


@pytest.mark.parametrize(
    "k,n,p",
    [(3, 10, 0.1), (1, 5, 0.5), (7, 7, 0.3), (2, 20, 0.025)],
)
def test_binomial_tail_matches_direct_summation(k, n, p):
    expected = sum(
        math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1)
    )
    assert binomial_tail(k, n, p) == pytest.approx(expected)


def test_binomial_tail_edges():
    assert binomial_tail(0, 10, 0.3) == 1.0        # P(X >= 0) is certain
    assert binomial_tail(11, 10, 0.3) == 0.0       # impossible
    assert binomial_tail(1, 10, 0.0) == 0.0        # chance rate of zero


def test_recovery_in_a_minority_of_runs_can_still_beat_chance():
    """The point of the binomial test: 3/20 is a lot when chance gives 2.5%.

    2.5% is this project's measured chance rate for a 9-mutation variant
    carrying any exact literature match. Recovering one in 3 of 20 runs is a
    15% rate, which looks unimpressive until compared against that baseline.
    """
    p = binomial_tail(3, 20, 0.025)
    assert p == pytest.approx(0.01296, abs=1e-5)
    assert p < 0.05

    # One run in twenty, by contrast, is roughly what chance alone produces.
    assert binomial_tail(1, 20, 0.025) > 0.3


def test_holm_is_monotone_and_never_below_bonferroni_on_the_smallest():
    raw = {"a": 0.001, "b": 0.02, "c": 0.4}
    adj = holm_adjust(raw)
    # smallest p is multiplied by the full family size
    assert adj["a"] == pytest.approx(0.003)
    # ordering preserved, values non-decreasing in raw p
    assert adj["a"] <= adj["b"] <= adj["c"]
    assert all(adj[k] >= raw[k] for k in raw)


def test_holm_enforces_monotonicity():
    """A later test must never get a smaller adjusted p than an earlier one."""
    raw = {"a": 0.04, "b": 0.045, "c": 0.05}
    adj = holm_adjust(raw)
    ordered = [adj[k] for k in sorted(raw, key=lambda k: raw[k])]
    assert ordered == sorted(ordered)


def test_holm_handles_empty_input():
    assert holm_adjust({}) == {}


def _target(seq_len=30):
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=seq_len))
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq=seq,
        residue_numbers=list(range(29, 29 + seq_len)),
        fixed_residue_numbers=(), fixed_idx=[],
        mutable_idx=list(range(seq_len)),
    )


def _report(seq):
    return VariantReport(
        sequence=seq, objectives={}, n_mutations=0, mutation_str="",
    )


def test_front_diversity_flags_a_collapsed_lineage():
    """Run 2's failure mode: two members differing in only a few positions."""
    target = _target()
    base = list(target.wildtype_seq)

    a = base.copy()
    a[0] = "W"
    b = base.copy()
    b[0] = "W"
    b[1] = "W"  # differs from `a` at exactly one position

    collapsed = front_diversity(target, [_report("".join(a)), _report("".join(b))])
    assert collapsed == pytest.approx(1.0)

    far = base.copy()
    for i in range(10, 25):
        far[i] = "W"
    diverse = front_diversity(target, [_report("".join(a)), _report("".join(far))])
    assert diverse > collapsed


def test_front_diversity_of_a_single_member_front_is_zero():
    target = _target()
    assert front_diversity(target, [_report(target.wildtype_seq)]) == 0.0


# --- end to end -------------------------------------------------------------


def _replication_target(seq_len=60):
    """Residue numbers 100..159, which span DuraPETase positions 117-159."""
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=seq_len))
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq=seq,
        residue_numbers=list(range(100, 100 + seq_len)),
        fixed_residue_numbers=(), fixed_idx=[],
        mutable_idx=list(range(seq_len)),
    )


def _fake_objectives(target):
    from polyphage.optimize import ObjectiveSpec, novelty_objective

    weights = np.random.default_rng(1).random((len(target.wildtype_seq), 20))
    idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}

    def score(seqs):
        return np.array(
            [sum(weights[p, idx[a]] for p, a in enumerate(s)) for s in seqs],
            dtype=float,
        )

    return [ObjectiveSpec("fake_nll", score), novelty_objective(target)]


def test_replicate_runs_every_seed_and_aggregates():
    from polyphage.replicate import replicate

    target = _replication_target()
    result = replicate(
        target, _fake_objectives(target), seeds=[1, 2, 3],
        pop_size=10, n_gen=4, max_mutations=8,
    )

    assert result.n_seeds == 3
    assert result.seeds == [1, 2, 3]
    assert len(result.front_sizes) == 3
    assert len(result.front_diversity) == 3
    assert set(result.per_seed_matches) == {1, 2, 3}
    assert 0 <= result.seeds_with_any_exact <= 3
    for rec in result.recoveries:
        assert 0 < rec.n_found <= 3
        assert 0.0 <= rec.rate <= 1.0
        lo, hi = rec.ci
        assert 0.0 <= lo <= rec.rate <= hi <= 1.0


def test_replicate_is_itself_reproducible():
    """Same seeds in, same recovery table out -- or the headline number drifts."""
    from polyphage.replicate import replicate

    target = _replication_target()
    kw = dict(seeds=[4, 5], pop_size=10, n_gen=4, max_mutations=8)
    a = replicate(target, _fake_objectives(target), **kw)
    b = replicate(target, _fake_objectives(target), **kw)

    assert a.front_sizes == b.front_sizes
    assert a.per_seed_matches == b.per_seed_matches


def test_replicate_attaches_pvalues_when_given_null_rates():
    from polyphage.replicate import replicate

    target = _replication_target()
    result = replicate(
        target, _fake_objectives(target), seeds=[1, 2, 3],
        pop_size=10, n_gen=4, max_mutations=8,
        null_rate_per_run={"Dura:140D": 0.01, "Dura:117F": 0.01},
    )
    for rec in result.recoveries:
        if rec.mutation in {"Dura:140D", "Dura:117F"}:
            assert rec.p_value is not None
            assert rec.p_adjusted is not None
            assert rec.p_adjusted >= rec.p_value


def test_replicate_reports_progress_per_seed(capsys):
    """A silent 45-minute cell is indistinguishable from a hang."""
    from polyphage.replicate import replicate

    target = _replication_target()
    replicate(target, _fake_objectives(target), seeds=[1, 2, 3],
              pop_size=8, n_gen=3, max_mutations=8)

    out = capsys.readouterr().out
    assert "[  1/3]" in out and "[  3/3]" in out
    assert "eta" in out
    assert out.count("seed=") == 3


def test_progress_can_be_silenced(capsys):
    from polyphage.replicate import replicate

    target = _replication_target()
    replicate(target, _fake_objectives(target), seeds=[1],
              pop_size=8, n_gen=3, max_mutations=8, progress=False)
    assert capsys.readouterr().out == ""


def test_print_replication_handles_an_empty_recovery_table(capsys):
    """No recoveries is a reportable result, not a crash."""
    from polyphage.replicate import ReplicationResult, print_replication

    print_replication(
        ReplicationResult(
            seeds=[1, 2], front_sizes=[3, 4], front_diversity=[5.0, 6.0],
            recoveries=[], seeds_with_any_exact=0,
        )
    )
    assert "No exact literature mutations" in capsys.readouterr().out
