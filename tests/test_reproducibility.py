"""The optimizer must be reproducible from its seed.

This is a regression test for a real bug. ``WildtypeBiasedSampling._do`` and
``RandomResetMutation._do`` used to read ``kwargs.get("seed")``, a key pymoo
never passes, so every run silently drew from OS entropy and two runs with the
same seed gave different fronts. Every reported number depends on this not
being true, including the recovery-rate statistics in ``control.py``.

No ProteinMPNN, no torch, no GPU: the target and the foldability objective are
synthetic, because what is under test is the search's RNG plumbing.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.encoding import AA_ALPHABET, Target
from polyphage.optimize import ObjectiveSpec, novelty_objective, run

SEQ_LEN = 40
TRIAD_IDX = (10, 20, 30)


def make_target() -> Target:
    """A small synthetic target with the same shape as a real one."""
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=SEQ_LEN))
    residue_numbers = list(range(29, 29 + SEQ_LEN))  # offset like 6EQE
    fixed_idx = list(TRIAD_IDX)
    return Target(
        pdb_id="SYNTH",
        chain="A",
        wildtype_seq=seq,
        residue_numbers=residue_numbers,
        fixed_residue_numbers=tuple(residue_numbers[i] for i in fixed_idx),
        fixed_idx=fixed_idx,
        mutable_idx=[i for i in range(SEQ_LEN) if i not in set(fixed_idx)],
    )


def fake_foldability() -> ObjectiveSpec:
    """A deterministic stand-in for ProteinMPNN: a fixed random scoring matrix.

    Deterministic in the sequence alone, so any run-to-run difference in the
    result comes from the search, never from the objective.
    """
    weights = np.random.default_rng(1).random((SEQ_LEN, 20))

    def score(seqs):
        idx = {aa: i for i, aa in enumerate(AA_ALPHABET)}
        return np.array(
            [sum(weights[p, idx[aa]] for p, aa in enumerate(s)) for s in seqs],
            dtype=float,
        )

    return ObjectiveSpec("fake_nll", score)


def front_of(seed: int):
    target = make_target()
    objectives = [fake_foldability(), novelty_objective(target)]
    res = run(
        target, objectives, pop_size=12, n_gen=6, max_mutations=8,
        seed=seed, verbose=False,
    )
    F = np.atleast_2d(res.F)
    return F[np.lexsort(F.T)]  # order-independent comparison


def test_same_seed_reproduces_the_front():
    a, b = front_of(1), front_of(1)
    assert a.shape == b.shape, (
        f"same seed gave fronts of different size ({a.shape[0]} vs {b.shape[0]}); "
        "the operators are not using pymoo's generator"
    )
    np.testing.assert_allclose(a, b, rtol=0, atol=0)


def test_different_seeds_give_different_runs():
    """Seeds must actually steer the search, not just sit in the config dict."""
    a, c = front_of(1), front_of(2)
    assert a.shape != c.shape or not np.allclose(a, c), (
        "seeds 1 and 2 produced identical fronts; the seed is being ignored"
    )


@pytest.mark.parametrize("key", ["seed", "random_state"])
def test_rng_helper_prefers_pymoos_generator(key):
    """``_rng`` must take pymoo's generator and ignore the key that never arrives."""
    from polyphage.optimize import _rng

    passed = np.random.default_rng(123)
    fallback = np.random.default_rng(456)
    got = _rng({key: passed}, fallback)

    if key == "random_state":
        assert got is passed
    else:
        # 'seed' is not a key pymoo uses; the fallback must be used instead of
        # silently constructing an unseeded generator.
        assert got is fallback
