"""ThermoMPNN's SSM output must land on the right residues.

ThermoMPNN reports a 0-based sequence index; the design target uses PDB
numbering, which on 6EQE starts at 29. Reading one as the other shifts every
ddG by 29 residues while still covering ~89% of positions numerically, so a
coverage check alone does not catch it. These tests pin the mapping down and
make a mismatch fail loudly.

No ThermoMPNN needed: ``_parse`` is exercised against a synthetic CSV.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.encoding import AA_ALPHABET, Target
from polyphage.stability import StabilityOracle

SEQ_LEN = 50
FIRST_RESNUM = 29  # 6EQE's first modeled residue


def make_target() -> Target:
    rng = np.random.default_rng(0)
    seq = "".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=SEQ_LEN))
    resnums = list(range(FIRST_RESNUM, FIRST_RESNUM + SEQ_LEN))
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq=seq, residue_numbers=resnums,
        fixed_residue_numbers=(), fixed_idx=[],
        mutable_idx=list(range(SEQ_LEN)),
    )


def write_ssm(tmp_path, target, *, corrupt_wt_at=None, as_pdb_numbers=False):
    """Write a ThermoMPNN-shaped site-saturation CSV for this target."""
    rows = ["position,wildtype,mutation,ddG_pred"]
    for i, wt in enumerate(target.wildtype_seq):
        pos = target.residue_numbers[i] if as_pdb_numbers else i
        letter = "X" if corrupt_wt_at == i else wt
        for aa in AA_ALPHABET:
            # ddG encodes the position so a shifted mapping is detectable.
            rows.append(f"{pos},{letter},{aa},{i / 100:.4f}")
    path = tmp_path / "ssm_ddg.csv"
    path.write_text("\n".join(rows) + "\n")
    return path


def make_oracle(tmp_path, target):
    return StabilityOracle(
        thermompnn_dir=tmp_path, pdb_path=tmp_path / "x.pdb",
        target=target, cache_csv=tmp_path / "ssm_ddg.csv",
    )


def test_index_is_mapped_to_pdb_residue_numbers(tmp_path):
    target = make_target()
    csv = write_ssm(tmp_path, target)
    table = make_oracle(tmp_path, target)._parse(csv)

    # Keys must be PDB numbers (29..78), never raw indices (0..49).
    keys = {rn for rn, _ in table}
    assert min(keys) == FIRST_RESNUM
    assert max(keys) == FIRST_RESNUM + SEQ_LEN - 1

    # And each ddG must still belong to the residue it was written for.
    for i in (0, 7, SEQ_LEN - 1):
        resnum = target.residue_numbers[i]
        aa = next(a for a in AA_ALPHABET if a != target.wildtype_seq[i])
        assert table[(resnum, aa)] == pytest.approx(i / 100)


def test_wildtype_rows_are_skipped(tmp_path):
    target = make_target()
    table = make_oracle(tmp_path, target)._parse(write_ssm(tmp_path, target))
    # 20 amino acids minus the wildtype one, per position.
    assert len(table) == SEQ_LEN * 19
    for i, wt in enumerate(target.wildtype_seq):
        assert (target.residue_numbers[i], wt) not in table


def test_mismatched_wildtype_letter_raises(tmp_path):
    """A shifted or foreign table must fail, not silently mis-assign ddG."""
    target = make_target()
    csv = write_ssm(tmp_path, target, corrupt_wt_at=12)
    with pytest.raises(ValueError, match="not aligned"):
        make_oracle(tmp_path, target)._parse(csv)


def test_pdb_numbered_input_is_rejected(tmp_path):
    """The old bug's input shape: positions already in PDB numbering.

    Those run past the end of the sequence, which must raise rather than
    quietly dropping ~29 residues.
    """
    target = make_target()
    csv = write_ssm(tmp_path, target, as_pdb_numbers=True)
    with pytest.raises(ValueError, match="outside the target sequence|not aligned"):
        make_oracle(tmp_path, target)._parse(csv)
