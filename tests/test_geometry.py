"""The catalytic geometry screen, measured against coordinates we chose.

No ESMFold, no GPU, no network: `triad_geometry` reads a PDB, so the tests hand
it one built atom by atom with known distances. Same approach as
`test_stability_parse.py`, and for the same reason -- the bug this code is most
likely to have is a silent off-by-29 between the predictor's 1..L numbering and
6EQE's 29..293, which only an explicit mapping test catches.
"""

from __future__ import annotations

import numpy as np
import pytest

from polyphage.encoding import AA_ALPHABET, Target
from polyphage.geometry import HBOND_MAX, ScreenResult, print_screen, triad_geometry

# 6EQE's real shape: 265 residues numbered 29..293, triad at 160/206/237.
SEQ_LEN = 265
FIRST_RESNUM = 29
TRIAD = (160, 206, 237)
IDX = {rn: rn - FIRST_RESNUM for rn in TRIAD}   # 131, 177, 208


def make_target() -> Target:
    """A synthetic 6EQE: right length, right numbering, S/D/H in the right places."""
    rng = np.random.default_rng(0)
    seq = list("".join(AA_ALPHABET[i] for i in rng.integers(0, 20, size=SEQ_LEN)))
    seq[IDX[160]] = "S"
    seq[IDX[206]] = "D"
    seq[IDX[237]] = "H"
    resnums = list(range(FIRST_RESNUM, FIRST_RESNUM + SEQ_LEN))
    fixed_idx = sorted(IDX.values())
    return Target(
        pdb_id="SYNTH", chain="A", wildtype_seq="".join(seq),
        residue_numbers=resnums, fixed_residue_numbers=TRIAD,
        fixed_idx=fixed_idx,
        mutable_idx=[i for i in range(SEQ_LEN) if i not in set(fixed_idx)],
    )


def _atom_line(serial, name, resname, resseq, xyz, b, element):
    name4 = name if len(name) == 4 else " " + name.ljust(3)
    x, y, z = xyz
    return ("ATOM  " + f"{serial:5d}" + " " + name4 + " " + f"{resname:>3}"
            + " A" + f"{resseq:4d}" + "    "
            + f"{x:8.3f}{y:8.3f}{z:8.3f}" + f"{1.00:6.2f}{b:6.2f}"
            + " " * 10 + f"{element:>2}")


HIS_NE2 = (2.8, 0.0, 0.0)   # everything else is placed relative to this


def build_pdb(*, ser_his_dist=2.8, his_ne2=HIS_NE2,
              his_nd1=(2.8, 4.0, 0.0), asp_od1=(2.8, 6.9, 0.0),
              asp_od2=(2.8, 8.0, 0.0), plddt=90.0, triad_plddt=None,
              n_res=SEQ_LEN, ser_resname="SER") -> str:
    """A PDB numbered 1..n_res, triad atoms exactly where we put them.

    Ser's OG is placed to give `ser_his_dist` exactly, rather than taking a
    coordinate -- sliding the atom along x toward His *shortens* the bond, which
    is an easy way to write a test that proves the opposite of what it claims.

    Defaults: Ser-His 2.80 A, His-Asp 2.90 A (OD1 is the nearer carboxylate
    oxygen; OD2 at 4.00 A is the decoy).
    """
    ser_og = (his_ne2[0] - ser_his_dist, his_ne2[1], his_ne2[2])
    tb = plddt if triad_plddt is None else triad_plddt
    lines, serial = [], 1
    for i in range(n_res):
        resseq = i + 1               # the predictor's own numbering, not the PDB's
        if i == IDX[160] and n_res == SEQ_LEN:
            atoms = [("CA", (0.0, -1.5, 0.0), "C"), ("OG", ser_og, "O")]
            resname, b = ser_resname, tb
        elif i == IDX[206] and n_res == SEQ_LEN:
            atoms = [("CA", (4.5, 7.5, 0.0), "C"),
                     ("OD1", asp_od1, "O"), ("OD2", asp_od2, "O")]
            resname, b = "ASP", tb
        elif i == IDX[237] and n_res == SEQ_LEN:
            atoms = [("CA", (4.2, 2.0, 0.0), "C"),
                     ("NE2", his_ne2, "N"), ("ND1", his_nd1, "N")]
            resname, b = "HIS", tb
        else:
            atoms = [("CA", (float(i) * 3.8, 20.0, 0.0), "C")]
            resname, b = "ALA", plddt
        for name, xyz, element in atoms:
            lines.append(_atom_line(serial, name, resname, resseq, xyz, b, element))
            serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


# --- the measurement ---------------------------------------------------------

def test_measures_the_distances_we_built():
    g = triad_geometry(build_pdb(), make_target(), triad=TRIAD)
    assert g.ser_his == pytest.approx(2.80, abs=0.01)
    assert g.his_asp == pytest.approx(2.90, abs=0.01)
    assert g.intact is True


def test_takes_the_nearer_carboxylate_oxygen():
    """His ND1 bonds whichever of OD1/OD2 is closer; the other must not win."""
    g = triad_geometry(
        build_pdb(asp_od1=(2.8, 9.0, 0.0), asp_od2=(2.8, 6.5, 0.0)),
        make_target(), triad=TRIAD,
    )
    assert g.his_asp == pytest.approx(2.50, abs=0.01)


def test_a_displaced_serine_breaks_the_triad():
    g = triad_geometry(build_pdb(ser_his_dist=6.0), make_target(), triad=TRIAD)
    assert g.ser_his == pytest.approx(6.00, abs=0.01)
    assert g.intact is False


def test_the_bound_is_where_it_says_it_is():
    """Just inside HBOND_MAX passes, just outside fails. No silent slack."""
    t = make_target()
    inside = triad_geometry(build_pdb(ser_his_dist=HBOND_MAX - 0.05), t, triad=TRIAD)
    outside = triad_geometry(build_pdb(ser_his_dist=HBOND_MAX + 0.05), t, triad=TRIAD)
    assert inside.intact is True
    assert outside.intact is False


# --- pLDDT -------------------------------------------------------------------

def test_plddt_comes_from_the_b_factor_column():
    g = triad_geometry(build_pdb(plddt=88.0, triad_plddt=95.0), make_target(), triad=TRIAD)
    assert g.plddt_triad == pytest.approx(95.0, abs=0.01)
    assert 88.0 <= g.plddt_mean < 89.0     # mostly backbone at 88, triad at 95


def test_an_unconfident_active_site_is_not_intact():
    """Good distances mean nothing if the model is guessing at the triad."""
    g = triad_geometry(build_pdb(triad_plddt=40.0), make_target(), triad=TRIAD)
    assert g.ser_his == pytest.approx(2.80, abs=0.01)
    assert g.intact is False


# --- the numbering trap, which is the point of this file ---------------------

def test_the_29_offset_is_applied():
    """PDB residue 160 is predicted residue 132, not 160.

    Put the real triad at the offset positions and a decoy SER/ASP/HIS at the
    un-offset ones. Reading the numbers literally would find the decoys.
    """
    pdb = build_pdb()
    lines = pdb.splitlines()
    # a decoy serine at predicted residue 160, where a naive reader would look
    decoy = _atom_line(99990, "OG", "SER", 160, (50.0, 50.0, 50.0), 90.0, "O")
    pdb_with_decoy = "\n".join(lines[:-1] + [decoy, "END"]) + "\n"

    g = triad_geometry(pdb_with_decoy, make_target(), triad=TRIAD)
    # the decoy is 50 A away; measuring it would give a huge distance
    assert g.ser_his == pytest.approx(2.80, abs=0.01)


def test_a_misaligned_structure_raises_rather_than_measuring_junk():
    """Wrong residue at the mapped position -- the off-by-N guard."""
    with pytest.raises(ValueError, match="misaligned"):
        triad_geometry(build_pdb(ser_resname="ALA"), make_target(), triad=TRIAD)


def test_the_error_names_both_residues():
    with pytest.raises(ValueError) as e:
        triad_geometry(build_pdb(ser_resname="ALA"), make_target(), triad=TRIAD)
    msg = str(e.value)
    assert "160" in msg and "ALA" in msg


def test_a_structure_of_the_wrong_length_raises():
    with pytest.raises(ValueError, match="not the same molecule"):
        triad_geometry(build_pdb(n_res=100), make_target(), triad=TRIAD)


def test_a_triad_outside_the_modelled_range_raises():
    t = make_target()
    with pytest.raises(ValueError, match="not modelled"):
        triad_geometry(build_pdb(), t, triad=(160, 206, 999))


# --- reporting ---------------------------------------------------------------

def test_print_screen_counts_only_the_variants(capsys):
    """The wild type is the calibration row, not one of the things being judged."""
    t = make_target()
    wt = triad_geometry(build_pdb(), t, triad=TRIAD)
    broken = triad_geometry(build_pdb(ser_his_dist=6.0), t, triad=TRIAD)

    print_screen([
        ScreenResult("wild type", wt),
        ScreenResult("variant a", wt),
        ScreenResult("variant b", broken),
    ])
    out = capsys.readouterr().out
    assert "1 of 2 preserve" in out
    assert "BROKEN" in out


def test_print_screen_survives_an_empty_run(capsys):
    print_screen([])
    assert "nothing screened" in capsys.readouterr().out
