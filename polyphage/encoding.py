"""Sequence <-> optimizer-vector encoding for a single design target.

The optimizer works on an integer vector with one entry per *mutable* position.
Everything that maps that vector back to a real amino-acid sequence, and back to
real PDB residue numbers, lives here so the rest of the pipeline never has to
guess about offsets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_ALPHABET)}

# Catalytic triad of IsPETase, in PDB residue numbering.
CATALYTIC_TRIAD = (160, 206, 237)
TRIAD_EXPECTED = {160: "S", 206: "D", 237: "H"}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


@dataclass
class Target:
    """A backbone to design against, plus the index bookkeeping it implies."""

    pdb_id: str
    chain: str
    wildtype_seq: str
    residue_numbers: list[int]  # PDB numbering, aligned 1:1 with wildtype_seq
    fixed_residue_numbers: tuple[int, ...]
    fixed_idx: list[int]        # 0-based positions into wildtype_seq
    mutable_idx: list[int]

    @property
    def n_mutable(self) -> int:
        return len(self.mutable_idx)

    def resnum(self, idx: int) -> int:
        """PDB residue number for a 0-based sequence position."""
        return self.residue_numbers[idx]


def load_target(
    pdb_path: str | Path,
    parsed_jsonl: str | Path,
    chain: str = "A",
    fixed_residue_numbers: tuple[int, ...] = CATALYTIC_TRIAD,
) -> Target:
    """Build a Target from a PDB file and ProteinMPNN's parsed .jsonl.

    Both are read from the same ATOM records, so the Biopython residue list and
    ProteinMPNN's sequence must come out the same length. That is asserted
    rather than assumed -- a mismatch means every downstream residue number is
    silently wrong.
    """
    pdb_path, parsed_jsonl = Path(pdb_path), Path(parsed_jsonl)
    pdb_id = pdb_path.stem

    with open(parsed_jsonl) as f:
        entry = next(json.loads(line) for line in f if line.strip())
    wildtype_seq = entry[f"seq_chain_{chain}"]

    structure = PDBParser(QUIET=True).get_structure(pdb_id, pdb_path)
    residues = [r for r in structure[0][chain] if r.id[0] == " "]
    residue_numbers = [r.id[1] for r in residues]

    if len(residue_numbers) != len(wildtype_seq):
        raise ValueError(
            f"{pdb_id} chain {chain}: Biopython sees {len(residue_numbers)} modeled "
            f"residues but ProteinMPNN parsed {len(wildtype_seq)}. Residue numbering "
            "cannot be trusted -- inspect the PDB for altlocs, insertion codes or "
            "HETATM residues before going further."
        )

    # Cross-check the two sources agree residue by residue, not just in length.
    bio_seq = "".join(THREE_TO_ONE.get(r.get_resname(), "X") for r in residues)
    if bio_seq != wildtype_seq:
        n_diff = sum(a != b for a, b in zip(bio_seq, wildtype_seq))
        raise ValueError(
            f"{pdb_id} chain {chain}: Biopython and ProteinMPNN sequences differ at "
            f"{n_diff} positions. Do not proceed; the index mapping is unreliable."
        )

    num_to_idx = {rn: i for i, rn in enumerate(residue_numbers)}
    missing = [rn for rn in fixed_residue_numbers if rn not in num_to_idx]
    if missing:
        raise ValueError(f"{pdb_id}: fixed residues not modeled in structure: {missing}")

    fixed_idx = sorted(num_to_idx[rn] for rn in fixed_residue_numbers)

    # Hard gate: the frozen positions must really be the catalytic triad.
    for rn, expected in TRIAD_EXPECTED.items():
        if rn in num_to_idx:
            found = wildtype_seq[num_to_idx[rn]]
            if found != expected:
                raise ValueError(
                    f"{pdb_id}: residue {rn} is {found}, expected {expected}. "
                    "This is not the IsPETase catalytic triad numbering."
                )

    fixed_set = set(fixed_idx)
    mutable_idx = [i for i in range(len(wildtype_seq)) if i not in fixed_set]

    return Target(
        pdb_id=pdb_id,
        chain=chain,
        wildtype_seq=wildtype_seq,
        residue_numbers=residue_numbers,
        fixed_residue_numbers=tuple(fixed_residue_numbers),
        fixed_idx=fixed_idx,
        mutable_idx=mutable_idx,
    )


def wildtype_vector(target: Target) -> np.ndarray:
    """The optimizer vector that decodes back to the unmutated sequence."""
    return np.array(
        [AA_TO_IDX[target.wildtype_seq[i]] for i in target.mutable_idx], dtype=int
    )


def decode(target: Target, x: np.ndarray) -> str:
    """Integer vector -> full-length sequence, with fixed positions untouched."""
    seq = list(target.wildtype_seq)
    for idx, aa_i in zip(target.mutable_idx, x):
        seq[idx] = AA_ALPHABET[int(aa_i)]
    return "".join(seq)


def decode_many(target: Target, X: np.ndarray) -> list[str]:
    return [decode(target, x) for x in np.atleast_2d(X)]


def hamming(target: Target, seq: str) -> int:
    return sum(a != b for a, b in zip(seq, target.wildtype_seq))


def mutations(target: Target, seq: str) -> list[tuple[int, str, str]]:
    """List of (pdb_residue_number, wildtype_aa, mutant_aa)."""
    return [
        (target.resnum(i), target.wildtype_seq[i], seq[i])
        for i in range(len(seq))
        if seq[i] != target.wildtype_seq[i]
    ]


def format_mutations(target: Target, seq: str) -> str:
    """Compact literature-style notation, e.g. 'S121E N233K'."""
    return " ".join(f"{wt}{rn}{mt}" for rn, wt, mt in mutations(target, seq))
