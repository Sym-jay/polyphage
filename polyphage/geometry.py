"""Does the variant still have an active site?

Every objective in this pipeline so far asks whether a sequence *folds* or
whether it *survives heat*. Nothing asks whether it can still do chemistry.
That gap has a cost: ``results/baseline_comparison.md`` records a greedy
baseline dominating the entire Pareto front on both objectives, and that
baseline stacks nine aggressive mutations including N233, immediately next to
the catalytic His237. Either the search is useless or the objectives are
incomplete, and only a look at the active site can tell those apart.

IsPETase cuts PET with a serine hydrolase triad: **Ser160** attacks the ester,
**His237** deprotonates it, and **Asp206** holds His237 in the right tautomer
and orientation. The three are never mutated here -- ``load_target`` freezes
them -- but the other 262 positions can still pull them out of alignment.
Geometry is how you detect that.

This is a **filter, not a score**. It removes candidates that cannot work. It
does not rank the survivors, and it still does not measure catalysis: a variant
that keeps its triad may be a fine enzyme or a useless one. Passing is
necessary, never sufficient.

Model and measurement are kept apart, as everywhere else in this package.
``triad_geometry`` is pure Biopython over a PDB and needs no GPU, no network
and no model, so it is testable against hand-written coordinates.
``StructurePredictor`` is the part that needs ESMFold.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

from .encoding import CATALYTIC_TRIAD, THREE_TO_ONE, TRIAD_EXPECTED

#: A hydrogen bond this long is not a hydrogen bond. Both triad contacts sit
#: near 2.7-3.0 A in crystal structures; past this they are broken whatever a
#: prediction says, so it holds as an absolute bound under any calibration.
HBOND_MAX = 3.5

#: Below this, the prediction is too unsure at the active site to judge it.
PLDDT_MIN = 70.0


@dataclass
class TriadGeometry:
    """The two hydrogen bonds that make the triad a triad, and the confidence."""

    ser_his: float      # Ser160 OG  -- His237 NE2
    his_asp: float      # His237 ND1 -- Asp206 OD1/OD2, whichever is nearer
    plddt_triad: float  # mean pLDDT over the three residues
    plddt_mean: float   # mean pLDDT over the whole chain
    intact: bool

    def __str__(self) -> str:
        mark = "intact" if self.intact else "BROKEN"
        return (f"Ser-His {self.ser_his:.2f} A   His-Asp {self.his_asp:.2f} A   "
                f"pLDDT {self.plddt_triad:.0f} (triad) / {self.plddt_mean:.0f}   {mark}")


@dataclass
class ScreenResult:
    label: str
    geometry: TriadGeometry
    #: Deviation from the wild-type prediction, which is the honest baseline:
    #: if ESMFold renders this fold a little loose, every variant inherits that.
    d_ser_his: float = 0.0
    d_his_asp: float = 0.0


def _load_chain(pdb_source):
    """Parse a PDB from a path, a file object or a string of PDB text."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    if hasattr(pdb_source, "read"):
        handle = pdb_source
    elif isinstance(pdb_source, (str, Path)) and "\n" not in str(pdb_source):
        handle = open(pdb_source)
    else:
        handle = StringIO(str(pdb_source))

    try:
        structure = parser.get_structure("pred", handle)
    finally:
        if handle is not pdb_source:
            handle.close()

    model = next(iter(structure))
    chain = next(iter(model))
    return [r for r in chain if r.id[0] == " "]


def _triad_indices(target, triad):
    """Map PDB residue numbers onto 0-based positions in the target sequence.

    The numbering trap, stated plainly because this repository keeps falling
    into it: a predicted structure is numbered 1..L from the sequence it was
    given, while 6EQE is numbered 29..293. Residue 160 of the crystal is *not*
    residue 160 of the prediction. It is at sequence index
    ``residue_numbers.index(160)``, which the predictor writes out as residue
    index + 1.
    """
    out = {}
    for resnum in triad:
        try:
            out[resnum] = target.residue_numbers.index(resnum)
        except ValueError:
            raise ValueError(
                f"residue {resnum} is not modelled in this target "
                f"(it spans {target.residue_numbers[0]}-{target.residue_numbers[-1]}). "
                "The triad cannot be located."
            ) from None
    return out


def triad_geometry(pdb_source, target, triad: tuple[int, ...] = CATALYTIC_TRIAD,
                   hbond_max: float = HBOND_MAX, plddt_min: float = PLDDT_MIN
                   ) -> TriadGeometry:
    """Measure the catalytic triad in a predicted structure.

    ``pdb_source`` is a path, a file object, or PDB text. The structure is
    assumed to be numbered 1..L over ``target.wildtype_seq``'s length, which is
    what every folding model emits; the mapping back to PDB numbering goes
    through ``target.residue_numbers``.

    pLDDT is read from the B-factor column, where ESMFold writes it.

    Raises if the residue found at a mapped position is not the amino acid the
    triad requires. That check is the whole defence against an off-by-29: a
    silent misalignment would otherwise measure three arbitrary residues and
    report a confident number.
    """
    residues = _load_chain(pdb_source)
    idx = _triad_indices(target, triad)

    if len(residues) != len(target.wildtype_seq):
        raise ValueError(
            f"structure has {len(residues)} residues but the target sequence is "
            f"{len(target.wildtype_seq)}. These are not the same molecule, so no "
            "position mapping is trustworthy."
        )

    found = {}
    for resnum, i in idx.items():
        res = residues[i]
        one = THREE_TO_ONE.get(res.get_resname(), "X")
        want = TRIAD_EXPECTED.get(resnum)
        if want and one != want:
            raise ValueError(
                f"sequence index {i} should be PDB residue {resnum} ({want}), but the "
                f"structure has {res.get_resname()} ({one}) there. The structure and "
                "the target are misaligned -- do not trust any distance from this."
            )
        found[resnum] = res

    ser, asp, his = found[triad[0]], found[triad[1]], found[triad[2]]

    def atom(res, name):
        if name not in res:
            raise ValueError(
                f"{res.get_resname()}{res.id[1]} has no {name} atom; the structure is "
                "incomplete at the active site."
            )
        return res[name]

    ser_his = atom(ser, "OG") - atom(his, "NE2")
    nd1 = atom(his, "ND1")
    his_asp = min(nd1 - atom(asp, "OD1"), nd1 - atom(asp, "OD2"))

    def mean_b(res_list):
        bs = [a.get_bfactor() for r in res_list for a in r]
        return sum(bs) / len(bs) if bs else 0.0

    plddt_triad = mean_b([ser, asp, his])
    plddt_mean = mean_b(residues)

    intact = (ser_his <= hbond_max and his_asp <= hbond_max
              and plddt_triad >= plddt_min)

    return TriadGeometry(
        ser_his=float(ser_his), his_asp=float(his_asp),
        plddt_triad=float(plddt_triad), plddt_mean=float(plddt_mean),
        intact=bool(intact),
    )


class StructurePredictor:
    """ESMFold, loaded once and reused -- the shape ``BatchScorer`` already uses.

    Folding is the slow half (roughly 30-60 s for 265 residues on a T4), so the
    model is built in ``__init__`` and ``fold`` is cheap to call in a loop.

    On a T4 both of these are needed or 265 residues will not fit: the language
    model in half precision, and a chunked trunk.
    """

    def __init__(self, model_name: str = "facebook/esmfold_v1",
                 device: str | None = None, chunk_size: int = 64):
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.torch = torch
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EsmForProteinFolding.from_pretrained(model_name)
        self.model = self.model.to(self.device).eval()

        if self.device.type == "cuda":
            self.model.esm = self.model.esm.half()
            self.model.trunk.set_chunk_size(chunk_size)

    def fold(self, seq: str) -> str:
        """Fold one sequence. Returns PDB text, pLDDT in the B-factor column."""
        torch = self.torch
        inputs = self.tokenizer([seq], return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self.model(**inputs)
        return self.model.output_to_pdb(out)[0]


def screen(predictor, target, named_seqs, wildtype_label: str = "wild type",
           **kwargs) -> list[ScreenResult]:
    """Fold each sequence and measure its triad.

    ``named_seqs`` is ``[(label, sequence)]``. Both ``VariantReport`` and
    ``BaselineVariant`` carry ``.sequence``, so front variants and the greedy
    baseline go in without an adapter.

    Put the wild type first and label it ``wildtype_label``: every later result
    is reported as a deviation from it, which is the only fair comparison when
    the predictor's own rendering of this fold may be slightly loose.
    """
    results: list[ScreenResult] = []
    reference: TriadGeometry | None = None

    for label, seq in named_seqs:
        geom = triad_geometry(predictor.fold(seq), target, **kwargs)
        r = ScreenResult(label=label, geometry=geom)
        if reference is None and label == wildtype_label:
            reference = geom
        if reference is not None:
            r.d_ser_his = geom.ser_his - reference.ser_his
            r.d_his_asp = geom.his_asp - reference.his_asp
        results.append(r)
        print(f"  {label:<22} {geom}", flush=True)

    return results


def check_wildtype(geom: TriadGeometry) -> bool:
    """Gate the predictor before trusting anything it says about variants.

    If ESMFold cannot reproduce a triad that a 0.92 A crystal structure
    resolves, every variant measurement downstream is noise. Same role as
    ``check_equivalence`` in ``scoring.py`` and ``check_sign_convention`` in
    ``stability.py``: prove the external tool works here before building on it.
    """
    print(f"wild-type triad: {geom}")
    if geom.intact:
        print("  OK -- the predictor reproduces the catalytic triad. Proceed.")
        return True
    print("  STOP. The predictor does not reproduce the wild-type triad, which a")
    print("  0.92 A crystal structure resolves. Nothing it says about a variant")
    print("  means anything. Report this rather than screening against it.")
    return False


def print_screen(results: list[ScreenResult]) -> None:
    """The table that belongs in the write-up."""
    if not results:
        print("nothing screened")
        return

    print(f"\n{'variant':<22}{'Ser-His':>9}{'His-Asp':>9}{'pLDDT':>7}"
          f"{'dSer-His':>10}{'dHis-Asp':>10}  verdict")
    for r in results:
        g = r.geometry
        print(f"{r.label:<22}{g.ser_his:>9.2f}{g.his_asp:>9.2f}{g.plddt_triad:>7.0f}"
              f"{r.d_ser_his:>+10.2f}{r.d_his_asp:>+10.2f}  "
              f"{'intact' if g.intact else 'BROKEN'}")

    variants = [r for r in results[1:]]
    if not variants:
        return
    kept = sum(1 for r in variants if r.geometry.intact)
    print(f"\n{kept} of {len(variants)} preserve the catalytic triad.")
    print("A filter, not a score: passing means a candidate is not obviously dead,")
    print("not that it works. Nothing here measures catalysis.", file=sys.stdout)
