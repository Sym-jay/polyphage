"""Thermostability objective via ThermoMPNN.

ProteinMPNN's score answers "does this sequence fit this backbone". It says
nothing about heat. ThermoMPNN (Kuhlman Lab, MIT licence) answers the stability
question directly: it reuses ProteinMPNN's residue embeddings and adds a head
trained on the Megascale dataset to predict ddG, the change in folding free
energy on mutation. **Negative ddG = stabilising.**

The efficient trick: ThermoMPNN's ``custom_inference.py`` runs *site-saturation
mutagenesis* -- every position against every amino acid in one pass. Run it once
on the wildtype backbone, cache the resulting table, and scoring a variant
inside the optimiser becomes a handful of dictionary lookups. No model calls in
the loop at all.

LIMITATION, to state plainly in the report: ThermoMPNN is trained on *single*
point mutants. Summing per-mutation ddG over a 9-mutation variant assumes the
mutations do not interact. They do -- that interaction is called epistasis, and
additive models generally overestimate how stabilising a mutation stack is. This
is a standard, published screening heuristic, not a ground truth. Its job is to
rank candidates cheaply; the top few should later be checked with a method that
models the whole variant at once (ThermoMPNN-D handles pairs; MD or FoldX on the
folded structure handles the rest).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

from .encoding import AA_ALPHABET, mutations


class StabilityOracle:
    """Cached site-saturation ddG table.

    Builds once from ThermoMPNN, then answers any variant by lookup.
    """

    def __init__(
        self,
        thermompnn_dir: str | Path,
        pdb_path: str | Path,
        target,
        chain: str = "A",
        cache_csv: str | Path = "outputs/ssm_ddg.csv",
        python: str | None = None,
    ):
        self.dir = Path(thermompnn_dir)
        self.pdb_path = Path(pdb_path).resolve()
        self.target = target
        self.chain = chain
        self.cache_csv = Path(cache_csv).resolve()
        self.python = python or sys.executable
        self.table: dict[tuple[int, str], float] = {}

    def build(self, force: bool = False, extra_args: list[str] | None = None) -> None:
        """Run ThermoMPNN's site-saturation scan once and parse it.

        ``custom_inference.py``'s exact flag names are not documented in the
        README, so run ``!python custom_inference.py -h`` in the notebook first
        and pass anything extra through ``extra_args`` if the defaults below do
        not match this checkout.
        """
        if not self.cache_csv.exists() or force:
            self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
            cmd = [
                self.python, "custom_inference.py",
                "--pdb", str(self.pdb_path),
                "--chain", self.chain,
                "--out", str(self.cache_csv),
            ]
            if extra_args:
                cmd += extra_args
            result = subprocess.run(
                cmd, cwd=self.dir, capture_output=True, text=True
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "ThermoMPNN custom_inference.py failed. Run "
                    "`!python custom_inference.py -h` in the ThermoMPNN directory "
                    "and adjust the flags in StabilityOracle.build.\n"
                    f"stdout:\n{result.stdout[-2000:]}\n\nstderr:\n{result.stderr[-2000:]}"
                )

        self.table = self._parse(self.cache_csv)
        self._validate_coverage()

    def _parse(self, csv_path: Path) -> dict[tuple[int, str], float]:
        """Parse the SSM output into {(residue_number, mutant_aa): ddG}.

        Column names vary between ThermoMPNN versions, so they are matched by
        likely aliases rather than assumed.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        cols = {c.lower().strip(): c for c in df.columns}

        def pick(*aliases, required=True):
            for a in aliases:
                if a in cols:
                    return cols[a]
            if required:
                raise ValueError(
                    f"could not find a column among {aliases} in {csv_path}. "
                    f"Columns present: {list(df.columns)}. Adjust "
                    "StabilityOracle._parse."
                )
            return None

        pos_col = pick("position", "pos", "resnum", "residue", "pdb_position")
        mut_col = pick("mutant", "mut", "mutation", "mut_aa", "mutant_aa")
        ddg_col = pick("ddg", "ddg_pred", "prediction", "predicted_ddg", "ddG")

        table: dict[tuple[int, str], float] = {}
        for _, row in df.iterrows():
            mut = str(row[mut_col]).strip()
            # A "mutation" column may be either a bare amino acid ("D") or full
            # notation ("T140D"); take the final character either way.
            aa = mut[-1].upper()
            if aa not in AA_ALPHABET:
                continue
            try:
                resnum = int(row[pos_col])
            except (TypeError, ValueError):
                continue
            table[(resnum, aa)] = float(row[ddg_col])
        return table

    def _validate_coverage(self) -> None:
        """Fail loudly if the table does not cover the design positions."""
        if not self.table:
            raise ValueError(f"no usable rows parsed from {self.cache_csv}")

        wanted = {self.target.resnum(i) for i in self.target.mutable_idx}
        covered = {rn for rn, _ in self.table}
        missing = wanted - covered
        frac = 1 - len(missing) / len(wanted)
        print(
            f"ddG table: {len(self.table)} entries, "
            f"{frac:.1%} of mutable positions covered"
        )
        if frac < 0.9:
            raise ValueError(
                f"ddG table covers only {frac:.1%} of mutable positions "
                f"({len(missing)} missing). Residue numbering probably differs "
                "between ThermoMPNN's output and the design target -- check "
                f"a few missing numbers: {sorted(missing)[:10]}"
            )

    def ddg(self, seq: str) -> float:
        """Additive ddG for a variant. Negative = predicted more stable."""
        total = 0.0
        for resnum, _wt, mut in mutations(self.target, seq):
            value = self.table.get((resnum, mut))
            if value is None:
                # Unknown mutation: treat as neutral rather than silently
                # rewarding it. Coverage is validated at build time, so this
                # should be rare.
                continue
            total += value
        return total

    def ddg_many(self, seqs: list[str]) -> np.ndarray:
        return np.array([self.ddg(s) for s in seqs], dtype=float)

    def best_stabilising(self, n: int = 20) -> list[tuple[int, str, float]]:
        """The n most stabilising single mutations ThermoMPNN predicts.

        Useful on its own: compare this list against the FAST-PETase and
        DuraPETase mutation sets before running any search. If ThermoMPNN
        already ranks the known mutations highly, the optimiser is starting from
        a well-informed prior and that is worth reporting.
        """
        items = sorted(self.table.items(), key=lambda kv: kv[1])[:n]
        return [(rn, aa, v) for (rn, aa), v in items]


def stability_objective(oracle: StabilityOracle):
    """ObjectiveSpec for ddG. Minimised, because negative ddG is stabilising."""
    from .optimize import ObjectiveSpec

    return ObjectiveSpec("ddg", lambda seqs: oracle.ddg_many(seqs))
