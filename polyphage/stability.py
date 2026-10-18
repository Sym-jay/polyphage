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

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from .encoding import AA_ALPHABET, mutations


class StabilityOracle:
    """Cached site-saturation ddG table.

    Builds once from ThermoMPNN, then answers any variant by lookup.
    """

    # Not at the repo root, despite what ThermoMPNN's README implies.
    SCRIPT = Path("analysis") / "custom_inference.py"

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

    def _find_weights(self) -> Path:
        """Locate ThermoMPNN's checkpoint without hardcoding a filename."""
        candidates = sorted(self.dir.glob("models/*.pt")) + sorted(self.dir.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"no .pt checkpoint under {self.dir / 'models'} or {self.dir}. "
                "Check that the ThermoMPNN clone includes its weights."
            )
        # Prefer the default checkpoint when several are shipped.
        for c in candidates:
            if "default" in c.name.lower():
                return c
        return candidates[0]

    def build(self, force: bool = False, extra_args: list[str] | None = None) -> None:
        """Run ThermoMPNN's site-saturation scan once and parse it.

        Three things about this invocation are not guessable from ThermoMPNN's
        README, and each one cost a debugging round before it was pinned down:

        * The script lives in ``analysis/``, not at the repo root.
        * Its real flags are ``--pdb --chain --model_path --out_dir``. ``--out_dir``
          takes a *directory*, not a CSV path, so the CSV is copied out afterwards.
        * ThermoMPNN ships its own ``datasets.py`` defining ``Mutation``, which
          collides with the installed HuggingFace ``datasets`` package. Running
          the script from ``analysis/`` does not put the repo root on the import
          path, so ``PYTHONPATH`` has to carry it or the import fails.
        """
        if not self.cache_csv.exists() or force:
            self.cache_csv.parent.mkdir(parents=True, exist_ok=True)
            script = self.dir / self.SCRIPT
            if not script.exists():
                raise FileNotFoundError(
                    f"{script} not found. ThermoMPNN may have moved it; locate it "
                    f"with `find {self.dir} -name custom_inference.py` and update "
                    "StabilityOracle.SCRIPT."
                )

            out_dir = self.cache_csv.parent / "thermompnn_out"
            out_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                self.python, str(self.SCRIPT),
                "--pdb", str(self.pdb_path),
                "--chain", self.chain,
                "--model_path", str(self._find_weights()),
                "--out_dir", str(out_dir),
            ]
            if extra_args:
                cmd += extra_args

            env = {**os.environ, "PYTHONPATH": str(self.dir.resolve())}
            result = subprocess.run(
                cmd, cwd=self.dir, capture_output=True, text=True, env=env
            )
            if result.returncode != 0:
                raise RuntimeError(
                    "ThermoMPNN custom_inference.py failed. Re-check its flags with\n"
                    f"  PYTHONPATH={self.dir} python {script} -h\n"
                    f"stdout:\n{result.stdout[-2000:]}\n\nstderr:\n{result.stderr[-2000:]}"
                )

            csvs = sorted(out_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime)
            if not csvs:
                raise RuntimeError(
                    f"ThermoMPNN exited cleanly but wrote no CSV to {out_dir}"
                )
            shutil.copy(csvs[-1], self.cache_csv)

        self.table = self._parse(self.cache_csv)
        self._validate_coverage()

    def _parse(self, csv_path: Path) -> dict[tuple[int, str], float]:
        """Parse the SSM output into {(pdb_residue_number, mutant_aa): ddG}.

        **ThermoMPNN's ``position`` is a 0-based sequence index, not a PDB
        residue number.** On 6EQE the two differ by 29 (the structure starts at
        residue 29), so reading the column as a residue number attaches every
        ddG to the wrong residue while still looking plausible -- roughly 89% of
        positions overlap numerically, enough to slip past a coverage check.
        The index is therefore mapped through ``target.residue_numbers``.

        Every row's wildtype letter is checked against the target sequence, so a
        wrong offset fails loudly on the first row instead of silently poisoning
        the objective.
        """
        import pandas as pd

        df = pd.read_csv(csv_path)
        cols = {c.lower().strip(): c for c in df.columns}

        def pick(*aliases):
            for a in aliases:
                if a in cols:
                    return cols[a]
            raise ValueError(
                f"could not find a column among {aliases} in {csv_path}. "
                f"Columns present: {list(df.columns)}. Adjust "
                "StabilityOracle._parse."
            )

        pos_col = pick("position", "pos", "idx", "index")
        wt_col = pick("wildtype", "wt", "wt_aa", "wildtype_aa")
        mut_col = pick("mutation", "mutant", "mut", "mut_aa", "mutant_aa")
        ddg_col = pick("ddg_pred", "ddg", "prediction", "predicted_ddg")

        resnums = self.target.residue_numbers
        seq = self.target.wildtype_seq

        table: dict[tuple[int, str], float] = {}
        for _, row in df.iterrows():
            try:
                pos = int(row[pos_col])
            except (TypeError, ValueError):
                continue

            wt = str(row[wt_col]).strip().upper()[-1:]
            if not 0 <= pos < len(seq):
                raise ValueError(
                    f"{csv_path}: position {pos} is outside the target sequence "
                    f"(length {len(seq)}). ThermoMPNN's indexing does not match "
                    "this design target."
                )
            if seq[pos] != wt:
                raise ValueError(
                    f"{csv_path}: at index {pos} ThermoMPNN reports wildtype {wt}, "
                    f"but the target sequence has {seq[pos]}. The two are not "
                    "aligned -- do not trust any ddG from this table."
                )

            # A "mutation" column may be a bare amino acid ("D") or full
            # notation ("T140D"); take the final character either way.
            aa = str(row[mut_col]).strip().upper()[-1:]
            if aa not in AA_ALPHABET or aa == wt:
                continue  # skip the wildtype->wildtype rows of an SSM scan
            table[(resnums[pos], aa)] = float(row[ddg_col])

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

    def check_sign_convention(self, verbose: bool = True) -> dict:
        """Verify that negative ddG really means *stabilising* in this output.

        The whole search runs backwards if the sign is flipped, and nothing else
        in the pipeline would notice: the optimiser would happily minimise
        towards the least stable variants it can find.

        The test uses the known heat-stable variants as ground truth. FAST-PETase
        and DuraPETase mutations were selected experimentally for thermostability,
        so under the documented convention they should sit below the median of
        all scored mutations. If they sit above it, treat the table as suspect
        before running anything.
        """
        from .validate import DURA_PETASE, FAST_PETASE

        known = dict(DURA_PETASE)
        known.update(FAST_PETASE)

        values = np.array(list(self.table.values()), dtype=float)
        found = {
            (rn, aa): self.table[(rn, aa)]
            for rn, aa in known.items()
            if (rn, aa) in self.table
        }
        if not found:
            raise ValueError(
                "none of the FAST-PETase or DuraPETase mutations are in the ddG "
                "table, so the sign convention cannot be checked. That itself "
                "points at a numbering mismatch."
            )

        median = float(np.median(values))
        known_vals = np.array(list(found.values()), dtype=float)
        frac_below = float((known_vals < median).mean())
        # Percentile of each known mutation within the full distribution.
        percentiles = {
            k: float((values < v).mean() * 100) for k, v in found.items()
        }

        verdict = (
            "consistent" if frac_below >= 0.6
            else "FLIPPED?" if frac_below <= 0.4
            else "inconclusive"
        )

        if verbose:
            print(f"Sign-convention check on {len(found)} known-stabilising mutations")
            print(f"  all mutations : median ddG = {median:+.3f}")
            for (rn, aa), v in sorted(found.items(), key=lambda kv: kv[1]):
                print(f"  {rn}{aa:<2} ddG={v:+.3f}  (percentile {percentiles[(rn, aa)]:.0f})")
            print(f"  {frac_below:.0%} of known mutations fall below the median")
            if verdict == "consistent":
                print("  OK -- negative ddG behaves as stabilising.")
            elif verdict == "FLIPPED?":
                print(
                    "  WARNING -- known-stabilising mutations cluster ABOVE the "
                    "median. The sign is probably inverted in this ThermoMPNN "
                    "output. Negate ddG in _parse before optimising."
                )
            else:
                print(
                    "  INCONCLUSIVE -- known mutations straddle the median. "
                    "Report this rather than assuming the convention."
                )

        return {
            "verdict": verdict,
            "n_known_found": len(found),
            "median_all": median,
            "fraction_below_median": frac_below,
            "percentiles": percentiles,
        }


def stability_objective(oracle: StabilityOracle):
    """ObjectiveSpec for ddG. Minimised, because negative ddG is stabilising."""
    from .optimize import ObjectiveSpec

    return ObjectiveSpec("ddg", lambda seqs: oracle.ddg_many(seqs))
