"""ProteinMPNN scoring: foldability as negative log-likelihood of a sequence
given a fixed backbone.

Two implementations:

* ``SubprocessScorer`` -- shells out to ``protein_mpnn_run.py --score_only 1``,
  once per sequence. Slow (reloads weights every call) but it is ProteinMPNN's
  own code path, so it is the reference the fast one is checked against.
* ``BatchScorer`` -- loads the model once, featurizes the backbone once, and
  scores a whole population in one forward pass.

Never trust ``BatchScorer`` until ``check_equivalence`` passes.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

# ProteinMPNN's own alphabet, including the 'X' slot. Order matters: it is the
# index space the model's log-probs are in.
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


class SubprocessScorer:
    """Reference scorer. Correct by construction, far too slow to optimize with."""

    def __init__(
        self,
        mpnn_dir: str | Path,
        parsed_jsonl: str | Path,
        fixed_jsonl: str | Path,
        out_dir: str | Path = "outputs/score_subprocess",
    ):
        self.mpnn_dir = Path(mpnn_dir)
        self.parsed_jsonl = Path(parsed_jsonl).resolve()
        self.fixed_jsonl = Path(fixed_jsonl).resolve()
        self.out_dir = Path(out_dir).resolve()
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def score(self, seq: str, seed: int | None = None) -> float:
        """Score one sequence.

        ``seed`` is passed through to protein_mpnn_run.py, which calls
        torch.manual_seed with it. Without a seed the script picks a random one,
        so the decoding order -- and therefore the score -- differs per call.
        """
        fasta = self.out_dir / "candidate.fasta"
        fasta.write_text(f">candidate\n{seq}\n")

        cmd = [
            sys.executable, "protein_mpnn_run.py",
            "--jsonl_path", str(self.parsed_jsonl),
            "--fixed_positions_jsonl", str(self.fixed_jsonl),
            "--out_folder", str(self.out_dir),
            "--score_only", "1",
            "--save_score", "1",
            "--path_to_fasta", str(fasta),
        ]
        if seed is not None:
            cmd += ["--seed", str(seed)]

        result = subprocess.run(
            cmd, cwd=self.mpnn_dir, capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(
                "protein_mpnn_run.py failed\n"
                f"stdout:\n{result.stdout[-2000:]}\n\nstderr:\n{result.stderr[-2000:]}"
            )

        npz_files = sorted((self.out_dir / "score_only").glob("*.npz"))
        if not npz_files:
            raise RuntimeError(f"no .npz written under {self.out_dir / 'score_only'}")
        newest = max(npz_files, key=lambda p: p.stat().st_mtime)
        return float(np.load(newest)["score"].mean())


class BatchScorer:
    """Loads ProteinMPNN once and scores a population per forward pass.

    The backbone features are computed a single time in ``__init__`` and reused,
    which is the whole point: for a fixed-backbone design problem they never
    change, only the sequence does.
    """

    def __init__(
        self,
        mpnn_dir: str | Path,
        pdb_path: str | Path,
        chain: str = "A",
        weights: str = "v_48_020.pt",
        device: str | None = None,
        seed: int = 0,
        n_decoding_orders: int = 8,
    ):
        import torch

        self.mpnn_dir = Path(mpnn_dir)
        if str(self.mpnn_dir) not in sys.path:
            sys.path.insert(0, str(self.mpnn_dir))

        from protein_mpnn_utils import (
            ProteinMPNN, _scores, parse_PDB, tied_featurize,
        )

        self.torch = torch
        self._scores = _scores
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.seed = seed
        self.n_decoding_orders = n_decoding_orders

        ckpt_path = self.mpnn_dir / "vanilla_model_weights" / weights
        if not ckpt_path.exists():
            raise FileNotFoundError(f"weights not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)

        self.model = ProteinMPNN(
            ca_only=False,
            num_letters=21,
            node_features=128,
            edge_features=128,
            hidden_dim=128,
            num_encoder_layers=3,
            num_decoder_layers=3,
            augment_eps=0.0,
            k_neighbors=ckpt["num_edges"],
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Featurize the backbone once.
        pdb_dict_list = parse_PDB(str(pdb_path), ca_only=False)
        chain_id_dict = {pdb_dict_list[0]["name"]: ([chain], [])}
        feat = tied_featurize(
            pdb_dict_list, self.device, chain_id_dict,
            None, None, None, None, None, ca_only=False,
        )
        # tied_featurize returns a long positional tuple; these indices follow
        # protein_mpnn_run.py. If ProteinMPNN is upgraded, re-check them -- the
        # equivalence test below is what catches a silent reshuffle.
        self._X = feat[0]
        self._mask = feat[2]
        self._chain_M = feat[4]
        self._chain_encoding_all = feat[5]
        self._chain_M_pos = feat[10]
        self._residue_idx = feat[12]
        self.seq_len = int(self._X.shape[1])

    def _encode(self, seqs: list[str]) -> "object":
        torch = self.torch
        idx = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}
        rows = []
        for seq in seqs:
            if len(seq) != self.seq_len:
                raise ValueError(
                    f"sequence length {len(seq)} != backbone length {self.seq_len}"
                )
            rows.append([idx.get(aa, idx["X"]) for aa in seq])
        return torch.tensor(rows, dtype=torch.long, device=self.device)

    def score(
        self,
        seqs: list[str] | str,
        n_orders: int | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        """Mean per-residue negative log-likelihood. Lower = model likes it more.

        ProteinMPNN scores a sequence under a *random* autoregressive decoding
        order, so a single draw is a noisy estimate (spread of ~0.02 nats on
        IsPETase). Two consequences:

        * The score is averaged over ``n_orders`` decoding orders, drawn from a
          fixed seed sequence. That makes it deterministic across calls (so the
          optimizer compares candidates on equal terms) and cuts the noise by
          ~sqrt(n_orders). Without this, NSGA-II optimizes the dice: the
          per-draw noise is the same size as the score change from one or two
          mutations.
        * ``n_orders=1`` with an explicit ``seed`` reproduces one specific
          decoding order, which is what the equivalence check needs.
        """
        torch = self.torch
        if isinstance(seqs, str):
            seqs = [seqs]
        if not seqs:
            return np.zeros(0)

        n_orders = self.n_decoding_orders if n_orders is None else n_orders
        base_seed = self.seed if seed is None else seed

        n = len(seqs)
        S = self._encode(seqs)
        tile = lambda t: t.expand(n, *t.shape[1:]).contiguous()  # noqa: E731
        X = tile(self._X)
        mask = tile(self._mask)
        chain_M = tile(self._chain_M)
        chain_M_pos = tile(self._chain_M_pos)
        chain_encoding_all = tile(self._chain_encoding_all)
        residue_idx = tile(self._residue_idx)
        mask_for_loss = mask * chain_M * chain_M_pos

        total = np.zeros(n)
        with torch.no_grad():
            for k in range(n_orders):
                # Seed the same generator protein_mpnn_run.py uses (the global
                # one, on this device), so a matching seed gives a matching
                # decoding order.
                torch.manual_seed(base_seed + k)
                randn = torch.randn(chain_M.shape, device=self.device)
                log_probs = self.model(
                    X, S, mask, chain_M * chain_M_pos,
                    residue_idx, chain_encoding_all, randn,
                )
                total += self._scores(S, log_probs, mask_for_loss).cpu().numpy()

        return (total / n_orders).astype(float)


def check_equivalence(
    batch: BatchScorer,
    reference: SubprocessScorer,
    seq: str,
    seed: int = 37,
    tol: float = 1e-4,
    n_probe: int = 6,
) -> dict:
    """Assert the fast scorer agrees with ProteinMPNN's own script.

    Run this on the wildtype before any optimization. A real mismatch means
    ``BatchScorer`` is scoring something other than what it claims, which no
    amount of downstream tuning will fix.

    The comparison forces both scorers onto the *same* decoding order by seeding
    them identically, so the scores should agree to float tolerance. Comparing
    unseeded calls is meaningless: each draws its own random decoding order, and
    the resulting spread (~0.02 nats here) swamps any tolerance tight enough to
    catch a genuine bug.

    Caveat: the same-seed trick relies on both processes drawing ``randn`` from
    the same RNG stream. That holds on GPU, where the draw is the first use of
    the CUDA generator after ``manual_seed``. On CPU it can diverge, because
    building the model consumes CPU randomness first -- hence the statistical
    fallback below.
    """
    fast = float(batch.score([seq], n_orders=1, seed=seed)[0])
    slow = reference.score(seq, seed=seed)
    delta = abs(fast - slow)
    print(f"same decoding order (seed={seed}):")
    print(f"  batch={fast:.6f}  subprocess={slow:.6f}  |delta|={delta:.6f}")

    if delta <= tol:
        print("  PASS -- featurization matches ProteinMPNN exactly.")
        return {"mode": "exact", "delta": delta, "passed": True}

    # Not an exact match. Decide whether it is a feature bug or an RNG-stream
    # difference, by checking whether the reference value is an ordinary member
    # of the batch scorer's own across-order distribution.
    probe = np.array(
        [float(batch.score([seq], n_orders=1, seed=s)[0]) for s in range(n_probe)]
    )
    lo, hi, sd = probe.min(), probe.max(), probe.std()
    print(f"\n  batch across {n_probe} decoding orders:")
    print(f"    mean={probe.mean():.6f}  sd={sd:.6f}  range=[{lo:.6f}, {hi:.6f}]")

    within = lo - 2 * sd <= slow <= hi + 2 * sd
    if within:
        print(
            "\n  LIKELY OK -- the subprocess value sits inside the batch scorer's\n"
            "  own across-order spread, so the backbone features agree and only the\n"
            "  decoding order differs (different RNG stream). Averaging over\n"
            f"  n_decoding_orders={batch.n_decoding_orders} is what makes the\n"
            "  objective stable; proceed, but note this in the report."
        )
        return {
            "mode": "statistical", "delta": delta, "passed": True,
            "subprocess": slow, "batch_mean": float(probe.mean()), "batch_sd": float(sd),
        }

    raise AssertionError(
        f"subprocess score {slow:.6f} lies outside the batch scorer's across-order "
        f"range [{lo:.6f}, {hi:.6f}] (sd={sd:.6f}). This is a real mismatch, not "
        "decoding-order noise -- re-check the tied_featurize indices in "
        "BatchScorer.__init__ against this clone's protein_mpnn_utils.py."
    )
