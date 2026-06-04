# brics_build_api.py
"""
Reusable API for BRICS-based molecular assembly and filtering.

Main functions:
1) prepare_fragment_library(...): build a fragment library from two CSV files.
2) time_limited_build(...): repeatedly assemble molecules within a time limit,
   filter by an SA threshold, and return the retained molecules.

Dependencies:
- rdkit, pandas
- molecule_builder.py in the same directory, which provides MoleculeBuilder.

Example:
    from brics_build_api import prepare_fragment_library, time_limited_build

    frags = prepare_fragment_library(
        pred_path="smiles_sis_predictions.csv",
        frags_path="data_fragments.csv",
        n_from_pred=2,
        n_from_frags=8,
        seed=42,
    )

    result = time_limited_build(
        fragment_library=frags,
        max_steps=5,
        sa_threshold=4.0,
        seconds=10,
        per_call_max_attempts=80,
        seed=42,
        reject_open=True,
    )

Optional command-line usage:
    python brics_build_api.py --pred smiles_sis_predictions.csv --frags data_fragments.csv \
        --sec 10 --sa_threshold 4.0 --max_steps 5 --per_call_max_attempts 80 --seed 42 --out out.csv
"""
from __future__ import annotations

import argparse
import time
import random
import re
from typing import List, Dict, Any, Tuple

import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS
from rdkit import RDLogger

# Disable verbose RDKit logs
RDLogger.DisableLog("rdApp.*")

# Use the external MoleculeBuilder class
from molecule_builder import MoleculeBuilder


# ================= Utility functions =================
def _read_column_case_insensitive(df: pd.DataFrame, preferred: str) -> pd.Series:
    """Read a column name case-insensitively; raise an error if it is missing."""
    cols = {c.lower(): c for c in df.columns}
    key = preferred.lower()
    if key not in cols:
        raise KeyError(f"Column {preferred} or a case variant was not found. Available columns: {list(df.columns)}")
    return df[cols[key]]

def _brics_fragments_from_smiles(smiles: str) -> List[str]:
    """Perform BRICS fragmentation on one SMILES and return fragment SMILES containing attachment points '*'."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    broken = BRICS.BreakBRICSBonds(mol)
    frags = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    out = []
    for f in frags:
        if any(a.GetSymbol() == "*" for a in f.GetAtoms()):
            out.append(Chem.MolToSmiles(f, canonical=True))
    return out

def _filter_labelled_fragments(smiles_list) -> List[str]:
    """Keep only BRICS-labeled fragments of the form [\\d*], matching the MoleculeBuilder regular expression."""
    patt = re.compile(r"\[\d+\*\]")
    return [s for s in map(str, smiles_list) if patt.search(s)]

def _has_attachment_star(smiles: str) -> bool:
    """Return whether the SMILES still contains '*' attachment points."""
    return "*" in smiles


# ================= Public API =================
def prepare_fragment_library(
    pred_path: str,
    frags_path: str,
    n_from_pred: int = 2,
    n_from_frags: int = 8,
    seed: int | None = None,
) -> List[str]:
    """
    Build a fragment library from two CSV files:
      - randomly sample n_from_pred molecules from pred_path, using the SMILES column, then perform BRICS fragmentation and extract [\d*]-labeled fragments;
      - randomly sample n_from_frags [\d*]-labeled fragments from frags_path, using the smiles column;
      - merge, deduplicate, and return the fragment library.

    Parameters:
      pred_path      : CSV path containing molecular SMILES in the SMILES column.
      frags_path     : CSV path containing fragment SMILES in the smiles column.
      n_from_pred    : number of molecules sampled from pred_path; default is 2.
      n_from_frags   : number of fragments sampled from frags_path; default is 8.
      seed           : random seed; default None means non-deterministic.

    Returns: deduplicated fragment library as List[str].
    """
    if seed is not None:
        random.seed(seed)

    df_pred = pd.read_csv(pred_path)
    df_frags = pd.read_csv(frags_path)

    # Select molecules
    series_pred = _read_column_case_insensitive(df_pred, "SMILES").dropna().astype(str)
    if len(series_pred) < n_from_pred:
        raise ValueError(f"{pred_path} contains fewer than {n_from_pred} available molecules.")
    # Molecule sampling
    chosen = series_pred.sample(n=n_from_pred, replace=False).tolist()

    # BRICS fragmentation and filtering for [\d*]-labeled fragments
    brics_lib = []
    for s in chosen:
        brics_lib.extend(_brics_fragments_from_smiles(s))
    brics_lib = _filter_labelled_fragments(brics_lib)

    # Select [\d*]-labeled fragments
    series_frag = _read_column_case_insensitive(df_frags, "smiles").dropna().astype(str)
    series_frag = pd.Series(_filter_labelled_fragments(series_frag))
    if len(series_frag) < n_from_frags:
        raise ValueError(f"{frags_path} contains fewer than {n_from_frags} digit-labeled fragments such as [14*].")
    chosen_frags = series_frag.sample(n=n_from_frags, replace=False).tolist()

    # Merge and deduplicate
    fragment_library = sorted(set(brics_lib + chosen_frags))
    if not fragment_library:
        raise ValueError("The merged fragment library is empty. Please check the input data.")
    return fragment_library


def time_limited_build(
    fragment_library: List[str],
    max_steps: int = 5,
    sa_threshold: float = 4.0,
    seconds: int = 10,
    per_call_max_attempts: int = 80,
    seed: int | None = None,
    reject_open: bool = True,
) -> Dict[str, Any]:
    """
    Repeatedly call MoleculeBuilder within a time limit, filter by the SA threshold, deduplicate, and return the retained molecules.

    Parameters:
      fragment_library       : fragment library containing BRICS-labeled [\d*] fragments.
      max_steps              : maximum assembly steps used by MoleculeBuilder.
      sa_threshold           : keep only molecules with SA <= threshold.
      seconds                : time limit in seconds.
      per_call_max_attempts  : maximum attempts for each build_one_molecule call.
      seed                   : random seed; default None means non-deterministic.
      reject_open            : when True, discard molecules that still contain '*' attachment points.

    Returns:
      {
        "seconds": seconds,
        "fragment_library_size": len(fragment_library),
        "total_success": <int>,   # raw successful builds before deduplication/filtering
        "kept_count": <int>,      # number of retained unique molecules
        "items": [ { "smiles": <str>, "sa": <float> }, ... ]  # sorted by ascending SA
      }
    """
    if seed is not None:
        random.seed(seed)

    builder = MoleculeBuilder(fragment_list=fragment_library, max_steps=max_steps)

    t0 = time.time()
    total_success = 0
    seen = set()
    kept: List[Tuple[str, float]] = []

    while time.time() - t0 < seconds:
        res = builder.build_one_molecule(max_attempts=per_call_max_attempts)
        if res is None:
            continue

        total_success += 1
        smi = str(res["final_smiles"])
        sa = float(res["sa_score"])

        if reject_open and _has_attachment_star(smi):
            continue
        if sa > sa_threshold:
            continue
        if smi in seen:
            continue

        seen.add(smi)
        kept.append((smi, sa))

    kept_sorted = sorted(kept, key=lambda x: x[1])
    items = [{"smiles": s, "sa": float(a)} for s, a in kept_sorted]

    return {
        "seconds": int(seconds),
        "fragment_library_size": len(fragment_library),
        "total_success": int(total_success),
        "kept_count": len(items),
        "items": items,
    }


# ============== Optional command-line interface ==============
def _cli():
    parser = argparse.ArgumentParser(description="BRICS assembly and SA filtering; importable API with optional CLI")
    parser.add_argument("--pred", type=str, default="smiles_sis_predictions.csv", help="Molecule CSV file with a SMILES column")
    parser.add_argument("--frags", type=str, default="data_fragments.csv", help="Fragment CSV file with a smiles column")
    parser.add_argument("--n_from_pred", type=int, default=2, help="Number of molecules sampled from the molecule file")
    parser.add_argument("--n_from_frags", type=int, default=8, help="Number of fragments sampled from the fragment file")
    parser.add_argument("--sec", type=int, default=1, help="Time limit in seconds")
    parser.add_argument("--sa_threshold", type=float, default=4.0, help="Keep only molecules with SA <= threshold")
    parser.add_argument("--max_steps", type=int, default=5, help="Maximum assembly steps in MoleculeBuilder")
    parser.add_argument("--per_call_max_attempts", type=int, default=80, help="Maximum attempts per build call")
    parser.add_argument("--seed", type=int, default=None, help="Random seed; default is non-deterministic")
    parser.add_argument("--no_reject_open", action="store_true", help="Do not filter open products that still contain *")
    parser.add_argument("--out", type=str, default=None, help="Optional output CSV with columns: smiles, sa")
    args = parser.parse_args()

    # 1) Prepare the fragment library
    frags = prepare_fragment_library(
        pred_path=args.pred,
        frags_path=args.frags,
        n_from_pred=args.n_from_pred,
        n_from_frags=args.n_from_frags,
        seed=args.seed,
    )

    # 2) Build molecules within the time limit and filter them
    result = time_limited_build(
        fragment_library=frags,
        max_steps=args.max_steps,
        sa_threshold=args.sa_threshold,
        seconds=args.sec,
        per_call_max_attempts=args.per_call_max_attempts,
        seed=args.seed,
        reject_open=not args.no_reject_open,
    )

    # Print summary
    print("\n================ Result summary ================")
    print(f"Time limit: {result['seconds']} seconds")
    print(f"Fragment library size: {result['fragment_library_size']}")
    print(f"Successful build attempts, including duplicates: {result['total_success']}")
    print(f"SA threshold: <= {args.sa_threshold:.2f}")
    print(f"Retained unique complete molecules: {result['kept_count']}")

    items = result["items"]
    if not items:
        print("No molecules met the criteria. Consider enlarging the fragment library, increasing attempts, or relaxing the threshold.")
    else:
        print("\nGenerated unique SMILES and SA scores, sorted by ascending SA:")
        for i, it in enumerate(items, 1):
            print(f"{i:>4d}. {it['smiles']}    SA={it['sa']:.3f}")

    # Optional CSV output
    if args.out:
        df_out = pd.DataFrame(items)
        df_out.to_csv(args.out, index=False)
        print(f"\nSaved CSV: {args.out} ({len(df_out)} rows)")


if __name__ == "__main__":
    _cli()
