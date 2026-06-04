#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
all_prediction_graphormer_rnn_sdbs_mixture_weighted_irsis.py

Stage-1 candidate generation for 1/2/3-component systems.

Compared with the original all_prediction_graphormer_rnn_sdbs.py:
    1. RNN still generates single-component SMILES first.
    2. Candidate mixtures are then constructed by joining components with ".".
    3. IR prediction and SIS calculation are performed through ir_sis.py.
    4. For 2- and 3-component mixtures, all predefined concentration patterns are tested,
       and the concentration pattern giving the highest SIS is recorded.

Important:
    This script does NOT call build_sdbs_linear_mixture_dataset.py.
    It only uses the same concentration patterns inside this script.

Outputs for each target:
    data_<row_index>/
        IR_data.csv
        generated_smiles_round_*.csv
        metadata.json
        smiles_sis_predictions.csv                  # combined 1/2/3-component result
        components_1/smiles_sis_predictions.csv
        components_2/smiles_sis_predictions.csv
        components_3/smiles_sis_predictions.csv

The component-specific files are all named smiles_sis_predictions.csv, as requested.
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


# ============================================================
# Concentration patterns
# Same patterns as build_sdbs_linear_mixture_dataset.py, copied here.
# No runtime call to build_sdbs_linear_mixture_dataset.py.
# ============================================================

BINARY_WEIGHT_PATTERNS = [
    [0.1, 0.9],
    [0.3, 0.7],
    [0.5, 0.5],
    [0.7, 0.3],
    [0.9, 0.1],
]

TERNARY_BASE_WEIGHT_PATTERNS = [
    [0.34, 0.33, 0.33],
    [0.50, 0.25, 0.25],
    [0.60, 0.30, 0.10],
    [0.70, 0.20, 0.10],
    [0.80, 0.10, 0.10],
]


def unique_permutations(weights: Sequence[float]) -> List[List[float]]:
    out = []
    seen = set()
    for p in itertools.permutations(weights, len(weights)):
        key = tuple(round(float(x), 6) for x in p)
        if key in seen:
            continue
        seen.add(key)
        out.append([float(x) for x in p])
    return out


def get_weight_patterns(n_components: int) -> List[Tuple[str, List[float]]]:
    if n_components == 1:
        return [("single_1.0", [1.0])]

    if n_components == 2:
        return [(f"binary_{w[0]:.1f}_{w[1]:.1f}", list(w)) for w in BINARY_WEIGHT_PATTERNS]

    if n_components == 3:
        rows = []
        for i, base in enumerate(TERNARY_BASE_WEIGHT_PATTERNS, start=1):
            for j, w in enumerate(unique_permutations(base), start=1):
                rows.append((f"ternary_pattern_{i}_perm_{j}", w))
        return rows

    raise ValueError("Only 1, 2, and 3 components are supported.")


# ============================================================
# Utilities
# ============================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize_phase(phase: str) -> str:
    p = "" if phase is None else str(phase).strip()
    low = " ".join(p.lower().split())

    mapping = {
        "kbr disc": "KBr",
        "kbr disk": "KBr",
        "kbr": "KBr",
        "kbr pellet": "KBr",
        "nujol mull": "nujol mull",
        "nujol": "nujol mull",
        "liquid film": "liquid film",
        "liquid": "liquid film",
        "gas phase": "gas",
        "gas": "gas",
        "vapor": "gas",
        "vapour": "gas",
    }
    return mapping.get(low, p if p else "unknown")


def parse_target_indices(text: Optional[str]) -> Optional[List[int]]:
    if text is None or str(text).strip() == "":
        return None
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def find_smiles_col(df: pd.DataFrame, user_col: Optional[str] = None) -> str:
    if user_col is not None and str(user_col).strip() != "":
        if user_col not in df.columns:
            raise ValueError(
                f"Specified SMILES column not found: {user_col}. "
                f"Available columns: {list(df.columns)}"
            )
        return user_col

    for col in ["smiles", "SMILES", "canonical_smiles", "Canonical_SMILES", "CanonicalSMILES", "Mixture", "mixture"]:
        if col in df.columns:
            return col

    raise ValueError(
        "Cannot find a SMILES column in the IR CSV. "
        "Please specify it with --smiles-col. "
        f"Available columns: {list(df.columns)}"
    )


def find_phase_col(df: pd.DataFrame, user_col: Optional[str] = None) -> Optional[str]:
    if user_col is not None and str(user_col).strip() != "":
        if user_col not in df.columns:
            raise ValueError(
                f"Specified phase column not found: {user_col}. "
                f"Available columns: {list(df.columns)}"
            )
        return user_col

    for col in ["phase", "Phase", "state", "State", "medium", "Medium"]:
        if col in df.columns:
            return col

    return None


def parse_numeric_col(col) -> Optional[float]:
    s = str(col).strip()
    if s.startswith("sp_"):
        s = s[3:]
    try:
        return float(s)
    except Exception:
        return None


def find_spectrum_cols(df: pd.DataFrame) -> List[str]:
    metadata = {
        "smiles", "SMILES", "canonical_smiles", "Canonical_SMILES", "CanonicalSMILES",
        "phase", "Phase", "state", "State", "medium", "Medium",
        "name", "Name", "cas", "CAS", "formula", "Formula",
    }

    cols = []
    for c in df.columns:
        if str(c) in metadata:
            continue
        v = parse_numeric_col(c)
        if v is not None:
            cols.append((v, c))

    cols = sorted(cols, key=lambda x: x[0])
    if not cols:
        raise ValueError("No spectral columns found. Expected 400..4000 or sp_* columns.")

    return [c for _, c in cols]


def canonicalize_smiles(smi: str) -> Optional[str]:
    smi = "" if smi is None else str(smi).strip()
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def canonicalize_component_list(comps: Sequence[str]) -> Optional[List[str]]:
    out = []
    for smi in comps:
        can = canonicalize_smiles(smi)
        if can is None:
            return None
        out.append(can)
    return out


def canonical_mixture_smiles(comps: Sequence[str]) -> str:
    return ".".join(sorted([str(x).strip() for x in comps if str(x).strip()]))


def align_prediction_to_target(pred: np.ndarray, target_dim: int) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64).reshape(-1)

    if len(pred) == target_dim:
        return pred

    if target_dim == 1701 and len(pred) >= 1751:
        return pred[50:1751]

    if target_dim < len(pred):
        return pred[:target_dim]

    raise ValueError(f"Prediction length {len(pred)} cannot be aligned to target dim {target_dim}.")


def import_ir_sis(ir_sis_path: str):
    """
    Dynamically import ir_sis.py from a file path or module path.
    """
    if ir_sis_path.endswith(".py") or os.path.sep in ir_sis_path:
        path = os.path.abspath(ir_sis_path)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"ir_sis.py not found: {path}")

        module_dir = os.path.dirname(path)
        if module_dir not in sys.path:
            sys.path.insert(0, module_dir)

        spec = importlib.util.spec_from_file_location("ir_sis_runtime", path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import ir_sis from {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return __import__(ir_sis_path)


def call_ir_load_models(ir_sis, checkpoint: str, device: str, local_pytorch_dir: str, batch_size: int):
    """
    Compatible with slightly different ir_sis.load_models signatures.
    """
    try:
        return ir_sis.load_models(
            checkpoint,
            device=device,
            local_pytorch_dir=local_pytorch_dir,
            predict_batch_size=batch_size,
        )
    except TypeError:
        try:
            return ir_sis.load_models(
                checkpoint,
                device=device,
                local_pytorch_dir=local_pytorch_dir,
            )
        except TypeError:
            return ir_sis.load_models(checkpoint, device=device)


def call_ir_predict_infrared(ir_sis, smiles_list, model, meta, device, phase, batch_size: int):
    try:
        return ir_sis.predict_infrared(
            smiles_list,
            gnn=model,
            kan=meta,
            device=device,
            phase=phase,
            batch_size=batch_size,
        )
    except TypeError:
        try:
            return ir_sis.predict_infrared(
                smiles_list,
                gnn=model,
                kan=meta,
                device=device,
                phase=phase,
            )
        except TypeError:
            return ir_sis.predict_infrared(smiles_list, model, meta, device)


# ============================================================
# Token-level RNN generator
# ============================================================

class RNNModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden: Optional[torch.Tensor] = None):
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)


def normalize_idx_to_token(idx_to_token: Dict) -> Dict[int, str]:
    return {int(k): v for k, v in idx_to_token.items()}


def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_k: int = 0) -> int:
    if temperature <= 0:
        return torch.argmax(logits).item()

    logits = logits / temperature

    if top_k and top_k > 0:
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        probs = torch.softmax(values, dim=-1)
        sampled_position = torch.multinomial(probs, num_samples=1).item()
        return indices[sampled_position].item()

    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).item()


def load_rnn_from_checkpoint(checkpoint_path: str, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    token_to_idx = checkpoint["token_to_idx"]
    idx_to_token = normalize_idx_to_token(checkpoint["idx_to_token"])

    model = RNNModel(
        vocab_size=len(token_to_idx),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        num_layers=checkpoint["num_layers"],
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_length = checkpoint.get("max_length", 200)
    return model, token_to_idx, idx_to_token, max_length


@torch.no_grad()
def generate_one_smiles(
    model: RNNModel,
    token_to_idx: Dict[str, int],
    idx_to_token: Dict[int, str],
    max_length: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 30,
) -> Tuple[str, List[str]]:
    go_idx = token_to_idx["GO"]
    eos_idx = token_to_idx["EOS"]
    pad_idx = token_to_idx["PAD"]

    current_idx = go_idx
    hidden = model.init_hidden(1, device)
    tokens = []

    for _ in range(max_length):
        x = torch.tensor([[current_idx]], dtype=torch.long, device=device)
        logits, hidden = model(x, hidden)
        next_idx = sample_next_token(logits[0, -1], temperature=temperature, top_k=top_k)

        if next_idx == eos_idx:
            break

        if next_idx in {go_idx, pad_idx}:
            continue

        token = idx_to_token[next_idx]
        tokens.append(token)
        current_idx = next_idx

    return "".join(tokens), tokens


def generate_many_smiles(
    model: RNNModel,
    token_to_idx: Dict[str, int],
    idx_to_token: Dict[int, str],
    max_length: int,
    device: torch.device,
    num_generate: int,
    temperature: float,
    top_k: int,
    valid_only: bool,
    unique_canonical: bool,
    max_attempt_factor: int,
) -> pd.DataFrame:
    records = []
    seen = set()

    max_attempts = num_generate
    if valid_only or unique_canonical:
        max_attempts = num_generate * max_attempt_factor

    attempts = 0
    pbar = tqdm(total=num_generate, desc="Generating single-component SMILES")

    while len(records) < num_generate and attempts < max_attempts:
        attempts += 1
        smi, tokens = generate_one_smiles(
            model=model,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            max_length=max_length,
            device=device,
            temperature=temperature,
            top_k=top_k,
        )

        mol = Chem.MolFromSmiles(smi)
        valid = mol is not None
        canonical = Chem.MolToSmiles(mol, canonical=True) if valid else ""

        if valid_only and not valid:
            continue

        if unique_canonical:
            if not valid:
                continue
            if canonical in seen:
                continue
            seen.add(canonical)

        records.append({
            "Generated_SMILES": smi,
            "Valid": valid,
            "Canonical_SMILES": canonical,
            "Token_Length": len(tokens),
            "Tokens": " ".join(tokens),
        })
        pbar.update(1)

    pbar.close()

    if len(records) < num_generate:
        print(f"Warning: requested {num_generate}, generated {len(records)} after {attempts} attempts.")

    return pd.DataFrame(records)


def get_single_component_pool(generated_df: pd.DataFrame) -> List[str]:
    if "Canonical_SMILES" in generated_df.columns:
        vals = generated_df["Canonical_SMILES"].fillna("").astype(str).tolist()
    elif "Generated_SMILES" in generated_df.columns:
        vals = generated_df["Generated_SMILES"].fillna("").astype(str).tolist()
    else:
        raise ValueError("Generated dataframe has neither Canonical_SMILES nor Generated_SMILES.")

    out = []
    seen = set()
    for smi in vals:
        can = canonicalize_smiles(smi)
        if can is None:
            continue
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
    return out


# ============================================================
# Mixture candidate construction and weighted SIS scoring
# ============================================================

def build_candidates_for_component_count(
    pool: Sequence[str],
    n_components: int,
    n_candidates: int,
    rng: random.Random,
) -> List[str]:
    """
    For k=1:
        return single-component SMILES.
    For k=2/3:
        randomly sample unique combinations from the single-component pool,
        then join with ".".
    """
    pool = list(dict.fromkeys([str(x).strip() for x in pool if str(x).strip()]))

    if n_components == 1:
        out = pool.copy()
        rng.shuffle(out)
        return out[:n_candidates] if n_candidates > 0 else out

    if len(pool) < n_components:
        return []

    # If the combinatorial space is small, enumerate all.
    max_all = 200000
    try:
        total_comb = 1
        # approximate n choose k for small k
        if n_components == 2:
            total_comb = len(pool) * (len(pool) - 1) // 2
        elif n_components == 3:
            total_comb = len(pool) * (len(pool) - 1) * (len(pool) - 2) // 6

        if total_comb <= min(max_all, max(n_candidates * 2, 10000)):
            combos = list(itertools.combinations(pool, n_components))
            rng.shuffle(combos)
            combos = combos[:n_candidates] if n_candidates > 0 else combos
            return [canonical_mixture_smiles(c) for c in combos]
    except Exception:
        pass

    seen = set()
    out = []

    max_attempts = max(n_candidates * 50, 1000)
    attempts = 0

    while len(out) < n_candidates and attempts < max_attempts:
        attempts += 1
        comps = rng.sample(pool, n_components)
        mix = canonical_mixture_smiles(comps)
        if mix in seen:
            continue
        seen.add(mix)
        out.append(mix)

    if len(out) < n_candidates:
        print(f"Warning: component={n_components}, requested {n_candidates}, generated {len(out)} mixtures.")

    return out


def collect_unique_components(mixture_candidates: Sequence[str]) -> List[str]:
    seen = set()
    out = []
    for mix in mixture_candidates:
        for comp in str(mix).split("."):
            comp = comp.strip()
            if not comp:
                continue
            if comp in seen:
                continue
            seen.add(comp)
            out.append(comp)
    return out


def predict_unique_components(
    component_smiles: Sequence[str],
    ir_sis,
    model,
    meta,
    device,
    phase: str,
    batch_size: int,
) -> Dict[str, np.ndarray]:
    cache: Dict[str, np.ndarray] = {}
    component_smiles = list(dict.fromkeys(component_smiles))

    for start in tqdm(range(0, len(component_smiles), batch_size), desc="Predicting unique component IR"):
        batch_smiles = component_smiles[start:start + batch_size]
        preds = call_ir_predict_infrared(
            ir_sis=ir_sis,
            smiles_list=batch_smiles,
            model=model,
            meta=meta,
            device=device,
            phase=phase,
            batch_size=batch_size,
        )
        for smi, pred in zip(batch_smiles, preds):
            cache[smi] = np.asarray(pred, dtype=np.float64).reshape(-1)

    return cache


def score_one_mixture_best_weight(
    mixture: str,
    component_cache: Dict[str, np.ndarray],
    target_matrix: np.ndarray,
    ir_sis,
    sigma: int,
) -> Dict:
    comps = [x.strip() for x in str(mixture).split(".") if x.strip()]
    n = len(comps)

    if n not in (1, 2, 3):
        raise ValueError(f"Only 1/2/3 components are supported, got {n}: {mixture}")

    preds = []
    for c in comps:
        if c not in component_cache:
            raise KeyError(f"Missing predicted spectrum for component: {c}")
        preds.append(component_cache[c])

    target_dim = int(target_matrix.shape[1])
    preds = [align_prediction_to_target(p, target_dim) for p in preds]

    best_sis = -1.0
    best_weights = None
    best_pattern = None
    best_pred = None

    for pattern_name, weights in get_weight_patterns(n):
        w = np.asarray(weights, dtype=np.float64).reshape(-1, 1)
        mix_pred = np.sum(np.stack(preds, axis=0) * w, axis=0)

        sis_vals = [
            ir_sis.calculate_sis(mix_pred, np.asarray(row, dtype=np.float64), sigma=sigma)
            for row in target_matrix
        ]
        sis = float(np.max(sis_vals)) if sis_vals else float("nan")

        if sis > best_sis:
            best_sis = sis
            best_weights = list(weights)
            best_pattern = pattern_name
            best_pred = mix_pred

    row = {
        "SMILES": mixture,
        "Mixture": mixture,
        "Components": " | ".join(comps),
        "Num_Components": n,
        "SIS": float(best_sis),
        "Best_SIS": float(best_sis),
        "Weight_Pattern": best_pattern,
        "Best_Weights": ",".join(f"{x:.6g}" for x in best_weights),
    }

    for i, comp in enumerate(comps, start=1):
        row[f"Component_{i}"] = comp

    for i, weight in enumerate(best_weights, start=1):
        row[f"Weight_{i}"] = float(weight)

    return row, best_pred


def score_mixture_candidates(
    mixture_candidates: Sequence[str],
    component_cache: Dict[str, np.ndarray],
    target_matrix: np.ndarray,
    ir_sis,
    sigma: int,
    phase: str,
    n_components: int,
    save_predicted_spectra: bool,
    spectrum_cols: Optional[List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    spectra_rows = []

    for mix in tqdm(mixture_candidates, desc=f"Scoring {n_components}-component candidates"):
        try:
            row, pred = score_one_mixture_best_weight(
                mixture=mix,
                component_cache=component_cache,
                target_matrix=target_matrix,
                ir_sis=ir_sis,
                sigma=sigma,
            )
            row["phase"] = phase
            rows.append(row)

            if save_predicted_spectra and pred is not None:
                sp_row = dict(row)
                if spectrum_cols is None or len(spectrum_cols) != len(pred):
                    spectrum_cols_use = [str(i) for i in range(len(pred))]
                else:
                    spectrum_cols_use = spectrum_cols
                for col, val in zip(spectrum_cols_use, pred):
                    sp_row[f"pred_{col}"] = float(val)
                spectra_rows.append(sp_row)

        except Exception as exc:
            rows.append({
                "SMILES": mix,
                "Mixture": mix,
                "Num_Components": n_components,
                "phase": phase,
                "SIS": -1.0,
                "Best_SIS": -1.0,
                "Error": f"{type(exc).__name__}: {exc}",
            })

    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("SIS", ascending=False).reset_index(drop=True)
        df.insert(0, "Rank", np.arange(1, len(df) + 1))

    spectra_df = pd.DataFrame(spectra_rows)
    if len(spectra_df):
        spectra_df = spectra_df.sort_values("SIS", ascending=False).reset_index(drop=True)
        spectra_df.insert(0, "Rank", np.arange(1, len(spectra_df) + 1))

    return df, spectra_df


# ============================================================
# Target selection and workflow
# ============================================================

def select_target_indices(df: pd.DataFrame, args: argparse.Namespace) -> List[int]:
    explicit = parse_target_indices(args.target_indices)
    if explicit is not None:
        for idx in explicit:
            if idx < 0 or idx >= len(df):
                raise ValueError(f"Target index out of range: {idx}")
        return explicit

    if args.target_indices_file:
        indices = []
        with open(args.target_indices_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                indices.append(int(line))
        for idx in indices:
            if idx < 0 or idx >= len(df):
                raise ValueError(f"Target index out of range: {idx}")
        return indices

    start = args.start
    end = len(df) if args.limit is None else min(len(df), start + args.limit)
    pool = list(range(start, end))

    if args.num_targets is None or args.num_targets <= 0 or args.num_targets >= len(pool):
        return pool

    rng = random.Random(args.seed)
    return rng.sample(pool, args.num_targets)


def parse_component_counts(text: str) -> List[int]:
    out = []
    for p in str(text).split(","):
        p = p.strip()
        if not p:
            continue
        k = int(p)
        if k not in (1, 2, 3):
            raise ValueError("--component-counts only supports 1,2,3")
        if k not in out:
            out.append(k)
    return out


def run_one_target(
    target_idx: int,
    ir_df: pd.DataFrame,
    smiles_col: str,
    phase_col: str,
    spectrum_cols: List[str],
    rnn_model: RNNModel,
    token_to_idx: Dict[str, int],
    idx_to_token: Dict[int, str],
    rnn_max_length: int,
    ir_sis,
    ir_model,
    ir_meta,
    ir_device,
    args: argparse.Namespace,
) -> None:
    rng = random.Random(args.seed + int(target_idx))

    target_row = ir_df.iloc[[target_idx]].copy()
    target_phase = normalize_phase(str(target_row.iloc[0][phase_col]))
    target_smiles = str(target_row.iloc[0][smiles_col])

    folder_name = os.path.join(args.out_root, f"data_{target_idx}")
    ensure_dir(folder_name)

    ir_data_path = os.path.join(folder_name, "IR_data.csv")
    target_row.to_csv(ir_data_path, index=False)

    # Use ir_sis.py target loading logic.
    target_matrix = ir_sis.load_ir_matrix(ir_data_path)
    target_dim = int(target_matrix.shape[1])

    metadata = {
        "target_index": int(target_idx),
        "target_smiles": target_smiles,
        "target_phase_original": str(target_row.iloc[0][phase_col]),
        "target_phase_used": target_phase,
        "num_generate_per_round": args.num_generate,
        "num_mixture_candidates": args.num_mixture_candidates,
        "rounds": args.rounds,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "rnn_checkpoint": args.rnn_checkpoint,
        "graphormer_checkpoint": args.graphormer_checkpoint,
        "ir_sis_path": args.ir_sis_path,
        "spectrum_cols_first": str(spectrum_cols[0]),
        "spectrum_cols_last": str(spectrum_cols[-1]),
        "spectrum_points": len(spectrum_cols),
        "target_matrix_dim": target_dim,
        "weight_patterns": {
            "binary": BINARY_WEIGHT_PATTERNS,
            "ternary_base": TERNARY_BASE_WEIGHT_PATTERNS,
            "ternary_all_permutations_tested": True,
        },
    }

    print("=" * 80)
    print(f"Target row     : {target_idx}")
    print(f"Target SMILES  : {target_smiles}")
    print(f"Target phase   : {target_phase}")
    print(f"Output folder  : {folder_name}")
    print(f"Target dim     : {target_dim}")
    print("=" * 80)

    component_counts = parse_component_counts(args.component_counts)

    all_component_results = []
    all_component_spectra = []

    for round_idx in range(1, args.rounds + 1):
        print(f"\nRound {round_idx}/{args.rounds}: generating single-component SMILES")

        generated_df = generate_many_smiles(
            model=rnn_model,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            max_length=args.max_length if args.max_length is not None else rnn_max_length,
            device=ir_device,
            num_generate=args.num_generate,
            temperature=args.temperature,
            top_k=args.top_k,
            valid_only=args.generate_valid_only or args.generate_unique_canonical,
            unique_canonical=args.generate_unique_canonical,
            max_attempt_factor=args.max_attempt_factor,
        )

        generated_path = os.path.join(folder_name, f"generated_smiles_round_{round_idx}.csv")
        generated_df.to_csv(generated_path, index=False)

        pool = get_single_component_pool(generated_df)
        print(f"Round {round_idx}: single-component pool size = {len(pool)}")

        # Build candidates for all requested component counts first.
        comp_to_candidates: Dict[int, List[str]] = {}
        all_unique_components = set()

        for k in component_counts:
            n_cand = args.num_mixture_candidates
            if k == 1 and args.num_single_candidates is not None:
                n_cand = args.num_single_candidates

            candidates = build_candidates_for_component_count(
                pool=pool,
                n_components=k,
                n_candidates=n_cand,
                rng=rng,
            )
            comp_to_candidates[k] = candidates

            for c in collect_unique_components(candidates):
                all_unique_components.add(c)

            print(f"Round {round_idx}: component={k}, candidates={len(candidates)}")

        # Predict all unique component spectra once.
        print(f"Predicting spectra for {len(all_unique_components)} unique components.")
        component_cache = predict_unique_components(
            component_smiles=sorted(all_unique_components),
            ir_sis=ir_sis,
            model=ir_model,
            meta=ir_meta,
            device=ir_device,
            phase=target_phase,
            batch_size=args.predict_batch_size,
        )

        for k in component_counts:
            out_dir = os.path.join(folder_name, f"components_{k}")
            ensure_dir(out_dir)

            candidates = comp_to_candidates[k]
            pd.DataFrame({"SMILES": candidates}).to_csv(
                os.path.join(out_dir, f"generated_candidates_round_{round_idx}.csv"),
                index=False,
            )

            sis_df, spectra_df = score_mixture_candidates(
                mixture_candidates=candidates,
                component_cache=component_cache,
                target_matrix=target_matrix,
                ir_sis=ir_sis,
                sigma=args.sigma,
                phase=target_phase,
                n_components=k,
                save_predicted_spectra=not args.no_save_predicted_spectra,
                spectrum_cols=[str(c) for c in spectrum_cols[:target_dim]],
            )

            if len(sis_df):
                sis_df["Round"] = round_idx
                # Keep one component-specific file named exactly smiles_sis_predictions.csv.
                current_out = os.path.join(out_dir, "smiles_sis_predictions.csv")

                if os.path.exists(current_out) and round_idx > 1:
                    old = pd.read_csv(current_out)
                    tmp = pd.concat([old, sis_df], ignore_index=True)
                    tmp = tmp.sort_values("SIS", ascending=False).reset_index(drop=True)
                    if "Rank" in tmp.columns:
                        tmp = tmp.drop(columns=["Rank"])
                    tmp.insert(0, "Rank", np.arange(1, len(tmp) + 1))
                    tmp.to_csv(current_out, index=False)
                else:
                    sis_df.to_csv(current_out, index=False)

                all_component_results.append(sis_df)

            if len(spectra_df):
                spectra_df["Round"] = round_idx
                spectra_out = os.path.join(out_dir, "predicted_spectra.csv")

                if os.path.exists(spectra_out) and round_idx > 1:
                    old = pd.read_csv(spectra_out)
                    tmp = pd.concat([old, spectra_df], ignore_index=True)
                    tmp = tmp.sort_values("SIS", ascending=False).reset_index(drop=True)
                    if "Rank" in tmp.columns:
                        tmp = tmp.drop(columns=["Rank"])
                    tmp.insert(0, "Rank", np.arange(1, len(tmp) + 1))
                    tmp.to_csv(spectra_out, index=False)
                else:
                    spectra_df.to_csv(spectra_out, index=False)

                all_component_spectra.append(spectra_df)

    # Combined file for backward compatibility.
    if all_component_results:
        combined = pd.concat(all_component_results, ignore_index=True)
        combined = combined.sort_values("SIS", ascending=False).reset_index(drop=True)
        if "Rank" in combined.columns:
            combined = combined.drop(columns=["Rank"])
        combined.insert(0, "Rank", np.arange(1, len(combined) + 1))
        combined.to_csv(os.path.join(folder_name, "smiles_sis_predictions.csv"), index=False)

        metadata["best_sis"] = float(combined["SIS"].iloc[0])
        metadata["best_smiles"] = str(combined["SMILES"].iloc[0])
        metadata["best_num_components"] = int(combined["Num_Components"].iloc[0])
        metadata["best_weights"] = str(combined["Best_Weights"].iloc[0])
        metadata["num_ranked_candidates"] = int(len(combined))
    else:
        combined = pd.DataFrame()
        metadata["best_sis"] = None
        metadata["best_smiles"] = None
        metadata["best_num_components"] = None
        metadata["best_weights"] = None
        metadata["num_ranked_candidates"] = 0

    if all_component_spectra and not args.no_save_predicted_spectra:
        combined_spectra = pd.concat(all_component_spectra, ignore_index=True)
        combined_spectra = combined_spectra.sort_values("SIS", ascending=False).reset_index(drop=True)
        if "Rank" in combined_spectra.columns:
            combined_spectra = combined_spectra.drop(columns=["Rank"])
        combined_spectra.insert(0, "Rank", np.arange(1, len(combined_spectra) + 1))
        combined_spectra.to_csv(os.path.join(folder_name, "predicted_spectra.csv"), index=False)

    with open(os.path.join(folder_name, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Finished target {target_idx}. Best SIS: {metadata['best_sis']}")
    print(f"Best SMILES: {metadata['best_smiles']}")
    print(f"Best weights: {metadata['best_weights']}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Generate 1/2/3-component candidate SMILES with a token-level RNN, "
            "predict component IR spectra with ir_sis.py, test predefined concentration "
            "patterns, and rank candidates by the best SIS."
        )
    )

    p.add_argument("--ir-csv", default="sdbs_ir_graphormer_preprocessed.csv")
    p.add_argument("--smiles-col", default=None, help="SMILES/mixture column in the target IR CSV.")
    p.add_argument("--phase-col", default=None, help="Phase column in the target IR CSV.")
    p.add_argument("--default-phase", default="unknown", help="Default phase if no phase column exists.")
    p.add_argument("--graphormer-checkpoint", required=True)
    p.add_argument("--rnn-checkpoint", required=True)
    p.add_argument("--ir-sis-path", default="ir_sis.py")
    p.add_argument("--local-pytorch-dir", default="local_pytorch")
    p.add_argument("--out-root", default="graphormer_rnn_mixture_search_out")

    p.add_argument("--num-targets", type=int, default=100)
    p.add_argument("--target-indices", default=None, help="Comma-separated target row indices, e.g. 0,5,10.")
    p.add_argument("--target-indices-file", default=None, help="Text file containing one row index per line.")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--limit", type=int, default=None)

    p.add_argument("--component-counts", default="1,2,3", help="Component counts to generate, e.g. 1,2,3.")
    p.add_argument("--num-generate", type=int, default=10000, help="Number of single-component SMILES generated per round.")
    p.add_argument("--num-mixture-candidates", type=int, default=10000, help="Number of candidates for each component count.")
    p.add_argument("--num-single-candidates", type=int, default=None, help="Optional override for 1-component candidates.")

    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--generate-valid-only", action="store_true")
    p.add_argument("--generate-unique-canonical", action="store_true")
    p.add_argument("--max-attempt-factor", type=int, default=20)

    p.add_argument("--predict-batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=2, help="Kept for CLI compatibility; ir_sis currently uses num_workers=0.")
    p.add_argument("--sigma", type=int, default=10, help="Gaussian sigma passed to ir_sis.calculate_sis.")

    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true", help="Kept for CLI compatibility. ir_sis handles prediction.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-save-predicted-spectra", action="store_true")

    return p


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    # Import ir_sis.py and load the Graphormer-IR-inspired forward predictor through it.
    ir_sis = import_ir_sis(args.ir_sis_path)

    if args.local_pytorch_dir and os.path.isdir(args.local_pytorch_dir):
        lp = os.path.abspath(args.local_pytorch_dir)
        if lp not in sys.path:
            sys.path.insert(0, lp)

    print(f"Loading IR data: {args.ir_csv}")
    ir_df = pd.read_csv(args.ir_csv)
    smiles_col = find_smiles_col(ir_df, args.smiles_col)
    phase_col = find_phase_col(ir_df, args.phase_col)

    if phase_col is None:
        print(f"WARNING: No phase column found. Using default phase: {args.default_phase}")
        ir_df["phase"] = args.default_phase
        phase_col = "phase"

    spectrum_cols = find_spectrum_cols(ir_df)

    print(f"Using SMILES column: {smiles_col}")
    print(f"Using phase column : {phase_col}")

    print(f"Loading IR predictor through ir_sis.py: {args.ir_sis_path}")
    ir_model, ir_meta, ir_device = call_ir_load_models(
        ir_sis=ir_sis,
        checkpoint=args.graphormer_checkpoint,
        device=args.device,
        local_pytorch_dir=args.local_pytorch_dir,
        batch_size=args.predict_batch_size,
    )

    print(f"Using device: {ir_device}")

    print(f"Loading token-level RNN checkpoint: {args.rnn_checkpoint}")
    rnn_model, token_to_idx, idx_to_token, rnn_max_length = load_rnn_from_checkpoint(
        args.rnn_checkpoint,
        device=ir_device,
    )
    print(f"RNN vocab size: {len(token_to_idx)}; checkpoint max length: {rnn_max_length}")

    target_indices = select_target_indices(ir_df, args)
    ensure_dir(args.out_root)

    target_list_path = os.path.join(args.out_root, "selected_targets.csv")
    ir_df.iloc[target_indices].assign(selected_row_index=target_indices).to_csv(target_list_path, index=False)
    print(f"Selected {len(target_indices)} targets. Saved list to: {target_list_path}")

    for target_idx in target_indices:
        run_one_target(
            target_idx=target_idx,
            ir_df=ir_df,
            smiles_col=smiles_col,
            phase_col=phase_col,
            spectrum_cols=spectrum_cols,
            rnn_model=rnn_model,
            token_to_idx=token_to_idx,
            idx_to_token=idx_to_token,
            rnn_max_length=rnn_max_length,
            ir_sis=ir_sis,
            ir_model=ir_model,
            ir_meta=ir_meta,
            ir_device=ir_device,
            args=args,
        )

    print("All targets finished.")


if __name__ == "__main__":
    main()
