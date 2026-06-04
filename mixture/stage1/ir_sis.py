# -*- coding: utf-8 -*-
"""
ir_sis.py

Graphormer-IR-inspired forward predictor utilities for IR spectrum prediction
and SIS calculation against target IR spectra.

Main functions:
  - load_ir_matrix(ir_csv) -> np.ndarray
  - load_models(checkpoint_path, device=None, local_pytorch_dir="local_pytorch") -> (model, model_meta, device)
  - predict_infrared(smiles_list, gnn, kan, device, phase=None, phases=None) -> List[np.ndarray(1801,)]
  - calculate_sis(prediction, target, sigma=10) -> float
  - best_sis_for_smiles(...)
  - best_sis_for_mixture(...)
  - batch_best_sis_for_mixtures(...)

Compatible target IR CSV formats:
  1) smiles, phase, 400, 402, ..., 4000
  2) smiles, phase, sp_400, sp_402, ..., sp_4000
  3) smiles, phase, sp_0, sp_1, ..., sp_1800
  4) sp_50..sp_1750 format, handled automatically as a 1701-point spectrum

Phase handling:
  - The phase is usually passed by the caller, e.g., "KBr", "nujol mull", or "liquid film".
  - Aliases such as "KBr disc" are normalized to "KBr".
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Sequence
import os
import sys
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit import RDLogger
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader

RDLogger.DisableLog('rdApp.*')


# =======================
# Compatibility placeholder class name
# =======================

class GraphormerIRInspiredPredictor(nn.Module):
    """Placeholder class name. The actual model definition comes from local_pytorch/graphormer_ir_local.py."""
    pass


# =======================
# local_pytorch / Graphormer-IR-inspired model import
# =======================

_GRAPHORMER_IMPORTED = False
GraphormerIRLocal = None
mol_to_features = None
collate_graphs = None
move_batch_to_device = None
phase_to_id = None


def _import_graphormer_local(local_pytorch_dir: str = "local_pytorch"):
    """
    Dynamically import local_pytorch/graphormer_ir_local.py.
    The path can be specified through --local-pytorch-dir in all_prediction_GA.
    """
    global _GRAPHORMER_IMPORTED
    global GraphormerIRLocal, mol_to_features, collate_graphs, move_batch_to_device, phase_to_id

    if _GRAPHORMER_IMPORTED:
        return

    candidates = []
    if local_pytorch_dir:
        candidates.append(os.path.abspath(local_pytorch_dir))

    # local_pytorch under the current working directory
    candidates.append(os.path.abspath("local_pytorch"))

    # local_pytorch next to this ir_sis.py file
    this_dir = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(this_dir, "local_pytorch"))

    added = False
    for c in candidates:
        if os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)
            added = True

    try:
        from graphormer_ir_local import (  # type: ignore
            GraphormerIRLocal as _GraphormerIRLocal,
            mol_to_features as _mol_to_features,
            collate_graphs as _collate_graphs,
            move_batch_to_device as _move_batch_to_device,
            phase_to_id as _phase_to_id,
        )
    except Exception as exc:
        msg = [
            "Cannot import local_pytorch/graphormer_ir_local.py.",
            "Please make sure this file exists, or pass the following argument to all_prediction_GA_small_molecular.py:",
            "  --local-pytorch-dir /path/to/local_pytorch",
            f"Tried paths: {candidates}",
            f"Original error: {repr(exc)}",
        ]
        raise ImportError("\n".join(msg)) from exc

    GraphormerIRLocal = _GraphormerIRLocal
    mol_to_features = _mol_to_features
    collate_graphs = _collate_graphs
    move_batch_to_device = _move_batch_to_device
    phase_to_id = _phase_to_id
    _GRAPHORMER_IMPORTED = True


# =======================
# Phase and spectral-column handling
# =======================

def normalize_phase(phase: Optional[str]) -> str:
    """
    Normalize phase names for the Graphormer-IR-inspired predictor.
    Common experimental phases include KBr disc, nujol mull, and liquid film.
    KBr disc is mapped to KBr.
    """
    if phase is None:
        return "KBr"

    s = str(phase).strip()
    low = " ".join(s.lower().split())

    if low in {"kbr", "kbr disc", "kbr disk", "kbr pellet", "solid (kbr)", "solid kbr"}:
        return "KBr"
    if low in {"nujol", "nujol mull", "mull"}:
        return "nujol mull"
    if low in {"liquid film", "film", "liquid"}:
        return "liquid film"
    if low in {"kcl", "kcl disc", "kcl disk", "kcl pellet"}:
        return "KCl"
    if low in {"gas", "vapor", "vapour"}:
        return "gas"
    if low in {"ccl4", "ccl₄"}:
        return "CCl4"

    # For unknown phases, return the original string and let phase_to_id map it to unknown or an existing vocabulary entry.
    return s


def read_phase_from_ir_csv(ir_csv: str, phase_col: str = "phase", default_phase: str = "KBr") -> str:
    """
    Read the phase from IR_data.csv or sdbs_ir_graphormer_preprocessed.csv.
    If multiple target rows contain different phases, the first row is used. Usually, each data_* folder contains only one IR_data.csv row.
    """
    df = pd.read_csv(ir_csv, nrows=10)
    if phase_col in df.columns and len(df) > 0:
        val = df[phase_col].iloc[0]
        if not pd.isna(val):
            return normalize_phase(str(val))
    return normalize_phase(default_phase)


def _parse_numeric_col(col) -> Optional[float]:
    s = str(col).strip()
    if s.startswith("sp_"):
        s = s[3:]
    try:
        return float(s)
    except Exception:
        return None


def get_ir_spectral_columns(df: pd.DataFrame) -> List[str]:
    """
    Automatically identify IR spectral columns and sort them by wavenumber or index.
    Priority:
      1) numeric column names: 400,402,...,4000
      2) sp_400,sp_402,...,sp_4000
      3) sp_0..sp_1800
      4) sp_50..sp_1750
    """
    metadata = {
        "smiles", "SMILES", "canonical_smiles", "Canonical_SMILES",
        "phase", "Phase", "state", "State", "medium", "Medium",
        "formula", "Formula", "name", "Name", "cas", "CAS",
    }

    numeric_cols = []
    for c in df.columns:
        if str(c) in metadata:
            continue
        v = _parse_numeric_col(c)
        if v is not None:
            numeric_cols.append((v, c))

    if numeric_cols:
        numeric_cols = sorted(numeric_cols, key=lambda x: x[0])
        return [c for _, c in numeric_cols]

    raise ValueError("No spectral columns were found in the IR CSV. Supported formats are 400..4000, sp_400..sp_4000, sp_0..sp_1800, or sp_50..sp_1750.")


def load_ir_matrix(ir_csv: str) -> np.ndarray:
    """
    Load the target IR matrix.
    The 1801-dimensional format is supported:
      400,402,...,4000 or sp_400..sp_4000 or sp_0..sp_1800.
    The 1701-dimensional sp_50..sp_1750 format is also supported.
    """
    df = pd.read_csv(ir_csv)
    cols = get_ir_spectral_columns(df)
    arr = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").values.astype(np.float64)

    if arr.shape[1] not in (1801, 1701):
        # If the number of columns is not 1801 or 1701, interpolation could be added here; for now, raise an explicit error.
        raise ValueError(f"IR matrix must have 1801 or 1701 spectral columns; got {arr.shape[1]}")

    return arr


def _align_prediction_to_target(pred_1801: np.ndarray, target_matrix: np.ndarray) -> np.ndarray:
    """
    Align predictions to either a 1701-point target or an 1801-point target.
    """
    pred_1801 = np.asarray(pred_1801, dtype=np.float64).reshape(-1)
    n = int(target_matrix.shape[1])

    if n == len(pred_1801):
        return pred_1801
    if n == 1701 and len(pred_1801) >= 1751:
        return pred_1801[50:1751]
    if n < len(pred_1801):
        return pred_1801[:n]

    raise ValueError(f"Predicted spectrum length {len(pred_1801)} cannot be aligned to target length {n}")


# =======================
# SIS metric
# =======================

def sid(y_pred: np.ndarray, y_target: np.ndarray) -> float:
    y_pred = np.asarray(y_pred, dtype=np.float64)
    y_target = np.asarray(y_target, dtype=np.float64)

    eps = 1e-12
    y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=0.0, neginf=0.0)
    y_target = np.nan_to_num(y_target, nan=0.0, posinf=0.0, neginf=0.0)

    y_pred = np.clip(y_pred, eps, None)
    y_target = np.clip(y_target, eps, None)

    y_pred = y_pred / max(np.sum(y_pred), eps)
    y_target = y_target / max(np.sum(y_target), eps)

    return float(
        np.sum(y_pred * np.log(y_pred / y_target)) +
        np.sum(y_target * np.log(y_target / y_pred))
    )


def calculate_sis(prediction: np.ndarray, target: np.ndarray, sigma: int = 10) -> float:
    """
    Calculate SIS.
    sigma is the standard deviation, in points, used by gaussian_filter1d.
    """
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    prediction = np.nan_to_num(prediction, nan=0.0, posinf=0.0, neginf=0.0)
    target = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)

    prediction = np.clip(prediction, 1e-7, 1.0)
    target = np.clip(target, 1e-7, 1.0)

    # Apply Gaussian smoothing before SIS calculation.
    y_pred_smoothed = gaussian_filter1d(prediction, sigma=sigma)
    y_target_smoothed = gaussian_filter1d(target, sigma=sigma)

    return float(1.0 / (1.0 + sid(y_pred_smoothed, y_target_smoothed)))


# =======================
# Model loading and prediction wrapper
# =======================

def load_models(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    local_pytorch_dir: str = "local_pytorch",
    predict_batch_size: int = 64,
) -> Tuple[nn.Module, object, torch.device]:
    """
    Load the forward predictor and return a 3-tuple: (graphormer_model, model_meta, device).
    The model_meta object stores auxiliary metadata used by predict_infrared.
    """
    _import_graphormer_local(local_pytorch_dir)

    if device is None:
        device_obj = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device_obj = torch.device(device)
    else:
        device_obj = device

    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_config = ckpt["model_config"]

    model = GraphormerIRLocal(**model_config).to(device_obj)  # type: ignore
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    # Attach metadata used by predict_infrared.
    model._graphormer_spectrum_cols = [str(c) for c in ckpt.get("spectrum_cols", [])]
    model._graphormer_max_dist = model_config.get("max_dist", 20)
    model._graphormer_predict_batch_size = int(predict_batch_size)
    model._graphormer_local_pytorch_dir = local_pytorch_dir

    model_meta = {
        "spectrum_cols": model._graphormer_spectrum_cols,
        "max_dist": model._graphormer_max_dist,
        "predict_batch_size": int(predict_batch_size),
        "local_pytorch_dir": local_pytorch_dir,
    }

    return model, model_meta, device_obj


def _make_graph_items(smiles_list: Sequence[str], phases: Sequence[str], max_dist: int):
    items = []
    bad = []

    for i, (smi, ph) in enumerate(zip(smiles_list, phases)):
        graph = mol_to_features(smi, max_dist=max_dist)  # type: ignore
        if graph is None:
            bad.append((i, smi))
            continue

        ph_norm = normalize_phase(ph)
        items.append({
            "row_index": i,
            "smiles": str(smi),
            "phase": ph_norm,
            "phase_id": phase_to_id(ph_norm),  # type: ignore
            "graph": graph,
            "y": None,
        })

    return items, bad


@torch.no_grad()
def predict_infrared(
    smiles_list: List[str],
    gnn: nn.Module,
    kan: object,
    device: torch.device,
    phase: Optional[str] = None,
    phases: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
) -> List[np.ndarray]:
    """
    Predict IR spectra for a list of SMILES.

    phase:
      A single phase used for all SMILES.
    phases:
      One phase per SMILES; this has higher priority than phase.
    """
    _import_graphormer_local(getattr(gnn, "_graphormer_local_pytorch_dir", "local_pytorch"))

    if phases is None:
        ph = normalize_phase(phase)
        phases = [ph] * len(smiles_list)
    else:
        phases = [normalize_phase(x) for x in phases]
        if len(phases) != len(smiles_list):
            raise ValueError("The length of phases must match the length of smiles_list.")

    max_dist = int(getattr(gnn, "_graphormer_max_dist", 20))
    bs = int(batch_size or getattr(gnn, "_graphormer_predict_batch_size", 64))

    items, bad = _make_graph_items(smiles_list, phases, max_dist=max_dist)
    if bad:
        bad_msg = "; ".join([f"{idx}:{smi}" for idx, smi in bad[:5]])
        raise ValueError(f"Found {len(bad)} unparsable SMILES; examples: {bad_msg}")

    if not items:
        return []

    loader = DataLoader(
        items,
        batch_size=bs,
        shuffle=False,
        num_workers=0,
        collate_fn=lambda b: collate_graphs(b, max_dist=max_dist),  # type: ignore
        drop_last=False,
    )

    preds_map = {}
    gnn.eval()

    for batch in loader:
        batch = move_batch_to_device(batch, device)  # type: ignore
        pred = gnn(batch).detach().cpu().numpy()
        for j, row_idx in enumerate(batch["row_indices"]):
            preds_map[int(row_idx)] = pred[j].astype(np.float64)

    return [preds_map[i] for i in range(len(smiles_list))]


# =======================
# Single-compound / mixture SIS scoring interface
# =======================

def best_sis_for_smiles(smiles: str,
                        ir_matrix: np.ndarray,
                        gnn: nn.Module,
                        kan: object,
                        device: torch.device,
                        sigma: int = 10,
                        phase: Optional[str] = None) -> float:
    """
    Single compound:
      1) predict an 1801-point spectrum;
      2) align it automatically to the ir_matrix dimension;
      3) calculate SIS against each row in ir_matrix;
      4) return the maximum SIS.
    """
    pred = predict_infrared([smiles], gnn, kan, device=device, phase=phase)[0]
    y_pred = _align_prediction_to_target(pred, ir_matrix)
    sis_vals = [calculate_sis(y_pred, row, sigma=sigma) for row in ir_matrix]
    return float(np.max(sis_vals)) if len(sis_vals) else float("nan")


def parse_mixture(mixture: str) -> List[str]:
    """Split 'SMI1.SMI2[.SMI3...]' into a list and remove empty or whitespace-only parts."""
    parts = [p.strip() for p in str(mixture).split(".")]
    parts = [p for p in parts if p]
    if not parts:
        raise ValueError(f"Empty mixture string: {mixture!r}")
    return parts


@torch.no_grad()
def mixture_spectrum_equal_avg_1701(components: List[str],
                                    gnn: nn.Module,
                                    kan: object,
                                    device: torch.device,
                                    phase: Optional[str] = None,
                                    target_dim: Optional[int] = None) -> np.ndarray:
    """
    Predict the 1801-point spectra of all components and average them equally.
    If target_dim=1701, the averaged spectrum is automatically sliced to [50:1751].
    """
    preds = predict_infrared(components, gnn, kan, device=device, phase=phase)
    mix = np.mean(np.stack([np.asarray(p, dtype=np.float64) for p in preds], axis=0), axis=0)

    if target_dim == 1701 and len(mix) >= 1751:
        return mix[50:1751]
    return mix


def best_sis_for_mixture(mixture: str,
                         ir_matrix: np.ndarray,
                         gnn: nn.Module,
                         kan: object,
                         device: torch.device,
                         sigma: int = 10,
                         phase: Optional[str] = None) -> float:
    """
    Mixture: 'SMI1.SMI2[.SMI3...]' -> equally averaged mixture spectrum -> best SIS against ir_matrix.
    """
    comps = parse_mixture(mixture)
    mix = mixture_spectrum_equal_avg_1701(
        comps, gnn, kan, device,
        phase=phase,
        target_dim=ir_matrix.shape[1],
    )
    y_pred = _align_prediction_to_target(mix, ir_matrix) if len(mix) != ir_matrix.shape[1] else mix
    sis_vals = [calculate_sis(y_pred, row, sigma=sigma) for row in ir_matrix]
    return float(np.max(sis_vals)) if len(sis_vals) else float("nan")


def batch_best_sis_for_mixtures(mixture_list: List[str],
                                ir_matrix: np.ndarray,
                                gnn: nn.Module,
                                kan: object,
                                device: torch.device,
                                sigma: int = 10,
                                phase: Optional[str] = None) -> pd.DataFrame:
    rows = []
    for mix in mixture_list:
        comps = parse_mixture(mix)
        val = best_sis_for_mixture(mix, ir_matrix, gnn, kan, device, sigma=sigma, phase=phase)
        rows.append({
            "Mixture": mix,
            "Components": " | ".join(comps),
            "Num_Components": len(comps),
            "Best_SIS": val
        })
    return pd.DataFrame(rows)
