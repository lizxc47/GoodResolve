#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pure PyTorch Graphormer-IR style model for local Python 3.12 environments.

This script is a fairseq-free refactor inspired by Graphormer-IR.
It is designed to run in the user's existing environment without relying on
fairseq, Hydra, OmegaConf, or torch_geometric.

Supported input CSV formats:
    1) smiles, phase, sp_0, sp_1, ..., sp_1800
    2) smiles, phase, sp_400, sp_402, ..., sp_4000
    3) smiles, phase, 400, 402, 404, ..., 4000

Typical training command:
    CUDA_VISIBLE_DEVICES=6 python local_pytorch/graphormer_ir_local.py train \
        --csv sample_data/sample_IR_train_data.csv \
        --out-dir checkpoints_local_pytorch \
        --batch-size 2 \
        --num-workers 2 \
        --epochs 1 \
        --device cuda \
        --amp

Typical prediction command:
    CUDA_VISIBLE_DEVICES=6 python local_pytorch/graphormer_ir_local.py predict \
        --csv my_smiles.csv \
        --checkpoint checkpoints_local_pytorch/best_model.pt \
        --out predictions_local.csv \
        --batch-size 32 \
        --device cuda

Prediction CSV can contain only:
    smiles, phase

For Graphormer-IR official SMILES convention, this script converts Q back to #
before RDKit parsing, because some official scripts replace triple-bond '#' with 'Q'.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    from rdkit import Chem
except ImportError as exc:
    raise ImportError(
        "RDKit is required. Install it with: pip install rdkit-pypi"
    ) from exc


# ============================================================
# Phase vocabulary used by Graphormer-IR-style data
# ============================================================

PHASE_VOCAB = {
    "unknown": 0,
    "nujol mull": 1,
    "CCl4": 2,
    "liquid film": 3,
    "KBr": 4,
    "KCl": 5,
    "gas": 6,
}


# ============================================================
# Utilities
# ============================================================

def seed_everything(seed: int = 23) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_column_name(col) -> str:
    return str(col).strip()


def find_smiles_col(df: pd.DataFrame, user_col: Optional[str]) -> str:
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(f"SMILES column '{user_col}' not found.")
        return user_col

    candidates = [
        "smiles",
        "SMILES",
        "canonical_smiles",
        "CanonicalSMILES",
        "canonical_SMILES",
        "mol",
        "Molecule",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    raise ValueError(
        "Cannot find SMILES column. Please specify --smiles-col."
    )


def find_phase_col(df: pd.DataFrame, user_col: Optional[str]) -> Optional[str]:
    if user_col is not None:
        if user_col not in df.columns:
            raise ValueError(f"Phase column '{user_col}' not found.")
        return user_col

    candidates = [
        "phase",
        "Phase",
        "state",
        "State",
        "medium",
        "Medium",
        "measurement",
        "Measurement",
    ]

    for col in candidates:
        if col in df.columns:
            return col

    return None


def try_parse_numeric_column_name(col) -> Optional[float]:
    col_str = normalize_column_name(col)
    try:
        return float(col_str)
    except Exception:
        return None


def find_spectrum_cols(
    df: pd.DataFrame,
    prefix: str = "sp_",
    exclude_cols: Optional[Sequence] = None,
) -> List:
    """
    Robustly find spectrum columns.

    Priority:
        1. Columns with prefix and numeric suffix, e.g. sp_0, sp_400.
        2. Pure numeric column names, e.g. 400, 402, ..., 4000.
        3. Numeric-valued columns after excluding metadata columns.

    This function is intentionally permissive because public/official sample
    data and the user's own datasets may use different spectral column names.
    """
    exclude_cols = set() if exclude_cols is None else set(exclude_cols)
    exclude_cols = {c for c in exclude_cols if c is not None}

    # --------------------------------------------------------
    # 1. Prefix columns: sp_0, sp_1, ..., or sp_400, sp_402, ...
    # --------------------------------------------------------
    prefix_cols: List[Tuple[int, object]] = []

    if prefix is not None and prefix != "":
        for col in df.columns:
            if col in exclude_cols:
                continue

            col_str = normalize_column_name(col)

            if col_str.startswith(prefix):
                suffix = col_str[len(prefix):]
                if suffix.isdigit():
                    prefix_cols.append((int(suffix), col))

        if len(prefix_cols) > 0:
            prefix_cols = sorted(prefix_cols, key=lambda x: x[0])
            return [col for _, col in prefix_cols]

    # --------------------------------------------------------
    # 2. Numeric column names: 400, 402, ..., 4000
    # --------------------------------------------------------
    numeric_name_cols: List[Tuple[float, object]] = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        value = try_parse_numeric_column_name(col)
        if value is not None:
            numeric_name_cols.append((value, col))

    # Require at least 10 columns to avoid accidentally using an ID column.
    if len(numeric_name_cols) >= 10:
        numeric_name_cols = sorted(numeric_name_cols, key=lambda x: x[0])
        return [col for _, col in numeric_name_cols]

    # --------------------------------------------------------
    # 3. Numeric-valued columns after excluding common metadata.
    # This is a fallback for unusual files.
    # --------------------------------------------------------
    common_metadata = {
        "smiles", "SMILES", "canonical_smiles", "CanonicalSMILES",
        "phase", "Phase", "state", "State", "medium", "Medium",
        "formula", "Formula", "name", "Name", "cas", "CAS", "CASRN",
        "inchi", "InChI", "inchikey", "InChIKey", "split", "label", "target",
        "index", "idx", "ID", "id",
    }

    numeric_value_cols: List[object] = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        if normalize_column_name(col) in common_metadata:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        valid_ratio = values.notna().mean()

        if valid_ratio >= 0.8:
            numeric_value_cols.append(col)

    if len(numeric_value_cols) >= 10:
        return numeric_value_cols

    return []


def canonicalize_smiles_for_rdkit(smiles: str) -> str:
    """
    Graphormer-IR scripts sometimes use Q to represent triple-bond '#'.
    Convert Q back to # for RDKit parsing.
    """
    smiles = str(smiles).strip()
    smiles = smiles.replace("Q", "#")
    return smiles


def phase_to_id(phase: str) -> int:
    phase = str(phase).strip()
    return PHASE_VOCAB.get(phase, PHASE_VOCAB["unknown"])


def clamp_int(value: int, min_value: int, max_value: int) -> int:
    value = int(value)
    if value < min_value:
        return min_value
    if value > max_value:
        return max_value
    return value


def mol_to_features(
    smiles: str,
    max_dist: int = 20,
) -> Optional[Dict[str, np.ndarray]]:
    """
    Convert SMILES to atom features and shortest-path distance matrix.

    The graph uses heavy atoms by default, as Chem.MolFromSmiles does not add
    explicit hydrogens unless requested.
    """
    smiles = canonicalize_smiles_for_rdkit(smiles)
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None

    num_atoms = mol.GetNumAtoms()
    if num_atoms <= 0:
        return None

    atomic_num = []
    degree = []
    formal_charge = []
    hybridization = []
    aromatic = []
    num_hs = []
    chirality = []
    ring = []

    hyb_map = {
        Chem.rdchem.HybridizationType.UNSPECIFIED: 0,
        Chem.rdchem.HybridizationType.S: 1,
        Chem.rdchem.HybridizationType.SP: 2,
        Chem.rdchem.HybridizationType.SP2: 3,
        Chem.rdchem.HybridizationType.SP3: 4,
        Chem.rdchem.HybridizationType.SP3D: 5,
        Chem.rdchem.HybridizationType.SP3D2: 6,
    }

    chi_map = {
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED: 0,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW: 1,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW: 2,
        Chem.rdchem.ChiralType.CHI_OTHER: 3,
    }

    for atom in mol.GetAtoms():
        atomic_num.append(clamp_int(atom.GetAtomicNum(), 0, 118))
        degree.append(clamp_int(atom.GetDegree(), 0, 6))
        formal_charge.append(clamp_int(atom.GetFormalCharge() + 5, 0, 10))
        hybridization.append(hyb_map.get(atom.GetHybridization(), 0))
        aromatic.append(1 if atom.GetIsAromatic() else 0)
        num_hs.append(clamp_int(atom.GetTotalNumHs(), 0, 4))
        chirality.append(chi_map.get(atom.GetChiralTag(), 0))
        ring.append(1 if atom.IsInRing() else 0)

    dist = Chem.GetDistanceMatrix(mol)
    dist = np.asarray(dist, dtype=np.int64)
    dist = np.clip(dist, 0, max_dist)

    return {
        "atomic_num": np.asarray(atomic_num, dtype=np.int64),
        "degree": np.asarray(degree, dtype=np.int64),
        "formal_charge": np.asarray(formal_charge, dtype=np.int64),
        "hybridization": np.asarray(hybridization, dtype=np.int64),
        "aromatic": np.asarray(aromatic, dtype=np.int64),
        "num_hs": np.asarray(num_hs, dtype=np.int64),
        "chirality": np.asarray(chirality, dtype=np.int64),
        "ring": np.asarray(ring, dtype=np.int64),
        "dist": dist,
        "num_atoms": np.asarray([num_atoms], dtype=np.int64),
    }


# ============================================================
# Dataset and collate
# ============================================================

class IRDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        smiles_col: Optional[str] = None,
        phase_col: Optional[str] = None,
        spectrum_prefix: str = "sp_",
        default_phase: str = "KBr",
        target_normalize: str = "none",
        max_dist: int = 20,
        require_spectrum: bool = True,
    ):
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)

        self.smiles_col = find_smiles_col(self.df, smiles_col)
        self.phase_col = find_phase_col(self.df, phase_col)

        exclude_cols = {self.smiles_col}
        if self.phase_col is not None:
            exclude_cols.add(self.phase_col)

        self.spectrum_cols = find_spectrum_cols(
            self.df,
            prefix=spectrum_prefix,
            exclude_cols=exclude_cols,
        )

        self.default_phase = default_phase
        self.target_normalize = target_normalize
        self.max_dist = max_dist
        self.require_spectrum = require_spectrum

        if require_spectrum and len(self.spectrum_cols) == 0:
            raise ValueError(
                "No spectrum columns found. Supported formats include: "
                "sp_0...sp_N, sp_400...sp_4000, or numeric columns 400...4000. "
                "You can also set --spectrum-prefix."
            )

        self.items: List[Dict] = []
        bad_smiles = 0
        bad_spectrum = 0

        for idx, row in self.df.iterrows():
            smiles = row[self.smiles_col]
            graph = mol_to_features(smiles, max_dist=max_dist)

            if graph is None:
                bad_smiles += 1
                continue

            if self.phase_col is None:
                phase = default_phase
            else:
                phase = row[self.phase_col]
                if pd.isna(phase):
                    phase = default_phase

            y = None
            if len(self.spectrum_cols) > 0:
                try:
                    y = row[self.spectrum_cols].to_numpy(dtype=np.float32)
                except Exception:
                    values = pd.to_numeric(row[self.spectrum_cols], errors="coerce")
                    y = values.to_numpy(dtype=np.float32)

                if np.any(~np.isfinite(y)):
                    bad_spectrum += 1
                    if require_spectrum:
                        continue
                    y = None
                else:
                    y = self.normalize_spectrum(y)

            self.items.append(
                {
                    "row_index": int(idx),
                    "smiles": str(smiles),
                    "phase": str(phase),
                    "phase_id": phase_to_id(phase),
                    "graph": graph,
                    "y": y,
                }
            )

        print(
            f"Loaded {len(self.items)} valid molecules from {csv_path}. "
            f"Invalid SMILES skipped: {bad_smiles}. "
            f"Invalid spectra skipped: {bad_spectrum}."
        )

        if len(self.spectrum_cols) > 0:
            print(
                f"Detected {len(self.spectrum_cols)} spectrum columns. "
                f"First: {self.spectrum_cols[0]}, Last: {self.spectrum_cols[-1]}"
            )
        else:
            print("No spectrum columns detected. This is allowed for prediction mode.")

    def normalize_spectrum(self, y: np.ndarray) -> np.ndarray:
        y = y.astype(np.float32)

        if self.target_normalize == "none":
            return y

        if self.target_normalize == "row_minmax":
            ymin = float(np.nanmin(y))
            ymax = float(np.nanmax(y))
            if ymax - ymin < 1e-8:
                return np.zeros_like(y, dtype=np.float32)
            return (y - ymin) / (ymax - ymin)

        if self.target_normalize == "row_max":
            ymax = float(np.nanmax(np.abs(y)))
            if ymax < 1e-8:
                return np.zeros_like(y, dtype=np.float32)
            return y / ymax

        raise ValueError(f"Unknown target_normalize: {self.target_normalize}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def collate_graphs(batch: List[Dict], max_dist: int = 20) -> Dict:
    """
    Collate variable-size molecular graphs.

    Adds one phase/global node to each molecule.

    node_type:
        0 = atom
        1 = phase/global node
        2 = padding
    """
    batch_size = len(batch)
    atom_counts = [int(item["graph"]["num_atoms"][0]) for item in batch]
    max_atoms = max(atom_counts)
    max_nodes = max_atoms + 1

    pad_dist_id = max_dist + 2
    phase_dist_id = max_dist + 1

    def make_long(fill: int = 0) -> torch.Tensor:
        return torch.full((batch_size, max_nodes), fill, dtype=torch.long)

    atomic_num = make_long(0)
    degree = make_long(0)
    formal_charge = make_long(5)
    hybridization = make_long(0)
    aromatic = make_long(0)
    num_hs = make_long(0)
    chirality = make_long(0)
    ring = make_long(0)
    node_type = make_long(2)

    padding_mask = torch.ones((batch_size, max_nodes), dtype=torch.bool)
    dist = torch.full((batch_size, max_nodes, max_nodes), pad_dist_id, dtype=torch.long)

    phase_ids = torch.zeros((batch_size,), dtype=torch.long)
    row_indices: List[int] = []
    smiles_list: List[str] = []
    phase_list: List[str] = []

    has_y = batch[0]["y"] is not None
    ys: List[torch.Tensor] = []

    for b, item in enumerate(batch):
        g = item["graph"]
        n = int(g["num_atoms"][0])
        phase_pos = n

        atomic_num[b, :n] = torch.from_numpy(g["atomic_num"])
        degree[b, :n] = torch.from_numpy(g["degree"])
        formal_charge[b, :n] = torch.from_numpy(g["formal_charge"])
        hybridization[b, :n] = torch.from_numpy(g["hybridization"])
        aromatic[b, :n] = torch.from_numpy(g["aromatic"])
        num_hs[b, :n] = torch.from_numpy(g["num_hs"])
        chirality[b, :n] = torch.from_numpy(g["chirality"])
        ring[b, :n] = torch.from_numpy(g["ring"])

        node_type[b, :n] = 0
        node_type[b, phase_pos] = 1
        padding_mask[b, : n + 1] = False

        d = torch.from_numpy(g["dist"]).long()
        dist[b, :n, :n] = d
        dist[b, phase_pos, : n + 1] = phase_dist_id
        dist[b, : n + 1, phase_pos] = phase_dist_id
        dist[b, phase_pos, phase_pos] = 0

        phase_ids[b] = int(item["phase_id"])
        row_indices.append(int(item["row_index"]))
        smiles_list.append(item["smiles"])
        phase_list.append(item["phase"])

        if has_y:
            ys.append(torch.from_numpy(item["y"]).float())

    output = {
        "atomic_num": atomic_num,
        "degree": degree,
        "formal_charge": formal_charge,
        "hybridization": hybridization,
        "aromatic": aromatic,
        "num_hs": num_hs,
        "chirality": chirality,
        "ring": ring,
        "node_type": node_type,
        "dist": dist,
        "padding_mask": padding_mask,
        "phase_id": phase_ids,
        "row_indices": row_indices,
        "smiles": smiles_list,
        "phases": phase_list,
    }

    if has_y:
        output["y"] = torch.stack(ys, dim=0)

    return output


# ============================================================
# Model
# ============================================================

class AtomFeatureEncoder(nn.Module):
    def __init__(self, dim: int, num_phases: int):
        super().__init__()
        self.atomic_num = nn.Embedding(119, dim)
        self.degree = nn.Embedding(7, dim)
        self.formal_charge = nn.Embedding(11, dim)
        self.hybridization = nn.Embedding(8, dim)
        self.aromatic = nn.Embedding(2, dim)
        self.num_hs = nn.Embedding(5, dim)
        self.chirality = nn.Embedding(4, dim)
        self.ring = nn.Embedding(2, dim)
        self.node_type = nn.Embedding(3, dim)
        self.phase = nn.Embedding(num_phases, dim)

        self.out = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = (
            self.atomic_num(batch["atomic_num"])
            + self.degree(batch["degree"])
            + self.formal_charge(batch["formal_charge"])
            + self.hybridization(batch["hybridization"])
            + self.aromatic(batch["aromatic"])
            + self.num_hs(batch["num_hs"])
            + self.chirality(batch["chirality"])
            + self.ring(batch["ring"])
            + self.node_type(batch["node_type"])
        )

        phase_embed = self.phase(batch["phase_id"])
        phase_mask = batch["node_type"].eq(1)

        # Each graph has exactly one phase/global node.
        for b in range(x.size(0)):
            x[b, phase_mask[b]] = x[b, phase_mask[b]] + phase_embed[b]

        return self.out(x)


class GraphormerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int,
        dropout: float,
        max_dist: int,
    ):
        super().__init__()

        if dim % num_heads != 0:
            raise ValueError(
                f"dim must be divisible by num_heads, got dim={dim}, heads={num_heads}."
            )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)

        # Distance IDs:
        # 0...max_dist: atom-atom shortest path distances
        # max_dist+1: phase/global-node distance
        # max_dist+2: padding distance
        self.dist_bias = nn.Embedding(max_dist + 3, num_heads)

        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        dist: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        bsz, n_nodes, dim = x.shape

        residual = x
        x_norm = self.norm1(x)

        qkv = self.qkv(x_norm)
        qkv = qkv.view(bsz, n_nodes, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2))
        scores = scores / math.sqrt(self.head_dim)

        bias = self.dist_bias(dist)
        bias = bias.permute(0, 3, 1, 2)
        scores = scores + bias

        # Mask padded keys.
        scores = scores.masked_fill(
            padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(bsz, n_nodes, dim)
        out = self.out_proj(out)
        out = self.dropout(out)

        x = residual + out
        x = x.masked_fill(padding_mask[:, :, None], 0.0)

        residual = x
        x = residual + self.ffn(self.norm2(x))
        x = x.masked_fill(padding_mask[:, :, None], 0.0)

        return x


class SpectrumSmoothingHead(nn.Module):
    def __init__(self, out_dim: int, kernel_size: int = 7):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.out_dim = out_dim
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )
        with torch.no_grad():
            self.conv.weight.fill_(1.0 / kernel_size)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        return self.conv(y.unsqueeze(1)).squeeze(1)


class GraphormerIRLocal(nn.Module):
    def __init__(
        self,
        out_dim: int,
        dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ffn_dim: int = 512,
        dropout: float = 0.1,
        max_dist: int = 20,
        num_phases: int = len(PHASE_VOCAB),
        use_softplus: bool = True,
        smooth_kernel: int = 7,
    ):
        super().__init__()

        self.out_dim = out_dim
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.max_dist = max_dist
        self.use_softplus = use_softplus
        self.smooth_kernel = smooth_kernel

        self.encoder = AtomFeatureEncoder(dim=dim, num_phases=num_phases)

        self.layers = nn.ModuleList(
            [
                GraphormerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    max_dist=max_dist,
                )
                for _ in range(num_layers)
            ]
        )

        self.graph_norm = nn.LayerNorm(dim)

        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, out_dim),
        )

        self.smooth = SpectrumSmoothingHead(out_dim=out_dim, kernel_size=smooth_kernel)
        self.softplus = nn.Softplus()

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = self.encoder(batch)

        for layer in self.layers:
            x = layer(
                x=x,
                dist=batch["dist"],
                padding_mask=batch["padding_mask"],
            )

        phase_mask = batch["node_type"].eq(1)
        graph_repr = []

        for b in range(x.size(0)):
            graph_repr.append(x[b, phase_mask[b]][0])

        graph_repr = torch.stack(graph_repr, dim=0)
        graph_repr = self.graph_norm(graph_repr)

        y = self.head(graph_repr)
        y = self.smooth(y)

        if self.use_softplus:
            y = self.softplus(y)

        return y


# ============================================================
# Losses and metrics
# ============================================================

def move_batch_to_device(batch: Dict, device: torch.device) -> Dict:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def sid_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Per-sample Spectral Information Divergence (SID), using the standard
    probability-distribution form used to derive SIS.

    For each spectrum pair x and y:
        p_i = x_i / sum_j x_j
        q_i = y_i / sum_j y_j

        SID(x, y) = sum_i p_i * log(p_i / q_i) + sum_i q_i * log(q_i / p_i)

    NaN positions are ignored. Both spectra are clipped to non-negative values
    before probability normalization.
    """
    pred = torch.clamp(pred, min=0.0)
    target = torch.clamp(target, min=0.0)

    valid_mask = torch.isfinite(pred) & torch.isfinite(target)

    pred = torch.where(valid_mask, pred, torch.zeros_like(pred))
    target = torch.where(valid_mask, target, torch.zeros_like(target))

    pred_sum = pred.sum(dim=1, keepdim=True).clamp_min(eps)
    target_sum = target.sum(dim=1, keepdim=True).clamp_min(eps)

    p = pred / pred_sum
    q = target / target_sum

    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)

    sid = p * torch.log(p / q) + q * torch.log(q / p)
    sid = torch.where(valid_mask, sid, torch.zeros_like(sid))

    return sid.sum(dim=1)


def sid_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Mean SID over a batch.

    This loss uses the same per-sample SID that is converted into SIS.
    """
    return sid_per_sample(pred, target, eps=eps).mean()


def gaussian_smooth_for_sis(
    spectra: torch.Tensor,
    std_dev_cm: float = 15.0,
    step_cm: float = 2.0,
    truncate: float = 5.0,
) -> torch.Tensor:
    """
    Gaussian convolution used for SIS evaluation.

    The public Graphormer-IR evaluation code constructs a Gaussian convolution
    matrix with std_dev=15 cm^-1 and applies it to both measured and predicted
    spectra before SID/SIS calculation. This function implements the same idea
    efficiently with 1D convolution.

    Because the smoothed spectra are sum-normalized before SID calculation,
    the absolute scale of the Gaussian kernel does not affect SIS.
    """
    if std_dev_cm <= 0:
        return spectra

    radius = max(1, int(math.ceil(truncate * std_dev_cm / step_cm)))
    offsets = torch.arange(
        -radius,
        radius + 1,
        device=spectra.device,
        dtype=spectra.dtype,
    ) * step_cm

    kernel = torch.exp(-0.5 * (offsets / std_dev_cm) ** 2)
    kernel = kernel.view(1, 1, -1)

    return torch.nn.functional.conv1d(
        spectra.unsqueeze(1),
        kernel,
        padding=radius,
    ).squeeze(1)


def sis_score_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
    gaussian_std_cm: float = 15.0,
    step_cm: float = 2.0,
) -> torch.Tensor:
    """
    Per-sample Spectral Information Similarity (SIS), following the
    Graphormer-IR evaluation procedure.

    Steps:
        1. Clip spectra to [1e-7, 1].
        2. Max-normalize each spectrum.
        3. Apply Gaussian convolution with std_dev=15 cm^-1.
        4. Sum-normalize both smoothed spectra.
        5. Compute symmetric SID.
        6. Convert to SIS = 1 / (1 + SID).

    Higher is better. Identical spectra give SIS = 1.
    """
    valid_mask = torch.isfinite(pred) & torch.isfinite(target)

    pred = torch.where(valid_mask, pred, torch.zeros_like(pred))
    target = torch.where(valid_mask, target, torch.zeros_like(target))

    pred = torch.clamp(pred, min=1e-7, max=1.0)
    target = torch.clamp(target, min=1e-7, max=1.0)

    pred = pred / pred.max(dim=1, keepdim=True).values.clamp_min(eps)
    target = target / target.max(dim=1, keepdim=True).values.clamp_min(eps)

    pred_smooth = gaussian_smooth_for_sis(
        pred,
        std_dev_cm=gaussian_std_cm,
        step_cm=step_cm,
    )
    target_smooth = gaussian_smooth_for_sis(
        target,
        std_dev_cm=gaussian_std_cm,
        step_cm=step_cm,
    )

    pred_smooth = torch.where(valid_mask, pred_smooth, torch.zeros_like(pred_smooth))
    target_smooth = torch.where(valid_mask, target_smooth, torch.zeros_like(target_smooth))

    p = pred_smooth / pred_smooth.sum(dim=1, keepdim=True).clamp_min(eps)
    q = target_smooth / target_smooth.sum(dim=1, keepdim=True).clamp_min(eps)

    p = torch.clamp(p, min=eps)
    q = torch.clamp(q, min=eps)

    sid = p * torch.log(p / q) + q * torch.log(q / p)
    sid = torch.where(valid_mask, sid, torch.zeros_like(sid))
    sid = sid.sum(dim=1)

    return 1.0 / (1.0 + sid)


def cosine_similarity_batch(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    pred = pred - pred.mean(dim=1, keepdim=True)
    target = target - target.mean(dim=1, keepdim=True)
    numerator = (pred * target).sum(dim=1)
    denominator = pred.norm(dim=1) * target.norm(dim=1)
    return numerator / denominator.clamp_min(eps)


def rmse_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=1))


def make_amp_context(enabled: bool):
    if enabled and torch.cuda.is_available():
        return torch.cuda.amp.autocast(enabled=True)
    return torch.cuda.amp.autocast(enabled=False)


def make_scaler(enabled: bool):
    return torch.cuda.amp.GradScaler(enabled=enabled and torch.cuda.is_available())


# ============================================================
# Training and evaluation
# ============================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mse_weight: float,
    sid_weight: float,
    amp: bool,
    scaler,
    grad_clip: float,
) -> Dict[str, float]:
    model.train()

    total_loss = 0.0
    total_mse = 0.0
    total_sid = 0.0
    total_sis = 0.0
    total_cos = 0.0
    total_rmse = 0.0
    total_n = 0

    for step, batch in enumerate(loader, start=1):
        batch = move_batch_to_device(batch, device)
        y = batch["y"]

        optimizer.zero_grad(set_to_none=True)

        with make_amp_context(amp):
            pred = model(batch)
            mse = torch.mean((pred - y) ** 2)
            sid = sid_loss(pred, y)
            loss = mse_weight * mse + sid_weight * sid

        if amp:
            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        with torch.no_grad():
            sis = sis_score_batch(pred.detach(), y).mean()
            cos = cosine_similarity_batch(pred.detach(), y).mean()
            rmse = rmse_batch(pred.detach(), y).mean()

        bs = y.size(0)
        total_loss += float(loss.detach()) * bs
        total_mse += float(mse.detach()) * bs
        total_sid += float(sid.detach()) * bs
        total_sis += float(sis.detach()) * bs
        total_cos += float(cos.detach()) * bs
        total_rmse += float(rmse.detach()) * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "mse": total_mse / max(total_n, 1),
        "sid": total_sid / max(total_n, 1),
        "sis": total_sis / max(total_n, 1),
        "cos": total_cos / max(total_n, 1),
        "rmse": total_rmse / max(total_n, 1),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mse_weight: float,
    sid_weight: float,
    amp: bool,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_sid = 0.0
    total_sis = 0.0
    total_cos = 0.0
    total_rmse = 0.0
    total_n = 0

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        y = batch["y"]

        with make_amp_context(amp):
            pred = model(batch)
            mse = torch.mean((pred - y) ** 2)
            sid = sid_loss(pred, y)
            loss = mse_weight * mse + sid_weight * sid

        sis = sis_score_batch(pred, y).mean()
        cos = cosine_similarity_batch(pred, y).mean()
        rmse = rmse_batch(pred, y).mean()

        bs = y.size(0)
        total_loss += float(loss.detach()) * bs
        total_mse += float(mse.detach()) * bs
        total_sid += float(sid.detach()) * bs
        total_sis += float(sis.detach()) * bs
        total_cos += float(cos.detach()) * bs
        total_rmse += float(rmse.detach()) * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "mse": total_mse / max(total_n, 1),
        "sid": total_sid / max(total_n, 1),
        "sis": total_sis / max(total_n, 1),
        "cos": total_cos / max(total_n, 1),
        "rmse": total_rmse / max(total_n, 1),
    }


def save_checkpoint(
    path: str,
    model: nn.Module,
    args: argparse.Namespace,
    spectrum_cols: List,
    epoch: int,
    metrics: Dict,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "out_dim": len(spectrum_cols),
            "dim": args.dim,
            "num_layers": args.layers,
            "num_heads": args.heads,
            "ffn_dim": args.ffn_dim,
            "dropout": args.dropout,
            "max_dist": args.max_dist,
            "use_softplus": not args.no_softplus,
            "smooth_kernel": args.smooth_kernel,
        },
        "spectrum_cols": [str(c) for c in spectrum_cols],
        "phase_vocab": PHASE_VOCAB,
        "epoch": epoch,
        "metrics": metrics,
        "args": vars(args),
    }

    torch.save(payload, path)


def write_metrics_csv(path: str, records: List[Dict]) -> None:
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)


# ============================================================
# Commands
# ============================================================

def cmd_train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)

    dataset = IRDataset(
        csv_path=args.csv,
        smiles_col=args.smiles_col,
        phase_col=args.phase_col,
        spectrum_prefix=args.spectrum_prefix,
        default_phase=args.default_phase,
        target_normalize=args.target_normalize,
        max_dist=args.max_dist,
        require_spectrum=True,
    )

    if len(dataset) < 2:
        raise ValueError("Dataset is too small. Need at least 2 valid molecules for train/val split.")

    if len(dataset.spectrum_cols) == 0:
        raise ValueError("Training requires spectrum columns.")

    val_size = max(1, int(len(dataset) * args.val_fraction))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        train_size = len(dataset) - 1
        val_size = 1

    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed),
    )

    collate_fn = lambda b: collate_graphs(b, max_dist=args.max_dist)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = GraphormerIRLocal(
        out_dim=len(dataset.spectrum_cols),
        dim=args.dim,
        num_layers=args.layers,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        dropout=args.dropout,
        max_dist=args.max_dist,
        use_softplus=not args.no_softplus,
        smooth_kernel=args.smooth_kernel,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )

    amp = bool(args.amp and device.type == "cuda")
    scaler = make_scaler(amp)

    os.makedirs(args.out_dir, exist_ok=True)

    config_path = os.path.join(args.out_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    spectrum_cols_path = os.path.join(args.out_dir, "spectrum_columns.json")
    with open(spectrum_cols_path, "w", encoding="utf-8") as f:
        json.dump([str(c) for c in dataset.spectrum_cols], f, ensure_ascii=False, indent=2)

    best_val = float("inf")
    metrics_records: List[Dict] = []

    print("=" * 60)
    print("Start training")
    print(f"Dataset        : {args.csv}")
    print(f"Train / Val    : {train_size} / {val_size}")
    print(f"Output dim     : {len(dataset.spectrum_cols)}")
    print(f"First/Last col : {dataset.spectrum_cols[0]} / {dataset.spectrum_cols[-1]}")
    print(f"Device         : {device}")
    print(f"AMP            : {amp}")
    print(f"Model dim      : {args.dim}")
    print(f"Layers / Heads : {args.layers} / {args.heads}")
    print(f"FFN dim        : {args.ffn_dim}")
    print(f"Loss weights   : MSE={args.mse_weight}, SID={args.sid_weight}")
    print("=" * 60)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mse_weight=args.mse_weight,
            sid_weight=args.sid_weight,
            amp=amp,
            scaler=scaler,
            grad_clip=args.grad_clip,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            device=device,
            mse_weight=args.mse_weight,
            sid_weight=args.sid_weight,
            amp=amp,
        )

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": train_metrics["loss"],
            "train_mse": train_metrics["mse"],
            "train_sid": train_metrics["sid"],
            "train_sis": train_metrics["sis"],
            "train_cos": train_metrics["cos"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mse": val_metrics["mse"],
            "val_sid": val_metrics["sid"],
            "val_sis": val_metrics["sis"],
            "val_cos": val_metrics["cos"],
            "val_rmse": val_metrics["rmse"],
        }
        metrics_records.append(record)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr {current_lr:.3e} | "
            f"train loss {train_metrics['loss']:.6f} "
            f"mse {train_metrics['mse']:.6f} "
            f"sid {train_metrics['sid']:.6f} "
            f"sis {train_metrics['sis']:.6f} "
            f"cos {train_metrics['cos']:.6f} "
            f"rmse {train_metrics['rmse']:.6f} | "
            f"val loss {val_metrics['loss']:.6f} "
            f"mse {val_metrics['mse']:.6f} "
            f"sid {val_metrics['sid']:.6f} "
            f"sis {val_metrics['sis']:.6f} "
            f"cos {val_metrics['cos']:.6f} "
            f"rmse {val_metrics['rmse']:.6f}"
        )

        metrics_path = os.path.join(args.out_dir, "training_metrics.csv")
        write_metrics_csv(metrics_path, metrics_records)

        last_path = os.path.join(args.out_dir, "last_model.pt")
        save_checkpoint(
            last_path,
            model=model,
            args=args,
            spectrum_cols=dataset.spectrum_cols,
            epoch=epoch,
            metrics={"train": train_metrics, "val": val_metrics},
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_path = os.path.join(args.out_dir, "best_model.pt")
            save_checkpoint(
                best_path,
                model=model,
                args=args,
                spectrum_cols=dataset.spectrum_cols,
                epoch=epoch,
                metrics={"train": train_metrics, "val": val_metrics},
            )
            print(f"  Saved best checkpoint: {best_path}")

    print("=" * 60)
    print("Training finished.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Output dir: {args.out_dir}")
    print("=" * 60)


@torch.no_grad()
def cmd_predict(args: argparse.Namespace) -> None:
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_config = ckpt["model_config"]
    spectrum_cols = ckpt["spectrum_cols"]

    dataset = IRDataset(
        csv_path=args.csv,
        smiles_col=args.smiles_col,
        phase_col=args.phase_col,
        spectrum_prefix=args.spectrum_prefix,
        default_phase=args.default_phase,
        target_normalize="none",
        max_dist=model_config.get("max_dist", 20),
        require_spectrum=False,
    )

    collate_fn = lambda b: collate_graphs(
        b,
        max_dist=model_config.get("max_dist", 20),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
        drop_last=False,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model = GraphormerIRLocal(**model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    amp = bool(args.amp and device.type == "cuda")

    rows: List[Dict] = []

    print("=" * 60)
    print("Start prediction")
    print(f"Input CSV  : {args.csv}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Output CSV : {args.out}")
    print(f"Device     : {device}")
    print(f"AMP        : {amp}")
    print(f"N samples  : {len(dataset)}")
    print(f"Out dim    : {len(spectrum_cols)}")
    print("=" * 60)

    for batch in loader:
        batch = move_batch_to_device(batch, device)

        with make_amp_context(amp):
            pred = model(batch)

        pred_np = pred.detach().cpu().numpy()

        for i in range(pred_np.shape[0]):
            row = {
                "row_index": batch["row_indices"][i],
                "smiles": batch["smiles"][i],
                "phase": batch["phases"][i],
            }
            for j, col in enumerate(spectrum_cols):
                row[f"pred_{col}"] = float(pred_np[i, j])
            rows.append(row)

    out_df = pd.DataFrame(rows)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Saved predictions to: {args.out}")


def cmd_inspect(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.csv, nrows=args.nrows)
    smiles_col = find_smiles_col(df, args.smiles_col)
    phase_col = find_phase_col(df, args.phase_col)

    exclude_cols = {smiles_col}
    if phase_col is not None:
        exclude_cols.add(phase_col)

    spectrum_cols = find_spectrum_cols(
        df,
        prefix=args.spectrum_prefix,
        exclude_cols=exclude_cols,
    )

    print("=" * 60)
    print(f"CSV: {args.csv}")
    print(f"Shape preview: {df.shape}")
    print(f"SMILES column: {smiles_col}")
    print(f"Phase column : {phase_col}")
    print(f"Detected spectrum columns: {len(spectrum_cols)}")
    if spectrum_cols:
        print(f"First 10 spectrum columns: {spectrum_cols[:10]}")
        print(f"Last 10 spectrum columns : {spectrum_cols[-10:]}")
    print("=" * 60)
    print("All columns preview:")
    for i, c in enumerate(df.columns):
        print(i, repr(c), df[c].dtype)


# ============================================================
# CLI
# ============================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pure PyTorch Graphormer-IR local refactor."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------------
    # inspect
    # -------------------------
    inspect = subparsers.add_parser("inspect")
    inspect.add_argument("--csv", required=True)
    inspect.add_argument("--smiles-col", default=None)
    inspect.add_argument("--phase-col", default=None)
    inspect.add_argument("--spectrum-prefix", default="sp_")
    inspect.add_argument("--nrows", type=int, default=5)

    # -------------------------
    # train
    # -------------------------
    train = subparsers.add_parser("train")
    train.add_argument("--csv", required=True)
    train.add_argument("--out-dir", default="checkpoints_local_pytorch")
    train.add_argument("--smiles-col", default=None)
    train.add_argument("--phase-col", default=None)
    train.add_argument("--spectrum-prefix", default="sp_")
    train.add_argument("--default-phase", default="KBr")
    train.add_argument(
        "--target-normalize",
        choices=["none", "row_minmax", "row_max"],
        default="none",
    )

    train.add_argument("--device", default="cuda")
    train.add_argument("--seed", type=int, default=23)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--batch-size", type=int, default=8)
    train.add_argument("--epochs", type=int, default=20)
    train.add_argument("--val-fraction", type=float, default=0.1)

    train.add_argument("--dim", type=int, default=256)
    train.add_argument("--layers", type=int, default=6)
    train.add_argument("--heads", type=int, default=8)
    train.add_argument("--ffn-dim", type=int, default=512)
    train.add_argument("--dropout", type=float, default=0.1)
    train.add_argument("--max-dist", type=int, default=20)
    train.add_argument("--smooth-kernel", type=int, default=7)

    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--weight-decay", type=float, default=1e-4)
    train.add_argument("--mse-weight", type=float, default=1.0)
    train.add_argument("--sid-weight", type=float, default=0.1)
    train.add_argument("--grad-clip", type=float, default=5.0)
    train.add_argument("--amp", action="store_true")
    train.add_argument("--no-softplus", action="store_true")

    # -------------------------
    # predict
    # -------------------------
    pred = subparsers.add_parser("predict")
    pred.add_argument("--csv", required=True)
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--out", required=True)
    pred.add_argument("--smiles-col", default=None)
    pred.add_argument("--phase-col", default=None)
    pred.add_argument("--spectrum-prefix", default="sp_")
    pred.add_argument("--default-phase", default="KBr")
    pred.add_argument("--device", default="cuda")
    pred.add_argument("--num-workers", type=int, default=4)
    pred.add_argument("--batch-size", type=int, default=32)
    pred.add_argument("--amp", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        cmd_inspect(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    else:
        raise ValueError(args.command)


if __name__ == "__main__":
    main()
