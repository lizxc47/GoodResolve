#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Complete Stage III reranking script.

Updated version:
  - Supports phase-aware Model III checkpoints.
  - Reads the phase column from IR_data.csv and passes it into Model III.
  - Supports both sp_400...sp_4000 and numeric columns 400,402,...,4000.

For each data* folder, this script reads:
    IR_data.csv
    generation_*_elites.csv, using the largest *

and produces a final Stage III ranking using:

    final_score =
        w_sis     * SIS_norm
      + w_model3  * ModelIII_norm
      + w_feature * selected_feature_norm

where selected_feature is specified by --feature.

Supported candidate-only SMILES features:
    mw_asc          : molecular weight, smaller is better
    exactmw_asc     : exact molecular weight, smaller is better
    heavy_asc       : heavy atom count, smaller is better
    bertz_asc       : BertzCT graph complexity, smaller is better
    rotbond_asc     : rotatable bond count, smaller is better
    tpsa_asc        : TPSA, smaller is better
    hetero_asc      : hetero atom count, smaller is better
    ring_asc        : ring count, smaller is better
    aromring_asc    : aromatic ring count, smaller is better
    hba_asc         : H-bond acceptor count, smaller is better
    hbd_asc         : H-bond donor count, smaller is better

Default setting uses the best single-feature ratio we found:
    feature = heavy_asc
    weights = 0.15 SIS + 0.15 ModelIII + 0.70 heavy_asc

Example:
    CUDA_VISIBLE_DEVICES=1 python stage3_final_rerank.py \
        --root all \
        --checkpoint model3_stage3_runs/ablation_1_gnn_base/best_model3_gnn.pt \
        --signature-vocab model3_stage3_data/signature_vocab.csv \
        --top-k 200 \
        --feature heavy_asc \
        --w-sis 0.15 \
        --w-model3 0.15 \
        --w-feature 0.70 \
        --device cuda \
        --amp

Outputs:
    per data* folder:
        stage3_final_<feature>_topK_wSIS..._wM3..._wF....csv

    root:
        stage3_final_<feature>_topK_wSIS..._wM3..._wF..._summary.csv
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
except ImportError as e:
    raise ImportError(
        "RDKit is required. Install with:\n"
        "conda install -c conda-forge rdkit"
    ) from e

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# Functional-group / TRIAD signature extraction
# ============================================================

FG_SMARTS = {
    "fg_aromatic_ring": "a1aaaaa1",
    "fg_hetero_aromatic": "[a;!#6]",
    "fg_alkene": "[CX3]=[CX3]",
    "fg_alkyne": "[CX2]#[CX2]",
    "fg_carbonyl": "[CX3]=[OX1]",
    "fg_aldehyde": "[CX3H1](=O)[#6,#1]",
    "fg_ketone": "[#6][CX3](=O)[#6]",
    "fg_carboxylic_acid": "[CX3](=O)[OX2H1]",
    "fg_ester": "[CX3](=O)[OX2][#6]",
    "fg_amide": "[NX3][CX3](=[OX1])",
    "fg_anhydride": "[CX3](=O)O[CX3](=O)",
    "fg_acid_chloride": "[CX3](=O)Cl",
    "fg_alcohol_phenol": "[OX2H][#6]",
    "fg_ether": "[OD2]([#6])[#6]",
    "fg_primary_amine": "[NX3H2][#6]",
    "fg_secondary_amine": "[NX3H1]([#6])[#6]",
    "fg_tertiary_amine": "[NX3H0]([#6])([#6])[#6]",
    "fg_nitrile": "[CX2]#N",
    "fg_nitro": "[$([NX3](=O)=O),$([NX3+](=O)[O-])]",
    "fg_imine": "[CX3]=[NX2]",
    "fg_halogen": "[F,Cl,Br,I]",
    "fg_fluoro": "[F]",
    "fg_chloro": "[Cl]",
    "fg_bromo": "[Br]",
    "fg_iodo": "[I]",
    "fg_thiol": "[SX2H]",
    "fg_sulfide": "[SX2]([#6])[#6]",
    "fg_sulfoxide": "[SX3](=O)([#6])[#6]",
    "fg_sulfone": "[SX4](=O)(=O)([#6])[#6]",
    "fg_sulfonamide": "[SX4](=O)(=O)(N)",
    "fg_phosphate_phosphonate": "[PX4](=O)(O)(O)",
    "fg_silicon": "[Si]",
}

SMARTS_PATTERNS = {k: Chem.MolFromSmarts(v) for k, v in FG_SMARTS.items() if Chem.MolFromSmarts(v) is not None}


def atom_token(atom):
    symbol = atom.GetSymbol()
    aromatic = "a" if atom.GetIsAromatic() else ""
    charge = atom.GetFormalCharge()
    charge_str = f"{charge:+d}" if charge != 0 else ""
    return f"{symbol}{aromatic}{charge_str}"


def bond_token(bond):
    if bond.GetIsAromatic():
        return ":"
    bt = bond.GetBondType()
    if bt == Chem.BondType.SINGLE:
        return "-"
    if bt == Chem.BondType.DOUBLE:
        return "="
    if bt == Chem.BondType.TRIPLE:
        return "#"
    return "~"


def extract_triad_signatures(mol):
    sigs = set()
    if mol is None:
        return sigs

    for center in mol.GetAtoms():
        neighbors = list(center.GetNeighbors())
        if len(neighbors) < 2:
            continue

        c_idx = center.GetIdx()
        c_tok = atom_token(center)

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                a = neighbors[i]
                b = neighbors[j]

                bond_ac = mol.GetBondBetweenAtoms(a.GetIdx(), c_idx)
                bond_cb = mol.GetBondBetweenAtoms(c_idx, b.GetIdx())
                if bond_ac is None or bond_cb is None:
                    continue

                left = atom_token(a) + bond_token(bond_ac) + c_tok + bond_token(bond_cb) + atom_token(b)
                right = atom_token(b) + bond_token(bond_cb) + c_tok + bond_token(bond_ac) + atom_token(a)
                sigs.add("triad_" + min(left, right))

    return sigs


def extract_signature_names(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    sigs = set()
    if mol is None:
        return sigs

    for name, patt in SMARTS_PATTERNS.items():
        try:
            if mol.HasSubstructMatch(patt):
                sigs.add(name)
        except Exception:
            pass

    sigs.update(extract_triad_signatures(mol))
    return sigs


def signature_vector_from_smiles(smiles, signature_names):
    sigs = extract_signature_names(smiles)
    return np.array([1.0 if s in sigs else 0.0 for s in signature_names], dtype=np.float32)


# ============================================================
# Molecular graph featurizer
# ============================================================

ATOM_LIST = [0, 5, 6, 7, 8, 9, 14, 15, 16, 17, 35, 53]
ATOM_TO_IDX = {z: i for i, z in enumerate(ATOM_LIST)}

HYB_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
HYB_TO_IDX = {h: i for i, h in enumerate(HYB_LIST)}

BOND_TO_IDX = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
}


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))

    if mol is None or mol.GetNumAtoms() == 0:
        return {
            "atom_num": np.array([0], dtype=np.int64),
            "degree": np.array([0], dtype=np.int64),
            "charge": np.array([2], dtype=np.int64),
            "hyb": np.array([len(HYB_LIST)], dtype=np.int64),
            "h_count": np.array([0], dtype=np.int64),
            "aromatic": np.array([0.0], dtype=np.float32),
            "ring": np.array([0.0], dtype=np.float32),
            "edge_src": np.array([], dtype=np.int64),
            "edge_dst": np.array([], dtype=np.int64),
            "edge_type": np.array([], dtype=np.int64),
            "num_nodes": 1,
        }

    atom_num, degree, charge, hyb, h_count, aromatic, ring = [], [], [], [], [], [], []
    for atom in mol.GetAtoms():
        atom_num.append(ATOM_TO_IDX.get(atom.GetAtomicNum(), 0))
        degree.append(min(int(atom.GetDegree()), 6))
        chg = max(-2, min(2, int(atom.GetFormalCharge())))
        charge.append(chg + 2)
        hyb.append(HYB_TO_IDX.get(atom.GetHybridization(), len(HYB_LIST)))
        h_count.append(min(int(atom.GetTotalNumHs()), 5))
        aromatic.append(1.0 if atom.GetIsAromatic() else 0.0)
        ring.append(1.0 if atom.IsInRing() else 0.0)

    edge_src, edge_dst, edge_type = [], [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bt = BOND_TO_IDX.get(bond.GetBondType(), 4)

        edge_src.extend([i, j])
        edge_dst.extend([j, i])
        edge_type.extend([bt, bt])

    return {
        "atom_num": np.array(atom_num, dtype=np.int64),
        "degree": np.array(degree, dtype=np.int64),
        "charge": np.array(charge, dtype=np.int64),
        "hyb": np.array(hyb, dtype=np.int64),
        "h_count": np.array(h_count, dtype=np.int64),
        "aromatic": np.array(aromatic, dtype=np.float32),
        "ring": np.array(ring, dtype=np.float32),
        "edge_src": np.array(edge_src, dtype=np.int64),
        "edge_dst": np.array(edge_dst, dtype=np.int64),
        "edge_type": np.array(edge_type, dtype=np.int64),
        "num_nodes": mol.GetNumAtoms(),
    }


def make_graph_batch(graphs, device):
    atom_num, degree, charge, hyb, h_count, aromatic, ring = [], [], [], [], [], [], []
    edge_src, edge_dst, edge_type = [], [], []
    node_to_graph = []

    offset = 0
    for g_idx, g in enumerate(graphs):
        n = int(g["num_nodes"])
        atom_num.append(g["atom_num"])
        degree.append(g["degree"])
        charge.append(g["charge"])
        hyb.append(g["hyb"])
        h_count.append(g["h_count"])
        aromatic.append(g["aromatic"])
        ring.append(g["ring"])
        node_to_graph.append(np.full(n, g_idx, dtype=np.int64))

        if len(g["edge_src"]) > 0:
            edge_src.append(g["edge_src"] + offset)
            edge_dst.append(g["edge_dst"] + offset)
            edge_type.append(g["edge_type"])
        offset += n

    atom_num = np.concatenate(atom_num)
    degree = np.concatenate(degree)
    charge = np.concatenate(charge)
    hyb = np.concatenate(hyb)
    h_count = np.concatenate(h_count)
    aromatic = np.concatenate(aromatic)
    ring = np.concatenate(ring)
    node_to_graph = np.concatenate(node_to_graph)

    if len(edge_src) > 0:
        edge_src = np.concatenate(edge_src)
        edge_dst = np.concatenate(edge_dst)
        edge_type = np.concatenate(edge_type)
    else:
        edge_src = np.array([], dtype=np.int64)
        edge_dst = np.array([], dtype=np.int64)
        edge_type = np.array([], dtype=np.int64)

    return {
        "atom_num": torch.tensor(atom_num, dtype=torch.long, device=device),
        "degree": torch.tensor(degree, dtype=torch.long, device=device),
        "charge": torch.tensor(charge, dtype=torch.long, device=device),
        "hyb": torch.tensor(hyb, dtype=torch.long, device=device),
        "h_count": torch.tensor(h_count, dtype=torch.long, device=device),
        "aromatic": torch.tensor(aromatic, dtype=torch.float32, device=device),
        "ring": torch.tensor(ring, dtype=torch.float32, device=device),
        "edge_src": torch.tensor(edge_src, dtype=torch.long, device=device),
        "edge_dst": torch.tensor(edge_dst, dtype=torch.long, device=device),
        "edge_type": torch.tensor(edge_type, dtype=torch.long, device=device),
        "node_to_graph": torch.tensor(node_to_graph, dtype=torch.long, device=device),
        "num_graphs": len(graphs),
    }


# ============================================================
# Model III GNN
# ============================================================

class SpectrumEncoder(nn.Module):
    def __init__(self, spec_len, embed_dim=256, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.conv(x)
        h_avg = F.adaptive_avg_pool1d(h, 1).squeeze(-1)
        h_max = F.adaptive_max_pool1d(h, 1).squeeze(-1)
        return self.fc(torch.cat([h_avg, h_max], dim=-1))


class AtomEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.atom_emb = nn.Embedding(len(ATOM_LIST), 32)
        self.degree_emb = nn.Embedding(7, 16)
        self.charge_emb = nn.Embedding(5, 16)
        self.hyb_emb = nn.Embedding(len(HYB_LIST) + 1, 16)
        self.h_emb = nn.Embedding(6, 16)
        self.proj = nn.Sequential(
            nn.Linear(32 + 16 + 16 + 16 + 16 + 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, graph):
        x = torch.cat(
            [
                self.atom_emb(graph["atom_num"]),
                self.degree_emb(graph["degree"]),
                self.charge_emb(graph["charge"]),
                self.hyb_emb(graph["hyb"]),
                self.h_emb(graph["h_count"]),
                graph["aromatic"].unsqueeze(-1),
                graph["ring"].unsqueeze(-1),
            ],
            dim=-1,
        )
        return self.proj(x)


class GNNLayer(nn.Module):
    def __init__(self, hidden_dim, num_bond_types=5, dropout=0.1):
        super().__init__()
        self.bond_emb = nn.Embedding(num_bond_types, hidden_dim)
        self.msg_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h, edge_src, edge_dst, edge_type):
        if edge_src.numel() == 0:
            return h
        msg = self.msg_mlp(h[edge_src] + self.bond_emb(edge_type))
        agg = torch.zeros(h.size(), dtype=msg.dtype, device=h.device)
        agg.index_add_(0, edge_dst, msg)
        agg = agg.to(dtype=h.dtype)
        h_new = self.update_mlp(h + agg).to(dtype=h.dtype)
        return self.norm(h + self.dropout(h_new))


class CandidateGNNEncoder(nn.Module):
    def __init__(self, gnn_hidden_dim=256, embed_dim=256, num_layers=4, dropout=0.2):
        super().__init__()
        self.atom_encoder = AtomEncoder(gnn_hidden_dim)
        self.layers = nn.ModuleList([GNNLayer(gnn_hidden_dim, dropout=dropout) for _ in range(num_layers)])
        self.readout = nn.Sequential(
            nn.Linear(gnn_hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, graph):
        h = self.atom_encoder(graph)
        for layer in self.layers:
            h = layer(h, graph["edge_src"], graph["edge_dst"], graph["edge_type"])

        node_to_graph = graph["node_to_graph"]
        num_graphs = graph["num_graphs"]

        graph_sum = torch.zeros(num_graphs, h.size(-1), device=h.device, dtype=h.dtype)
        graph_sum.index_add_(0, node_to_graph, h)

        counts = torch.zeros(num_graphs, device=h.device, dtype=h.dtype)
        counts.index_add_(0, node_to_graph, torch.ones_like(node_to_graph, dtype=h.dtype))

        return self.readout(graph_sum / counts.clamp(min=1.0).unsqueeze(-1))


class ModelIII_GNN(nn.Module):
    def __init__(
        self,
        spec_len,
        n_signatures,
        spec_embed_dim=256,
        mol_embed_dim=256,
        gnn_hidden_dim=256,
        gnn_layers=4,
        hidden_dim=512,
        dropout=0.2,
        use_sig_importance=False,
        num_phases=1,
    ):
        super().__init__()
        self.n_signatures = n_signatures
        self.use_sig_importance = use_sig_importance

        self.spectrum_encoder = SpectrumEncoder(spec_len, spec_embed_dim, dropout)
        self.phase_embedding = nn.Embedding(max(int(num_phases), 1), spec_embed_dim)
        self.phase_norm = nn.LayerNorm(spec_embed_dim)

        self.signature_head = nn.Sequential(
            nn.Linear(spec_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_signatures),
        )

        self.candidate_encoder = CandidateGNNEncoder(
            gnn_hidden_dim=gnn_hidden_dim,
            embed_dim=mol_embed_dim,
            num_layers=gnn_layers,
            dropout=dropout,
        )

        input_dim = spec_embed_dim + mol_embed_dim + 5 * n_signatures

        if self.use_sig_importance:
            self.sig_importance_head = nn.Sequential(
                nn.Linear(spec_embed_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, n_signatures),
            )
            input_dim += 3 * n_signatures + 6

        self.consistency_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, spectra, phase_ids, graph_batch, cand_sig, group_index):
        z_spec = self.spectrum_encoder(spectra)
        if phase_ids is None:
            phase_ids = torch.zeros(spectra.size(0), dtype=torch.long, device=spectra.device)
        z_spec = self.phase_norm(z_spec + self.phase_embedding(phase_ids.long()))
        sig_logits = self.signature_head(z_spec)
        p_sig = torch.sigmoid(sig_logits)

        z_spec_pair = z_spec[group_index]
        p_sig_pair = p_sig[group_index]

        z_mol = self.candidate_encoder(graph_batch)
        y_sig = cand_sig

        matched = p_sig_pair * y_sig
        missing = p_sig_pair * (1.0 - y_sig)
        extra = (1.0 - p_sig_pair) * y_sig

        feat_list = [z_spec_pair, z_mol, p_sig_pair, y_sig, matched, missing, extra]

        if self.use_sig_importance:
            sig_w = torch.sigmoid(self.sig_importance_head(z_spec_pair))
            weighted_matched = sig_w * matched
            weighted_missing = sig_w * missing
            weighted_extra = sig_w * extra
            sig_summary = torch.cat(
                [
                    weighted_matched.sum(dim=1, keepdim=True),
                    weighted_missing.sum(dim=1, keepdim=True),
                    weighted_extra.sum(dim=1, keepdim=True),
                    weighted_matched.mean(dim=1, keepdim=True),
                    weighted_missing.mean(dim=1, keepdim=True),
                    weighted_extra.mean(dim=1, keepdim=True),
                ],
                dim=-1,
            )
            feat_list.extend([weighted_matched, weighted_missing, weighted_extra, sig_summary])

        feat = torch.cat(feat_list, dim=-1)
        return self.consistency_head(feat).squeeze(-1), sig_logits


# ============================================================
# Utility functions
# ============================================================

FEATURES = {
    "mw_asc": "mw",
    "exactmw_asc": "exactmw",
    "heavy_asc": "heavy",
    "bertz_asc": "bertz",
    "rotbond_asc": "rotbond",
    "tpsa_asc": "tpsa",
    "hetero_asc": "hetero",
    "ring_asc": "ring",
    "aromring_asc": "aromring",
    "hba_asc": "hba",
    "hbd_asc": "hbd",
}


def canonical_smiles(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def inchikey(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def mol_descriptors(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return {k: np.nan for k in ["mw", "exactmw", "heavy", "bertz", "rotbond", "tpsa", "hetero", "ring", "aromring", "hba", "hbd"]}

    atoms = list(mol.GetAtoms())

    def safe(fn):
        try:
            return float(fn())
        except Exception:
            return np.nan

    return {
        "mw": safe(lambda: Descriptors.MolWt(mol)),
        "exactmw": safe(lambda: Descriptors.ExactMolWt(mol)),
        "heavy": safe(lambda: mol.GetNumHeavyAtoms()),
        "bertz": safe(lambda: Descriptors.BertzCT(mol)),
        "rotbond": safe(lambda: rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "tpsa": safe(lambda: rdMolDescriptors.CalcTPSA(mol)),
        "hetero": safe(lambda: sum(1 for a in atoms if a.GetAtomicNum() not in (1, 6))),
        "ring": safe(lambda: mol.GetRingInfo().NumRings()),
        "aromring": safe(lambda: rdMolDescriptors.CalcNumAromaticRings(mol)),
        "hba": safe(lambda: rdMolDescriptors.CalcNumHBA(mol)),
        "hbd": safe(lambda: rdMolDescriptors.CalcNumHBD(mol)),
    }


def normalize_minmax(values, higher_is_better=True, eps=1e-12):
    x = np.asarray(values, dtype=np.float64)
    valid = np.isfinite(x)

    if not valid.any():
        return np.zeros_like(x, dtype=np.float32)

    y = x.copy()
    if not higher_is_better:
        y = -y

    min_v = np.nanmin(y[valid])
    max_v = np.nanmax(y[valid])

    if abs(max_v - min_v) < eps:
        out = np.full_like(y, 0.5, dtype=np.float64)
    else:
        out = (y - min_v) / (max_v - min_v)

    out[~valid] = 0.0
    return out.astype(np.float32)


def infer_sis_col(df, preferred):
    if preferred in df.columns:
        return preferred

    for c in df.columns:
        if str(c).lower() == "sis":
            return c

    for c in df.columns:
        if "sis" in str(c).lower():
            print(f"Warning: SIS column {preferred!r} not found. Using inferred column: {c}")
            return c

    raise ValueError(f"Cannot find SIS column {preferred!r}. Available columns: {list(df.columns)[:30]}")


def normalize_phase_value(x, default_phase="unknown"):
    """Normalize measurement/theoretical phase strings into categorical tokens."""
    if pd.isna(x):
        return str(default_phase)
    s = str(x).strip()
    if not s:
        return str(default_phase)
    low = s.lower()
    if low in {"kbr", "kbr disc", "kbr disk", "kbr pellet"}:
        return "KBr"
    if low in {"nujol", "nujol mull"}:
        return "nujol mull"
    if low in {"liquid", "liquid film"}:
        return "liquid film"
    if low in {"gas", "gas phase", "gas-phase"}:
        return "gas"
    if low in {"theoretical", "simulated", "simulation"}:
        return "theoretical"
    return s


def phase_to_id(phase, phase_vocab, default_phase="unknown"):
    phase = normalize_phase_value(phase, default_phase=default_phase)
    if phase in phase_vocab:
        return int(phase_vocab[phase])
    if default_phase in phase_vocab:
        return int(phase_vocab[default_phase])
    if "unknown" in phase_vocab:
        return int(phase_vocab["unknown"])
    return 0


def resolve_column_name(df, col):
    """Resolve exact, string, or integer-like column names."""
    if col in df.columns:
        return col
    s = str(col)
    if s in df.columns:
        return s
    try:
        i = int(float(s))
        if i in df.columns:
            return i
        if str(i) in df.columns:
            return str(i)
    except Exception:
        pass
    return None


def get_sp_cols_from_checkpoint(meta, fallback_start=50, fallback_end=1750):
    """Get spectral columns from checkpoint metadata.

    New data-preparation scripts may store numeric column names such as
    400, 402, ..., 4000 rather than sp_400, sp_402, ... . This function
    preserves the metadata names exactly when available.
    """
    sp_cols = meta.get("sp_cols") or meta.get("spectral_columns") or meta.get("spectrum_cols")
    if sp_cols:
        return [str(c) for c in list(sp_cols)]
    return [f"sp_{i}" for i in range(fallback_start, fallback_end + 1)]


def load_model(checkpoint, device):
    ckpt = torch.load(checkpoint, map_location=device)
    args = ckpt.get("args", {})
    meta = ckpt.get("meta", {})

    spec_len = int(meta.get("spec_len", len(meta.get("sp_cols", []))))
    n_signatures = int(meta.get("n_signatures", len(meta.get("sig_cols", []))))

    if spec_len <= 0 or n_signatures <= 0:
        raise ValueError("Checkpoint meta lacks spec_len/n_signatures.")

    model = ModelIII_GNN(
        spec_len=spec_len,
        n_signatures=n_signatures,
        spec_embed_dim=int(args.get("spec_embed_dim", 256)),
        mol_embed_dim=int(args.get("mol_embed_dim", 256)),
        gnn_hidden_dim=int(args.get("gnn_hidden_dim", 256)),
        gnn_layers=int(args.get("gnn_layers", 4)),
        hidden_dim=int(args.get("hidden_dim", 512)),
        dropout=float(args.get("dropout", 0.2)),
        use_sig_importance=bool(args.get("use_sig_importance", False)),
        num_phases=int(meta.get("num_phases", len(meta.get("phase_vocab", {})) or 1)),
    ).to(device)

    load_info = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    missing = list(getattr(load_info, "missing_keys", []))
    unexpected = list(getattr(load_info, "unexpected_keys", []))
    if missing:
        print(f"Warning: missing checkpoint keys when loading Model III: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if unexpected:
        print(f"Warning: unexpected checkpoint keys when loading Model III: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    model.eval()
    return model, meta


@torch.no_grad()
def score_model3(model, spectrum, phase_id, smiles_list, sig_mat, device, batch_size=256, amp=False):
    spectrum_tensor = torch.tensor(spectrum[None, :], dtype=torch.float32, device=device)
    phase_tensor = torch.tensor([int(phase_id)], dtype=torch.long, device=device)

    all_scores = []
    all_logits = []

    for start in range(0, len(smiles_list), batch_size):
        end = min(start + batch_size, len(smiles_list))
        smi_batch = smiles_list[start:end]
        sig_batch = sig_mat[start:end]

        graphs = [smiles_to_graph(smi) for smi in smi_batch]
        graph_batch = make_graph_batch(graphs, device)
        cand_sig = torch.tensor(sig_batch, dtype=torch.float32, device=device)
        group_index = torch.zeros(len(smi_batch), dtype=torch.long, device=device)

        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            logits, _ = model(spectrum_tensor, phase_tensor, graph_batch, cand_sig, group_index)

        scores = torch.sigmoid(logits)
        all_scores.extend(scores.detach().cpu().numpy().tolist())
        all_logits.extend(logits.detach().cpu().numpy().tolist())

    return np.asarray(all_scores, dtype=np.float32), np.asarray(all_logits, dtype=np.float32)


def read_ir_row(ir_path, sp_cols, smiles_col="SMILES", row_index=0, phase_col="phase", default_phase="unknown"):
    df = pd.read_csv(ir_path)

    resolved_smiles_col = resolve_column_name(df, smiles_col)
    if resolved_smiles_col is None:
        # Common fallback because many of our Graphormer-format files use lower-case smiles.
        resolved_smiles_col = resolve_column_name(df, "smiles")
    if resolved_smiles_col is None:
        raise ValueError(f"{ir_path} does not contain SMILES column {smiles_col!r} or 'smiles'.")

    resolved_sp_cols = []
    missing = []
    for c in sp_cols:
        rc = resolve_column_name(df, c)
        if rc is None:
            missing.append(c)
        else:
            resolved_sp_cols.append(rc)
    if missing:
        raise ValueError(f"{ir_path} missing spectral columns. First missing: {missing[:10]}")

    row = df.iloc[row_index]
    phase = default_phase
    resolved_phase_col = resolve_column_name(df, phase_col)
    if resolved_phase_col is not None:
        phase = normalize_phase_value(row[resolved_phase_col], default_phase=default_phase)

    return str(row[resolved_smiles_col]), row[resolved_sp_cols].values.astype(np.float32), phase


def safe_name(x):
    return str(x).replace(".", "p").replace("-", "m")



def find_candidate_file(folder, candidate_filename=None, candidate_glob="generation_*_elites.csv"):
    """
    Select candidate file from a data folder.

    If candidate_filename is provided, use that exact file.
    Otherwise, search generation_*_elites.csv and choose the file with the
    largest numeric generation index.

    Examples:
        generation_5_elites.csv
        generation_*_elites.csv, using the largest *
        generation_100_elites.csv

    returns:
        Path object or None
    """
    folder = Path(folder)

    if candidate_filename:
        p = folder / candidate_filename
        return p if p.exists() else None

    paths = list(folder.glob(candidate_glob))
    if not paths:
        return None

    numbered = []
    unnumbered = []

    pattern = re.compile(r"generation_(\d+)_elites\.csv$")
    for p in paths:
        m = pattern.search(p.name)
        if m:
            numbered.append((int(m.group(1)), p))
        else:
            unnumbered.append(p)

    if numbered:
        numbered.sort(key=lambda x: x[0], reverse=True)
        return numbered[0][1]

    # Fallback: no numeric generation found, use lexicographically last file.
    return sorted(unnumbered)[-1] if unnumbered else None



# ============================================================
# Folder processing
# ============================================================

def rerank_one_folder(folder, args, model, sp_cols, signature_names, device):
    folder = Path(folder)
    ir_path = folder / args.ir_filename
    cand_path = find_candidate_file(folder, args.candidate_filename, args.candidate_glob)

    if not ir_path.exists():
        return {"folder": str(folder), "status": "missing_ir", "message": f"Missing {args.ir_filename}"}

    if cand_path is None or not cand_path.exists():
        return {
            "folder": str(folder),
            "status": "missing_candidates",
            "message": f"No candidate file found. Pattern: {args.candidate_glob}; exact: {args.candidate_filename}",
        }

    true_smiles, spectrum, target_phase = read_ir_row(
        ir_path,
        sp_cols,
        smiles_col=args.true_smiles_col,
        row_index=args.ir_row_index,
        phase_col=args.phase_col,
        default_phase=args.default_phase,
    )
    target_phase_id = phase_to_id(target_phase, args.phase_vocab, default_phase=args.default_phase)
    true_can = canonical_smiles(true_smiles)
    true_key = inchikey(true_smiles)

    cand_df = pd.read_csv(cand_path)

    if args.candidate_smiles_col not in cand_df.columns:
        raise ValueError(f"{cand_path} lacks candidate SMILES column: {args.candidate_smiles_col}")

    sis_col = infer_sis_col(cand_df, args.sis_col)

    cand_df = cand_df.copy()
    cand_df["target_phase"] = target_phase
    cand_df["target_phase_id"] = target_phase_id
    cand_df["stage2_rank"] = np.arange(1, len(cand_df) + 1)
    cand_df["candidate_smiles"] = cand_df[args.candidate_smiles_col].astype(str)
    cand_df["stage2_sis"] = pd.to_numeric(cand_df[sis_col], errors="coerce")

    cans, keys, valid = [], [], []
    desc_raw = []

    for smi in cand_df["candidate_smiles"].tolist():
        can = canonical_smiles(smi)
        key = inchikey(smi)
        cans.append(can)
        keys.append(key)
        valid.append(can is not None and key is not None)
        desc_raw.append(mol_descriptors(smi))

    cand_df["candidate_canonical_smiles"] = cans
    cand_df["candidate_inchikey"] = keys
    cand_df["valid_candidate"] = valid

    if true_key is not None:
        cand_df["is_true"] = cand_df["candidate_inchikey"] == true_key
    else:
        cand_df["is_true"] = cand_df["candidate_canonical_smiles"] == true_can

    true_rows_stage2 = cand_df[cand_df["is_true"] == True]
    true_found_all = len(true_rows_stage2) > 0
    true_stage2_rank = int(true_rows_stage2["stage2_rank"].iloc[0]) if true_found_all else None

    top_mask = cand_df["stage2_rank"] <= args.top_k if args.top_k > 0 else np.ones(len(cand_df), dtype=bool)
    top_df = cand_df[top_mask].copy()
    rest_df = cand_df[~top_mask].copy()

    if len(top_df) == 0:
        return {"folder": str(folder), "status": "no_candidates", "message": "No candidates in selected top-K"}

    # Feature raw and normalized values for whole candidate table.
    raw_name = FEATURES[args.feature]
    feature_raw_all = np.array([d[raw_name] for d in desc_raw], dtype=np.float64)
    cand_df[f"{args.feature}_raw"] = feature_raw_all

    # Normalize only within top-K.
    top_indices = top_df.index.to_numpy()
    sis_norm_top = normalize_minmax(cand_df.loc[top_indices, "stage2_sis"].values, higher_is_better=True)
    feat_norm_top = normalize_minmax(cand_df.loc[top_indices, f"{args.feature}_raw"].values, higher_is_better=False)

    cand_df["sis_norm"] = np.nan
    cand_df[f"{args.feature}_norm"] = np.nan
    cand_df.loc[top_indices, "sis_norm"] = sis_norm_top
    cand_df.loc[top_indices, f"{args.feature}_norm"] = feat_norm_top

    # Model III for valid top-K candidates.
    valid_top_df = cand_df.loc[top_indices]
    valid_top_df = valid_top_df[valid_top_df["valid_candidate"] == True].copy()
    unique_smiles = valid_top_df["candidate_canonical_smiles"].dropna().drop_duplicates().tolist()

    score_map = {}
    logit_map = {}

    if unique_smiles:
        sig_mat = np.stack([signature_vector_from_smiles(smi, signature_names) for smi in unique_smiles]).astype(np.float32)
        scores, logits = score_model3(
            model=model,
            spectrum=spectrum,
            phase_id=target_phase_id,
            smiles_list=unique_smiles,
            sig_mat=sig_mat,
            device=device,
            batch_size=args.candidate_batch_size,
            amp=args.amp,
        )
        score_map = {smi: float(v) for smi, v in zip(unique_smiles, scores)}
        logit_map = {smi: float(v) for smi, v in zip(unique_smiles, logits)}

    cand_df["model3_score"] = cand_df["candidate_canonical_smiles"].map(score_map)
    cand_df["model3_logit"] = cand_df["candidate_canonical_smiles"].map(logit_map)

    model3_raw = cand_df.loc[top_indices, "model3_logit"].values if args.use_model3_logit else cand_df.loc[top_indices, "model3_score"].values
    model3_norm_top = normalize_minmax(model3_raw, higher_is_better=True)

    cand_df["model3_norm"] = np.nan
    cand_df.loc[top_indices, "model3_norm"] = model3_norm_top

    # Final score only for top-K.
    cand_df["stage3_score"] = np.nan
    cand_df.loc[top_indices, "stage3_score"] = (
        args.w_sis * cand_df.loc[top_indices, "sis_norm"].values
        + args.w_model3 * cand_df.loc[top_indices, "model3_norm"].values
        + args.w_feature * cand_df.loc[top_indices, f"{args.feature}_norm"].values
    )

    # Rerank: top-K by final score; outside top-K keeps original Stage II order.
    top_part = cand_df.loc[top_indices].copy()
    rest_part = cand_df.drop(index=top_indices).copy()

    top_part = top_part.sort_values(
        by=["stage3_score", "stage2_sis", "stage2_rank"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    rest_part = rest_part.sort_values("stage2_rank").reset_index(drop=True)

    reranked = pd.concat([top_part, rest_part], ignore_index=True)
    reranked["stage3_rank"] = np.arange(1, len(reranked) + 1)

    front_cols = [
        "stage3_rank",
        "stage3_score",
        "sis_norm",
        "model3_norm",
        f"{args.feature}_norm",
        f"{args.feature}_raw",
        "stage2_sis",
        "target_phase",
        "target_phase_id",
        "model3_score",
        "model3_logit",
        "is_true",
        "stage2_rank",
        "candidate_smiles",
        "candidate_canonical_smiles",
        "candidate_inchikey",
        "valid_candidate",
    ]
    other_cols = [c for c in reranked.columns if c not in front_cols]
    reranked = reranked[front_cols + other_cols]

    out_path = folder / args.output_filename
    reranked.to_csv(out_path, index=False)

    true_rows_stage3 = reranked[reranked["is_true"] == True]
    true_found_stage3 = len(true_rows_stage3) > 0
    true_stage3_rank = int(true_rows_stage3["stage3_rank"].iloc[0]) if true_found_stage3 else None
    true_found_in_topk = bool(true_found_all and true_stage2_rank is not None and (args.top_k <= 0 or true_stage2_rank <= args.top_k))

    top1 = reranked.iloc[0] if len(reranked) > 0 else None

    return {
        "folder": str(folder),
        "status": "ok",
        "candidate_file": str(cand_path),
        "candidate_file_name": cand_path.name,
        "top_k": int(args.top_k),
        "feature": args.feature,
        "w_sis": float(args.w_sis),
        "w_model3": float(args.w_model3),
        "w_feature": float(args.w_feature),
        "n_candidates": int(len(cand_df)),
        "n_topk_candidates": int(len(top_part)),
        "n_valid_topk_candidates": int(valid_top_df.shape[0]),
        "true_smiles": true_smiles,
        "true_canonical_smiles": true_can,
        "target_phase": target_phase,
        "target_phase_id": int(target_phase_id),
        "true_inchikey": true_key,
        "true_found_in_candidates": bool(true_found_all),
        "true_found_in_topk": bool(true_found_in_topk),
        "true_stage2_rank": true_stage2_rank,
        "true_stage3_rank": true_stage3_rank,
        "stage2_top1_correct": bool(true_stage2_rank == 1) if true_found_all else False,
        "stage2_top3_correct": bool(true_stage2_rank is not None and true_stage2_rank <= 3) if true_found_all else False,
        "stage2_top5_correct": bool(true_stage2_rank is not None and true_stage2_rank <= 5) if true_found_all else False,
        "stage2_top10_correct": bool(true_stage2_rank is not None and true_stage2_rank <= 10) if true_found_all else False,
        "stage3_top1_correct": bool(true_stage3_rank == 1) if true_found_stage3 else False,
        "stage3_top3_correct": bool(true_stage3_rank is not None and true_stage3_rank <= 3) if true_found_stage3 else False,
        "stage3_top5_correct": bool(true_stage3_rank is not None and true_stage3_rank <= 5) if true_found_stage3 else False,
        "stage3_top10_correct": bool(true_stage3_rank is not None and true_stage3_rank <= 10) if true_found_stage3 else False,
        "top1_smiles": top1["candidate_smiles"] if top1 is not None else None,
        "top1_stage3_score": float(top1["stage3_score"]) if top1 is not None and pd.notna(top1["stage3_score"]) else np.nan,
        "output_csv": str(out_path),
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default="all")
    parser.add_argument("--pattern", default="data*")

    parser.add_argument("--checkpoint", default="model3_stage3_runs/ablation_1_gnn_base/best_model3_gnn.pt")
    parser.add_argument("--signature-vocab", default="model3_stage3_data/signature_vocab.csv")

    parser.add_argument(
        "--candidate-filename",
        default=None,
        help=(
            "Exact candidate filename. If omitted, the script searches "
            "generation_*_elites.csv and uses the largest generation number."
        ),
    )
    parser.add_argument(
        "--candidate-glob",
        default="generation_*_elites.csv",
        help="Glob pattern used when --candidate-filename is omitted.",
    )
    parser.add_argument("--ir-filename", default="IR_data.csv")
    parser.add_argument("--true-smiles-col", default="SMILES")
    parser.add_argument("--phase-col", default="phase", help="Phase column in IR_data.csv.")
    parser.add_argument("--default-phase", default="unknown", help="Fallback phase if IR_data.csv has no phase column.")
    parser.add_argument("--candidate-smiles-col", default="SMILES")
    parser.add_argument("--sis-col", default="SIS")
    parser.add_argument("--ir-row-index", type=int, default=0)

    parser.add_argument("--top-k", type=int, default=200)

    parser.add_argument("--feature", default="heavy_asc", choices=sorted(FEATURES.keys()))
    parser.add_argument("--w-sis", type=float, default=0.15)
    parser.add_argument("--w-model3", type=float, default=0.15)
    parser.add_argument("--w-feature", type=float, default=0.70)

    parser.add_argument("--candidate-batch-size", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-model3-logit", action="store_true")

    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--output-filename", default=None)
    parser.add_argument("--summary-filename", default=None)

    args = parser.parse_args()

    weight_sum = args.w_sis + args.w_model3 + args.w_feature
    if weight_sum <= 0:
        raise ValueError("Sum of weights must be positive.")
    if abs(weight_sum - 1.0) > 1e-6:
        print(f"Warning: weights sum to {weight_sum:.6f}, not 1. They will be used as given.")

    if args.output_filename is None:
        args.output_filename = (
            f"stage3_final_{args.feature}_top{args.top_k}_"
            f"wSIS{safe_name(args.w_sis)}_wM3{safe_name(args.w_model3)}_wF{safe_name(args.w_feature)}.csv"
        )

    if args.summary_filename is None:
        args.summary_filename = (
            f"stage3_final_{args.feature}_top{args.top_k}_"
            f"wSIS{safe_name(args.w_sis)}_wM3{safe_name(args.w_model3)}_wF{safe_name(args.w_feature)}_summary.csv"
        )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        args.device = "cpu"

    device = torch.device(args.device)

    vocab_df = pd.read_csv(args.signature_vocab)
    if "signature_name" not in vocab_df.columns:
        raise ValueError("signature_vocab.csv must contain column: signature_name")
    signature_names = vocab_df["signature_name"].tolist()

    print("Loading Model III...")
    model, meta = load_model(args.checkpoint, device)
    sp_cols = get_sp_cols_from_checkpoint(meta)
    args.phase_vocab = meta.get("phase_vocab", {"unknown": 0})
    if not isinstance(args.phase_vocab, dict) or len(args.phase_vocab) == 0:
        args.phase_vocab = {"unknown": 0}

    print(f"Root: {args.root}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Signature vocab: {args.signature_vocab}")
    print(f"Signatures: {len(signature_names)}")
    print(f"Spectral columns: {sp_cols[0]} ... {sp_cols[-1]} ({len(sp_cols)})")
    print(f"Phase vocab: {args.phase_vocab}")
    print(f"IR phase column: {args.phase_col}; default phase: {args.default_phase}")
    print(f"Feature: {args.feature}")
    print(f"Score = {args.w_sis} * SIS_norm + {args.w_model3} * ModelIII_norm + {args.w_feature} * {args.feature}_norm")
    print(f"Top-K: {args.top_k}")
    print(f"Device: {device}")
    print(f"AMP: {args.amp}")

    root = Path(args.root)
    folders = sorted([p for p in root.glob(args.pattern) if p.is_dir()])
    if args.limit and args.limit > 0:
        folders = folders[:args.limit]

    print(f"Found folders: {len(folders)}")

    results = []

    for folder in tqdm(folders, desc="Stage III reranking"):
        out_path = folder / args.output_filename
        if args.skip_existing and out_path.exists():
            results.append({"folder": str(folder), "status": "skipped_existing", "output_csv": str(out_path)})
            continue

        try:
            res = rerank_one_folder(folder, args, model, sp_cols, signature_names, device)
        except Exception as e:
            res = {"folder": str(folder), "status": "error", "message": repr(e)}
        results.append(res)

    summary_df = pd.DataFrame(results)
    summary_path = root / args.summary_filename
    summary_df.to_csv(summary_path, index=False)

    print(f"\nSaved summary: {summary_path}")

    ok = summary_df[summary_df["status"] == "ok"].copy()
    if len(ok) > 0:
        found_all = ok[ok["true_found_in_candidates"] == True].copy()
        found_topk = ok[ok["true_found_in_topk"] == True].copy()

        print("\n===== Summary on OK folders =====")
        print(f"OK folders: {len(ok)}")
        print(f"True found in candidates: {len(found_all)} / {len(ok)}")
        print(f"True found in top-{args.top_k}: {len(found_topk)} / {len(ok)}")

        # ------------------------------------------------------------
        # Main system-level accuracy:
        # denominator = all OK folders, e.g. 498.
        # If the true molecule is not generated in the candidate pool,
        # the corresponding top-k correctness is already False.
        # ------------------------------------------------------------
        print("\nStage II accuracy among ALL OK folders:")
        for k in [1, 3, 5, 10]:
            col = f"stage2_top{k}_correct"
            if col in ok.columns:
                print(f"  Top-{k}: {ok[col].mean():.4f}")

        print(f"\nStage III accuracy among ALL OK folders:")
        for k in [1, 3, 5, 10]:
            col = f"stage3_top{k}_correct"
            if col in ok.columns:
                print(f"  Top-{k}: {ok[col].mean():.4f}")

        # Save all-OK aggregate metrics as columns in a small extra CSV.
        aggregate_rows = []
        for subset_name, subset_df in [
            ("all_ok", ok),
            ("true_found_in_candidates", found_all),
            (f"true_found_in_top_{args.top_k}", found_topk),
        ]:
            if len(subset_df) == 0:
                continue

            row = {
                "subset": subset_name,
                "n": int(len(subset_df)),
                "denominator": int(len(ok)),
                "true_found_in_candidates": int(subset_df["true_found_in_candidates"].sum()) if "true_found_in_candidates" in subset_df.columns else None,
                "true_found_in_topk": int(subset_df["true_found_in_topk"].sum()) if "true_found_in_topk" in subset_df.columns else None,
            }

            for prefix in ["stage2", "stage3"]:
                for k in [1, 3, 5, 10]:
                    col = f"{prefix}_top{k}_correct"
                    if col in subset_df.columns:
                        row[f"{prefix}_top{k}"] = float(subset_df[col].mean())

            if "true_stage2_rank" in subset_df.columns:
                row["stage2_mean_rank"] = float(pd.to_numeric(subset_df["true_stage2_rank"], errors="coerce").mean())
            if "true_stage3_rank" in subset_df.columns:
                row["stage3_mean_rank"] = float(pd.to_numeric(subset_df["true_stage3_rank"], errors="coerce").mean())

            aggregate_rows.append(row)

        aggregate_df = pd.DataFrame(aggregate_rows)
        aggregate_path = root / args.summary_filename.replace("_summary.csv", "_aggregate_summary.csv")
        aggregate_df.to_csv(aggregate_path, index=False)
        print(f"\nSaved aggregate summary: {aggregate_path}")

        # ------------------------------------------------------------
        # Diagnostic reranking-only accuracy:
        # denominator = folders where the true molecule is available.
        # These are not full-system accuracies, but useful for evaluating
        # ranking power conditional on candidate coverage.
        # ------------------------------------------------------------
        if len(found_all) > 0:
            print("\n[Diagnostic] Stage II accuracy among folders where true is found anywhere:")
            for k in [1, 3, 5, 10]:
                print(f"  Top-{k}: {found_all[f'stage2_top{k}_correct'].mean():.4f}")

        if len(found_topk) > 0:
            print(f"\n[Diagnostic] Stage III accuracy among folders where true is in top-{args.top_k}:")
            for k in [1, 3, 5, 10]:
                print(f"  Top-{k}: {found_topk[f'stage3_top{k}_correct'].mean():.4f}")

            print("\n[Diagnostic] Mean ranks among folders where true is in top-K:")
            print(f"  Stage2 mean rank: {found_topk['true_stage2_rank'].mean():.2f}")
            print(f"  Stage3 mean rank: {found_topk['true_stage3_rank'].mean():.2f}")

            improved = (found_topk["true_stage3_rank"] < found_topk["true_stage2_rank"]).sum()
            worsened = (found_topk["true_stage3_rank"] > found_topk["true_stage2_rank"]).sum()
            unchanged = (found_topk["true_stage3_rank"] == found_topk["true_stage2_rank"]).sum()

            print("\n[Diagnostic] Rank movement among folders where true is in top-K:")
            print(f"  Improved: {improved}")
            print(f"  Worsened: {worsened}")
            print(f"  Unchanged: {unchanged}")

    err = summary_df[summary_df["status"] == "error"]
    if len(err) > 0:
        print("\nErrors occurred in some folders. First few:")
        print(err[["folder", "message"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
