#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hub_brics_canonical.py

Canonical hub-BRICS utilities for multi-component mixture SMILES.

Purpose
-------
This module converts dot-separated mixture SMILES into canonical hub-connected
pseudo-molecules and supports round-trip processing through BRICS fragmentation
and reconstruction.

Core behavior
-------------
1) Split a mixture into components.
2) Canonicalize each component independently.
3) Sort components lexicographically to obtain a deterministic component order.
4) Assign hub and arm dummy-atom maps:
       [*:9000] = central hub
       [*:9001] = arm for canonical component 1
       [*:9002] = arm for canonical component 2
       ...
5) Enumerate valid arm-attachment choices for each component.
6) Build candidate hub pseudo-molecules.
7) Select the candidate with the lexicographically smallest RDKit canonical SMILES.

The module supports mixtures with two or more components, such as A.B, A.B.C,
A.B.C.D, and higher-component inputs.

Dependencies
------------
rdkit
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import prod
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import BRICS, rdchem


# =========================
# Constants
# =========================

# [*:9000] is the central hub; [*:9001], [*:9002], ... are component arms.
# BRICS-pair dummy atom map numbers are kept in 1..8999.
MAP_BASE = 9000
PAIR_MIN, PAIR_MAX = 1, 8999

BT2INT = {
    rdchem.BondType.SINGLE: 1,
    rdchem.BondType.DOUBLE: 2,
    rdchem.BondType.TRIPLE: 3,
    rdchem.BondType.AROMATIC: 12,
}
INT2BT = {v: k for k, v in BT2INT.items()}


# =========================
# General utilities
# =========================

def _safe_sanitize(m: Chem.Mol) -> Chem.Mol:
    """Best-effort sanitization, with a fallback that skips strict kekulization."""
    try:
        Chem.SanitizeMol(m)
        return m
    except Exception:
        Chem.SanitizeMol(
            m,
            sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE,
        )
        try:
            Chem.Kekulize(m, clearAromaticFlags=True)
        except Exception:
            pass
        return m


def _mol_from_smiles_or_raise(smiles: str, *, what: str = "SMILES") -> Chem.Mol:
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError(f"{what} parse failed: {smiles!r}")
    return m


def _canonical_smiles(m: Chem.Mol) -> str:
    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=True)


def _canonical_ranks(m: Chem.Mol, *, break_ties: bool = True) -> List[int]:
    return list(Chem.CanonicalRankAtoms(m, breakTies=break_ties))


# =========================
# Canonical mixture handling
# =========================

def canonicalize_component_smiles(component_smiles: str) -> str:
    """Return the RDKit canonical isomeric SMILES of a single component."""
    m = _mol_from_smiles_or_raise(component_smiles, what="component SMILES")
    return _canonical_smiles(m)


def canonicalize_mixture_components(smiles: str) -> List[str]:
    """
    Split a mixture SMILES into components, canonicalize each component independently,
    and sort components lexicographically.

    This makes A.B.C, C.B.A, and equivalent component SMILES use the same component order.
    Duplicate components are retained, e.g. A.A.B -> [A, A, B].
    """
    mol = _mol_from_smiles_or_raise(smiles, what="mixture SMILES")
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    component_smis = [_canonical_smiles(f) for f in frags]
    component_smis.sort()
    return component_smis


def canonicalize_mixture_smiles(smiles: str) -> str:
    """Return a dot-separated canonical mixture SMILES with sorted components."""
    return ".".join(canonicalize_mixture_components(smiles))


# =========================
# Canonical arm-attachment enumeration
# =========================

@dataclass(frozen=True)
class AttachmentChoice:
    """A single arm-attachment choice for one component."""
    mode: str  # "direct", "split", or "fallback"
    anchor_idx: Optional[int] = None
    bond_i: Optional[int] = None
    bond_j: Optional[int] = None
    bond_type: rdchem.BondType = rdchem.BondType.SINGLE
    tier: int = 0


@dataclass(frozen=True)
class ComponentVariant:
    """A component molecule after one arm has been attached."""
    mol: Chem.Mol
    smiles: str
    choice: AttachmentChoice
    arm_map: int


def _direct_attachable_atoms(mol: Chem.Mol) -> List[int]:
    """
    Direct arm-attachment candidates.

    Definition: heavy atoms with at least one implicit hydrogen.
    Attaching an arm is interpreted as replacing one implicit H by a dummy arm.
    """
    ranks = _canonical_ranks(mol, break_ties=True)
    cands = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() > 1 and a.GetNumImplicitHs() >= 1
    ]
    return sorted(cands, key=lambda i: (ranks[i], i))


def _split_bond_candidates(mol: Chem.Mol) -> List[Tuple[int, int, rdchem.BondType, int]]:
    """
    Enumerate single bonds that can be split by inserting an arm dummy.

    Tiers used for candidate ordering:
      tier 1: non-aromatic-pair and non-ring single bonds
      tier 2: non-aromatic-pair and ring single bonds
      tier 3: other single bonds

    For canonical uniqueness, this function returns all candidate single bonds in a stable order.
    """
    ranks = _canonical_ranks(mol, break_ties=True)
    out: List[Tuple[int, int, rdchem.BondType, int]] = []

    for b in mol.GetBonds():
        if b.GetBondType() != rdchem.BondType.SINGLE:
            continue
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
        aromatic_pair = ai.GetIsAromatic() and aj.GetIsAromatic()
        if not aromatic_pair and not b.IsInRing():
            tier = 1
        elif not aromatic_pair and b.IsInRing():
            tier = 2
        else:
            tier = 3
        out.append((i, j, b.GetBondType(), tier))

    return sorted(
        out,
        key=lambda x: (
            x[3],
            min(ranks[x[0]], ranks[x[1]]),
            max(ranks[x[0]], ranks[x[1]]),
            min(x[0], x[1]),
            max(x[0], x[1]),
        ),
    )


def _fallback_anchor_atoms(mol: Chem.Mol) -> List[int]:
    """
    Last-resort anchors when no direct H-bearing atom and no splittable single bond exists.
    The returned atoms are ordered canonically.
    """
    ranks = _canonical_ranks(mol, break_ties=True)
    carbons = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 6]
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    cands = carbons or heavy
    return sorted(cands, key=lambda i: (ranks[i], i))


def _attach_arm_direct(mol: Chem.Mol, anchor_idx: int, arm_map: int) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    arm = rdchem.Atom(0)
    arm.SetAtomMapNum(arm_map)
    arm_idx = rw.AddAtom(arm)
    rw.AddBond(anchor_idx, arm_idx, rdchem.BondType.SINGLE)
    out = rw.GetMol()
    _safe_sanitize(out)
    return out


def _attach_arm_by_splitting_bond(
    mol: Chem.Mol,
    i: int,
    j: int,
    bond_type: rdchem.BondType,
    arm_map: int,
) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    if rw.GetBondBetweenAtoms(i, j) is None:
        raise RuntimeError(f"Cannot split missing bond: {i}-{j}")
    rw.RemoveBond(i, j)

    arm = rdchem.Atom(0)
    arm.SetAtomMapNum(arm_map)
    arm.SetIntProp("orig_bo", BT2INT.get(bond_type, 1))
    arm_idx = rw.AddAtom(arm)

    rw.AddBond(i, arm_idx, bond_type)
    rw.AddBond(arm_idx, j, bond_type)
    out = rw.GetMol()
    _safe_sanitize(out)
    return out


def enumerate_component_arm_variants(component_smiles: str, arm_map: int) -> List[ComponentVariant]:
    """
    Enumerate all arm-attached variants of one canonicalized component.

    Rule:
      1) If direct attachable atoms exist, enumerate all direct attachments.
      2) Otherwise, enumerate all splittable single bonds.
      3) Otherwise, use a last-resort direct attachment to a canonical heavy atom.

    The returned variants are sorted by their own canonical SMILES for stable behavior.
    """
    component_smiles = canonicalize_component_smiles(component_smiles)
    mol = _mol_from_smiles_or_raise(component_smiles, what="canonical component SMILES")

    variants: List[ComponentVariant] = []

    direct = _direct_attachable_atoms(mol)
    if direct:
        for idx in direct:
            choice = AttachmentChoice(mode="direct", anchor_idx=idx)
            try:
                vmol = _attach_arm_direct(mol, idx, arm_map)
                variants.append(ComponentVariant(vmol, _canonical_smiles(vmol), choice, arm_map))
            except Exception:
                # Skip chemically impossible candidates if RDKit refuses sanitization.
                continue
    else:
        split_bonds = _split_bond_candidates(mol)
        if split_bonds:
            for i, j, bo, tier in split_bonds:
                choice = AttachmentChoice(mode="split", bond_i=i, bond_j=j, bond_type=bo, tier=tier)
                try:
                    vmol = _attach_arm_by_splitting_bond(mol, i, j, bo, arm_map)
                    variants.append(ComponentVariant(vmol, _canonical_smiles(vmol), choice, arm_map))
                except Exception:
                    continue
        else:
            # Very rare fallback: monoatomic or unusual components.
            for idx in _fallback_anchor_atoms(mol):
                choice = AttachmentChoice(mode="fallback", anchor_idx=idx)
                try:
                    vmol = _attach_arm_direct(mol, idx, arm_map)
                    variants.append(ComponentVariant(vmol, _canonical_smiles(vmol), choice, arm_map))
                except Exception:
                    continue
                break

    if not variants:
        raise ValueError(f"No valid arm-attachment variant for component: {component_smiles}")

    # Deduplicate by variant SMILES. If symmetry gives the same pseudo-SMILES, one is enough
    # because the final representation is identical.
    dedup: Dict[str, ComponentVariant] = {}
    for v in variants:
        dedup.setdefault(v.smiles, v)

    return [dedup[k] for k in sorted(dedup)]


def _combine_component_variants_with_hub(
    variants: Sequence[ComponentVariant],
    *,
    map_base: int = MAP_BASE,
) -> Chem.Mol:
    """Combine arm-attached component variants and add one central hub [*:map_base]."""
    if not variants:
        raise ValueError("No component variants were provided.")

    combo = Chem.Mol(variants[0].mol)
    for v in variants[1:]:
        combo = Chem.CombineMols(combo, v.mol)

    rw = Chem.RWMol(combo)
    hub = rdchem.Atom(0)
    hub.SetAtomMapNum(map_base)
    hub_idx = rw.AddAtom(hub)

    for v in variants:
        arm_idxs = [
            a.GetIdx()
            for a in rw.GetAtoms()
            if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == v.arm_map
        ]
        if len(arm_idxs) != 1:
            raise RuntimeError(f"Expected exactly one arm dummy [*:{v.arm_map}], found {len(arm_idxs)}")
        arm_idx = arm_idxs[0]
        if rw.GetBondBetweenAtoms(arm_idx, hub_idx) is None:
            rw.AddBond(arm_idx, hub_idx, rdchem.BondType.SINGLE)

    out = rw.GetMol()
    _safe_sanitize(out)
    return out


@dataclass
class CanonicalHubResult:
    """Result of converting a mixture into a canonical hub-connected pseudo-molecule."""
    canonical_mixture_smiles: str
    canonical_hub_smiles: str
    component_smiles: List[str]
    selected_component_variants: List[str]
    selected_attachment_choices: List[AttachmentChoice]
    total_candidate_combinations: int


def mixture_to_canonical_hub_mol(
    smiles: str,
    *,
    map_base: int = MAP_BASE,
    max_combinations: Optional[int] = None,
    return_info: bool = False,
) -> Chem.Mol | Tuple[Chem.Mol, CanonicalHubResult]:
    """
    Convert a mixture SMILES into a unique canonical hub-connected pseudo-molecule.

    Supports A.B, A.B.C, A.B.C.D, ... without hard-coding the number of components.

    If max_combinations is not None and the exact enumeration exceeds it, this function
    raises ValueError. This avoids silently losing the uniqueness guarantee.
    """
    component_smis = canonicalize_mixture_components(smiles)
    canonical_mix = ".".join(component_smis)

    # For a single molecule, return the canonical molecule itself.
    if len(component_smis) <= 1:
        mol = _mol_from_smiles_or_raise(canonical_mix, what="canonical single-component SMILES")
        _safe_sanitize(mol)
        info = CanonicalHubResult(
            canonical_mixture_smiles=canonical_mix,
            canonical_hub_smiles=_canonical_smiles(mol),
            component_smiles=component_smis,
            selected_component_variants=component_smis,
            selected_attachment_choices=[],
            total_candidate_combinations=1,
        )
        return (mol, info) if return_info else mol

    per_component_variants: List[List[ComponentVariant]] = []
    for k, comp_smi in enumerate(component_smis, start=1):
        arm_map = map_base + k
        variants = enumerate_component_arm_variants(comp_smi, arm_map=arm_map)
        per_component_variants.append(variants)

    total = prod(len(vs) for vs in per_component_variants)
    if max_combinations is not None and total > max_combinations:
        raise ValueError(
            f"Exact canonical enumeration would require {total} combinations, "
            f"which exceeds max_combinations={max_combinations}. "
            "Increase max_combinations or leave it as None to preserve uniqueness."
        )

    best_smi: Optional[str] = None
    best_mol: Optional[Chem.Mol] = None
    best_combo: Optional[Tuple[ComponentVariant, ...]] = None

    for combo in product(*per_component_variants):
        try:
            cand = _combine_component_variants_with_hub(combo, map_base=map_base)
            cand_smi = _canonical_smiles(cand)
        except Exception:
            continue
        if best_smi is None or cand_smi < best_smi:
            best_smi = cand_smi
            best_mol = cand
            best_combo = combo

    if best_mol is None or best_smi is None or best_combo is None:
        raise RuntimeError("No valid hub pseudo-molecule could be generated.")

    info = CanonicalHubResult(
        canonical_mixture_smiles=canonical_mix,
        canonical_hub_smiles=best_smi,
        component_smiles=component_smis,
        selected_component_variants=[v.smiles for v in best_combo],
        selected_attachment_choices=[v.choice for v in best_combo],
        total_candidate_combinations=total,
    )
    return (best_mol, info) if return_info else best_mol


# Alias for converting a mixture SMILES into a hub-connected pseudo-molecule.
def ab_to_unimol(smiles: str, map_base: int = MAP_BASE) -> Chem.Mol:
    return mixture_to_canonical_hub_mol(smiles, map_base=map_base, return_info=False)  # type: ignore[return-value]


# =========================
# BRICS break with metadata
# =========================

def _get_brics_pairs_with_bo(mol: Chem.Mol) -> List[Tuple[int, int, int, int, rdchem.BondType]]:
    pairs = []
    for rec in BRICS.FindBRICSBonds(mol):
        try:
            (ai, aj), (li, lj) = rec
        except Exception:
            bond_or_idx, (li, lj) = rec
            b = mol.GetBondWithIdx(bond_or_idx) if isinstance(bond_or_idx, int) else bond_or_idx
            ai, aj = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        b = mol.GetBondBetweenAtoms(ai, aj)
        bo = b.GetBondType() if b is not None else rdchem.BondType.SINGLE
        pairs.append((ai, aj, int(li), int(lj), bo))

    ranks = _canonical_ranks(mol, break_ties=True)
    pairs.sort(
        key=lambda x: (
            min(ranks[x[0]], ranks[x[1]]),
            max(ranks[x[0]], ranks[x[1]]),
            min(x[0], x[1]),
            max(x[0], x[1]),
            x[2],
            x[3],
        )
    )
    return pairs


def brics_break_with_metadata(mol: Chem.Mol) -> Tuple[Chem.Mol, Dict[int, rdchem.BondType]]:
    """
    Break BRICS bonds and assign pair atom-map numbers to each broken dummy pair.

    Returns
    -------
    broken_mol : Chem.Mol
        The broken molecule. BRICS dummy atoms keep their isotope BRICS type, and
        each broken pair gets the same AtomMapNum in 1..8999.
    tag2bo : Dict[int, rdchem.BondType]
        Pair tag -> original bond type, used for reconstruction.
    """
    pairs = _get_brics_pairs_with_bo(mol)
    if not pairs:
        return mol, {}

    broken = BRICS.BreakBRICSBonds(mol)
    tag2bo: Dict[int, rdchem.BondType] = {}
    tag_id = 1

    for ai, aj, li, lj, bo in pairs:
        ai_atom = broken.GetAtomWithIdx(ai)
        aj_atom = broken.GetAtomWithIdx(aj)

        def pick_dummy(host_atom: Chem.Atom, brics_label: int) -> Chem.Atom:
            for nbr in host_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 0 and nbr.GetIsotope() == brics_label and nbr.GetAtomMapNum() == 0:
                    return nbr
            for nbr in host_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 0 and nbr.GetAtomMapNum() == 0:
                    return nbr
            raise RuntimeError("Cannot find BRICS dummy at broken bond end.")

        di = pick_dummy(ai_atom, li)
        dj = pick_dummy(aj_atom, lj)
        di.SetAtomMapNum(tag_id)
        dj.SetAtomMapNum(tag_id)
        tag2bo[tag_id] = bo
        tag_id += 1

    _safe_sanitize(broken)
    return broken, tag2bo


# =========================
# Fragment export
# =========================

def _strip_pair_maps_keep_hub(mol: Chem.Mol, map_base: int = MAP_BASE) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            amap = a.GetAtomMapNum()
            if PAIR_MIN <= amap < map_base:
                a.SetAtomMapNum(0)
    out = rw.GetMol()
    _safe_sanitize(out)
    return out


def fragments_for_display(broken: Chem.Mol, map_base: int = MAP_BASE) -> List[str]:
    """
    Return display fragment SMILES.

    Pair atom-map numbers are stripped because they are internal reconstruction metadata.
    Hub/arm maps >= map_base are retained.
    """
    if broken is None:
        return []

    has_pairs = any(
        a.GetAtomicNum() == 0 and (PAIR_MIN <= a.GetAtomMapNum() < map_base)
        for a in broken.GetAtoms()
    )
    frag_mols = [broken] if not has_pairs else Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)

    out = []
    for fm in frag_mols:
        fm2 = _strip_pair_maps_keep_hub(fm, map_base)
        out.append(_canonical_smiles(fm2))
    return sorted(out, key=lambda s: (len(s), s))


# =========================
# Rebuild fragments -> hub pseudo-molecule
# =========================

def rebuild_unimol_from_fragments_keep_hub(
    frag_smiles_with_pairs: Sequence[str],
    tag2bo: Dict[int, rdchem.BondType],
    map_base: int = MAP_BASE,
) -> str:
    """
    Rebuild a hub pseudo-molecule from fragments that still contain pair atom-map numbers.

    Pair dummy atoms are deleted after the original bonds are restored.
    Hub/arm dummy atoms with maps >= map_base are retained.
    """
    ms = [Chem.MolFromSmiles(s) for s in frag_smiles_with_pairs if s]
    if not ms or any(m is None for m in ms):
        return ""

    combo = ms[0]
    for m in ms[1:]:
        combo = Chem.CombineMols(combo, m)
    rw = Chem.RWMol(combo)

    by_tag: Dict[int, List[int]] = {}
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            tag = a.GetAtomMapNum()
            if PAIR_MIN <= tag < map_base:
                by_tag.setdefault(tag, []).append(a.GetIdx())

    for tag, didxs in by_tag.items():
        if len(didxs) != 2:
            continue
        a1_idx, a2_idx = didxs
        a1 = rw.GetAtomWithIdx(a1_idx)
        a2 = rw.GetAtomWithIdx(a2_idx)
        n1 = [n.GetIdx() for n in a1.GetNeighbors() if n.GetAtomicNum() != 0]
        n2 = [n.GetIdx() for n in a2.GetNeighbors() if n.GetAtomicNum() != 0]
        if not n1 or not n2:
            continue
        i, j = n1[0], n2[0]
        bo = tag2bo.get(tag, rdchem.BondType.SINGLE)
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, bo)

    # Delete internal pair dummies only; keep hub/arms >= map_base.
    del_idxs = sorted(
        [
            a.GetIdx()
            for a in rw.GetAtoms()
            if a.GetAtomicNum() == 0 and (PAIR_MIN <= a.GetAtomMapNum() < map_base)
        ],
        reverse=True,
    )
    for idx in del_idxs:
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    _safe_sanitize(mol)
    return _canonical_smiles(mol)


# =========================
# Hub pseudo-molecule -> mixture
# =========================

def unimol_to_mixture(unimol_smiles: str, map_base: int = MAP_BASE) -> str:
    """
    Convert a hub pseudo-molecule back to a dot-separated mixture SMILES.

    This removes the central hub and all component arms. If an arm was inserted into
    a single bond, the two heavy atoms are healed by orig_bo when available, otherwise
    by a single/aromatic heuristic.
    """
    m = Chem.MolFromSmiles(unimol_smiles)
    if m is None:
        return ""
    rw = Chem.RWMol(m)

    hub_idx = None
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == map_base:
            hub_idx = a.GetIdx()
            break

    if hub_idx is None:
        out = rw.GetMol()
        _safe_sanitize(out)
        return canonicalize_mixture_smiles(_canonical_smiles(out))

    # Collect arm data before editing atom indices.
    arm_infos = []
    for nbr in rw.GetAtomWithIdx(hub_idx).GetNeighbors():
        if nbr.GetAtomicNum() == 0 and nbr.GetAtomMapNum() > map_base:
            arm_idx = nbr.GetIdx()
            arm_atom = rw.GetAtomWithIdx(arm_idx)
            heavy_neighbors = [n.GetIdx() for n in arm_atom.GetNeighbors() if n.GetAtomicNum() != 0]
            orig_bo = arm_atom.GetIntProp("orig_bo") if arm_atom.HasProp("orig_bo") else None
            arm_infos.append({"idx": arm_idx, "heavy": heavy_neighbors, "orig_bo": orig_bo})

    for info in arm_infos:
        arm_idx = info["idx"]
        heavy = info["heavy"]

        if rw.GetBondBetweenAtoms(arm_idx, hub_idx):
            rw.RemoveBond(arm_idx, hub_idx)
        for h in heavy:
            if rw.GetBondBetweenAtoms(arm_idx, h):
                rw.RemoveBond(arm_idx, h)

        # Split-inserted arm: heal the original heavy-heavy bond.
        if len(heavy) == 2:
            i, j = heavy
            if rw.GetBondBetweenAtoms(i, j) is None:
                if info["orig_bo"] is not None:
                    bo = INT2BT.get(int(info["orig_bo"]), rdchem.BondType.SINGLE)
                else:
                    ai = rw.GetAtomWithIdx(i).GetIsAromatic()
                    aj = rw.GetAtomWithIdx(j).GetIsAromatic()
                    bo = rdchem.BondType.AROMATIC if (ai and aj) else rdchem.BondType.SINGLE
                rw.AddBond(i, j, bo)

    to_delete = sorted([d["idx"] for d in arm_infos] + [hub_idx], reverse=True)
    for idx in to_delete:
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    _safe_sanitize(out)
    return canonicalize_mixture_smiles(_canonical_smiles(out))


# Alias for converting a hub pseudo-molecule back to a mixture SMILES.
def unimol_to_ab(unimol_smiles: str, map_base: int = MAP_BASE) -> str:
    return unimol_to_mixture(unimol_smiles, map_base=map_base)


# =========================
# One-shot pipeline
# =========================

@dataclass
class RoundTripResult:
    """Intermediate and final outputs of the canonical hub-BRICS round trip."""
    canonical_mixture_smiles: str
    unimol_smiles: str
    frags_display: List[str]
    unimol_rebuilt: str
    mixture_back: str
    total_candidate_combinations: int
    selected_component_variants: List[str]
    selected_attachment_choices: List[AttachmentChoice]

    @property
    def ab_back(self) -> str:
        """Alias for the mixture-back result."""
        return self.mixture_back


def full_pipeline_mixture_to_mixture(
    mixture_smiles: str,
    *,
    map_base: int = MAP_BASE,
    max_combinations: Optional[int] = None,
) -> RoundTripResult:
    """
    Full pipeline:
        mixture A.B.C...
          -> canonical hub pseudo-molecule
          -> BRICS break with metadata
          -> display fragments
          -> rebuild hub pseudo-molecule
          -> mixture A.B.C...
    """
    unimol, info = mixture_to_canonical_hub_mol(
        mixture_smiles,
        map_base=map_base,
        max_combinations=max_combinations,
        return_info=True,
    )
    assert isinstance(info, CanonicalHubResult)
    unimol_smi = _canonical_smiles(unimol)

    broken, tag2bo = brics_break_with_metadata(unimol)
    frags_display = fragments_for_display(broken, map_base)

    frag_mols = [broken] if not tag2bo else Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    frags_with_pairs = [_canonical_smiles(m) for m in frag_mols]
    unimol_rebuilt = rebuild_unimol_from_fragments_keep_hub(frags_with_pairs, tag2bo, map_base)

    mixture_back = unimol_to_mixture(unimol_rebuilt, map_base)

    return RoundTripResult(
        canonical_mixture_smiles=info.canonical_mixture_smiles,
        unimol_smiles=unimol_smi,
        frags_display=frags_display,
        unimol_rebuilt=unimol_rebuilt,
        mixture_back=mixture_back,
        total_candidate_combinations=info.total_candidate_combinations,
        selected_component_variants=info.selected_component_variants,
        selected_attachment_choices=info.selected_attachment_choices,
    )


# Alias for converting a hub pseudo-molecule back to a mixture SMILES. Now also supports ternary and higher mixtures.
def full_pipeline_ab_to_ab(smiles_ab: str) -> RoundTripResult:
    return full_pipeline_mixture_to_mixture(smiles_ab)


# =========================
# Small CLI/demo
# =========================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Canonical hub-BRICS conversion for mixture SMILES.")
    parser.add_argument("smiles", help="Mixture SMILES, e.g. 'O=C(Cl)OCC=C.OCC1(O)CCC1' or 'A.B.C'.")
    parser.add_argument("--max-combinations", type=int, default=None, help="Optional safety cap for exact enumeration.")
    args = parser.parse_args()

    result = full_pipeline_mixture_to_mixture(args.smiles, max_combinations=args.max_combinations)
    print("Canonical mixture SMILES:", result.canonical_mixture_smiles)
    print("Canonical hub pseudo-SMILES:", result.unimol_smiles)
    print("Total candidate combinations:", result.total_candidate_combinations)
    print("Selected component variants:")
    for s in result.selected_component_variants:
        print("  ", s)
    print("Display fragments:")
    for f in result.frags_display:
        print("  ", f)
    print("Rebuilt hub pseudo-SMILES:", result.unimol_rebuilt)
    print("Mixture back:", result.mixture_back)
