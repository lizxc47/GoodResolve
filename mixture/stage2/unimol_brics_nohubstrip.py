#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
unimol_brics_nohubstrip.py

Utilities for converting a dot-separated mixture SMILES into a hub-connected
pseudo-molecule, applying BRICS fragmentation with reconstruction metadata, and
converting the rebuilt hub pseudo-molecule back to a mixture SMILES.

The workflow is:
  1) mixture SMILES -> hub-connected pseudo-molecule;
  2) BRICS fragmentation with paired dummy-atom tags and bond-type metadata;
  3) fragment display with internal pair tags stripped while preserving hub tags;
  4) fragment reconstruction into a hub-connected pseudo-molecule;
  5) hub-connected pseudo-molecule -> mixture SMILES.
"""


from rdkit import Chem
from rdkit.Chem import BRICS, rdchem
import argparse

MAP_BASE = 9000  # [*:9000] is the central hub; [*:9001+] are arm dummies

INT2BT = {
    1: rdchem.BondType.SINGLE,
    2: rdchem.BondType.DOUBLE,
    3: rdchem.BondType.TRIPLE,
    12: rdchem.BondType.AROMATIC,
}

# ---------- Anchor selection, attachment sites, and splittable single bonds ----------
def _canonical_ranks(m): return list(Chem.CanonicalRankAtoms(m))

def _find_attachable_anchor(mol, comp_atoms):
    ranks = _canonical_ranks(mol)
    cands = [i for i in comp_atoms
              if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
              and mol.GetAtomWithIdx(i).GetNumImplicitHs() >= 1]
    return min(cands, key=lambda i: (ranks[i], i)) if cands else None

def _find_bond_to_split(mol, comp_set):
    tier1, tier2, tier3 = [], [], []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i not in comp_set or j not in comp_set:
            continue
        if b.GetBondType() == rdchem.BondType.SINGLE:
            ai, aj = mol.GetAtomWithIdx(i), mol.GetAtomWithIdx(j)
            aromatic_pair = ai.GetIsAromatic() and aj.GetIsAromatic()
            if not aromatic_pair and not b.IsInRing(): tier1.append(b)
            elif not aromatic_pair and b.IsInRing():   tier2.append(b)
            else:                                      tier3.append(b)
    for group in (tier1, tier2, tier3):
        if group:
            return sorted(group, key=lambda bb: (min(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx()),
                                                  max(bb.GetBeginAtomIdx(), bb.GetEndAtomIdx())))[0]
    return None

# ---------- Mixture SMILES -> hub-connected pseudo-molecule ----------
def ab_to_unimol(smiles: str, map_base: int = MAP_BASE) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Input SMILES parsing failed.")
    comps = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=True)
    if len(comps) <= 1:
        return mol

    rw = Chem.RWMol(mol)
    ranks = _canonical_ranks(mol)
    arm_indices = []

    for k, comp in enumerate(comps, start=1):
        comp_set = set(comp)
        anchor = _find_attachable_anchor(mol, comp)
        if anchor is not None:
            arm = rdchem.Atom(0); arm.SetAtomMapNum(map_base + k)
            arm_idx = rw.AddAtom(arm)
            rw.AddBond(anchor, arm_idx, rdchem.BondType.SINGLE)
            arm_indices.append(arm_idx)
            continue

        b = _find_bond_to_split(mol, comp_set)
        if b is None:
            carbon_cands = [i for i in comp if mol.GetAtomWithIdx(i).GetAtomicNum() == 6]
            target = min(carbon_cands or list(comp), key=lambda i: (ranks[i], i))
            arm = rdchem.Atom(0); arm.SetAtomMapNum(map_base + k)
            arm_idx = rw.AddAtom(arm)
            rw.AddBond(target, arm_idx, rdchem.BondType.SINGLE)
            arm_indices.append(arm_idx)
            continue

        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bo = b.GetBondType()  # SINGLE
        rw.RemoveBond(i, j)
        arm = rdchem.Atom(0); arm.SetAtomMapNum(map_base + k)
        # arm.SetIntProp("orig_bo", BT2INT.get(bo, 1))  # Store the original bond order for split-inserted arms.
        arm_idx = rw.AddAtom(arm)
        rw.AddBond(i, arm_idx, bo)
        rw.AddBond(arm_idx, j, bo)
        arm_indices.append(arm_idx)

    hub = rdchem.Atom(0); hub.SetAtomMapNum(map_base)
    hub_idx = rw.AddAtom(hub)
    for arm_idx in arm_indices:
        rw.AddBond(arm_idx, hub_idx, rdchem.BondType.SINGLE)

    out = rw.GetMol()
    _safe_sanitize(out)
    return out

# ---------- BRICS bond detection and breaking with pair tags and bond-type metadata ----------
def _get_brics_pairs_with_bo(mol):
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
    pairs.sort(key=lambda x: (min(x[0], x[1]), max(x[0], x[1]), x[2], x[3]))
    return pairs

def brics_break_with_metadata(mol):
    """
    Return broken_mol and tag2bo.

    Each broken BRICS bond receives a pair tag in the range 1..N.
    tag2bo[tag] stores the original bond type for that broken pair.
    """
    pairs = _get_brics_pairs_with_bo(mol)
    if not pairs:
        return mol, {}

    broken = BRICS.BreakBRICSBonds(mol)
    tag2bo = {}
    tag_id = 1

    for ai, aj, li, lj, bo in pairs:
        ai_atom = broken.GetAtomWithIdx(ai)
        aj_atom = broken.GetAtomWithIdx(aj)

        def pick_dummy(host_atom, brics_label):
            for nbr in host_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 0 and nbr.GetIsotope() == brics_label and nbr.GetAtomMapNum() == 0:
                    return nbr
            for nbr in host_atom.GetNeighbors():
                if nbr.GetAtomicNum() == 0 and nbr.GetAtomMapNum() == 0:
                    return nbr
            raise RuntimeError("Cannot find a dummy atom at the broken bond end.")

        di = pick_dummy(ai_atom, li)
        dj = pick_dummy(aj_atom, lj)
        di.SetAtomMapNum(tag_id)
        dj.SetAtomMapNum(tag_id)
        tag2bo[tag_id] = bo  # Store the original bond type.
        tag_id += 1

    _safe_sanitize(broken)
    return broken, tag2bo

# ---------- Display fragments: remove pair tags while preserving hub tags and BRICS types ----------
def strip_pair_maps_keep_hub(mol, map_base=MAP_BASE):
    rw = Chem.RWMol(mol)
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            amap = a.GetAtomMapNum()
            if 0 < amap < map_base:
                a.SetAtomMapNum(0)
    out = rw.GetMol()
    _safe_sanitize(out)
    return out

# ---------- Fragments -> rebuilt hub-connected pseudo-molecule ----------
def rebuild_unimol_from_fragments_keep_hub(frag_smiles_with_pairs, tag2bo):
    ms = [Chem.MolFromSmiles(s) for s in frag_smiles_with_pairs if s]
    if not ms or any(m is None for m in ms):
        return ""

    combo = ms[0]
    for m in ms[1:]:
        combo = Chem.CombineMols(combo, m)
    rw = Chem.RWMol(combo)

    by_tag = {}
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0:
            tag = a.GetAtomMapNum()
            if 0 < tag < MAP_BASE:
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
        bo = tag2bo.get(tag, rdchem.BondType.SINGLE)  # Reconnect using the stored bond type.
        if rw.GetBondBetweenAtoms(i, j) is None:
            rw.AddBond(i, j, bo)

    # Delete all internal pair dummy atoms while preserving hub/arm dummies >= MAP_BASE.
    del_idxs = sorted([a.GetIdx() for a in rw.GetAtoms()
                        if a.GetAtomicNum() == 0 and 0 < a.GetAtomMapNum() < MAP_BASE], reverse=True)
    for idx in del_idxs:
        rw.RemoveAtom(idx)

    mol = rw.GetMol()
    _safe_sanitize(mol)
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)

# ---------- Hub-connected pseudo-molecule -> mixture SMILES ----------
def _safe_sanitize(m: Chem.Mol) -> Chem.Mol:
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

def unimol_to_ab(unimol_smiles: str, map_base: int = MAP_BASE) -> str:
    """
    Convert a hub-connected pseudo-molecule SMILES back to a mixture SMILES.

    The central hub [*:9000] and all arm dummy atoms [*:9001+] are removed.
    If an arm is inserted between two heavy atoms, the two heavy atoms are
    reconnected before deletion. The stored intProp 'orig_bo' is used when
    available; otherwise, aromatic endpoints are connected by an aromatic bond
    and other endpoints by a single bond.
    """
    m = Chem.MolFromSmiles(unimol_smiles)
    if m is None:
        return ""
    rw = Chem.RWMol(m)

    # 1) Locate the central hub
    hub_idx = None
    for a in rw.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() == map_base:
            hub_idx = a.GetIdx()
            break
    if hub_idx is None:
        out = rw.GetMol(); _safe_sanitize(out)
        return Chem.MolToSmiles(out, canonical=True, isomericSmiles=True)

    # 2) Collect all arm indices and neighbor information before editing atom indices
    arm_infos = []
    for nbr in rw.GetAtomWithIdx(hub_idx).GetNeighbors():
        if nbr.GetAtomicNum() == 0 and nbr.GetAtomMapNum() > map_base:
            arm_idx = nbr.GetIdx()
            arm_atom = rw.GetAtomWithIdx(arm_idx)
            heavy_neighbors = [n.GetIdx() for n in arm_atom.GetNeighbors() if n.GetAtomicNum() != 0]
            orig_bo = arm_atom.GetIntProp("orig_bo") if arm_atom.HasProp("orig_bo") else None
            arm_infos.append({"idx": arm_idx, "heavy": heavy_neighbors, "orig_bo": orig_bo})

    # 3) For each arm, remove arm bonds and reconnect heavy atoms if needed
    for info in arm_infos:
        arm_idx = info["idx"]
        heavy = info["heavy"]
        # Remove the arm-hub bond
        if rw.GetBondBetweenAtoms(arm_idx, hub_idx):
            rw.RemoveBond(arm_idx, hub_idx)
        # Remove arm-heavy-atom bonds
        for h in heavy:
            if rw.GetBondBetweenAtoms(arm_idx, h):
                rw.RemoveBond(arm_idx, h)
        # Split-inserted arm: reconnect the two heavy atoms
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

    # 4) Delete all arms and the hub in descending atom-index order
    to_delete = sorted([d["idx"] for d in arm_infos] + [hub_idx], reverse=True)
    for idx in to_delete:
        rw.RemoveAtom(idx)

    out = rw.GetMol()
    _safe_sanitize(out)
    return Chem.MolToSmiles(out, canonical=True, isomericSmiles=True)


# ---------- Full round-trip pipeline ----------
def full_pipeline_ab_to_ab(smiles_ab: str):
    # 1) Mixture SMILES -> hub-connected pseudo-molecule
    unimol = ab_to_unimol(smiles_ab)
    unimol_smiles = Chem.MolToSmiles(unimol, canonical=True, isomericSmiles=True)

    # 2) BRICS fragmentation with pair tags and bond-type metadata
    broken, tag2bo = brics_break_with_metadata(unimol)

    # 3) Export fragments: display version without pair tags and reconstruction version with pair tags
    if not tag2bo:
        frag_mols = [broken]
    else:
        frag_mols = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True)
    frags_with_pairs = [Chem.MolToSmiles(m, canonical=True, isomericSmiles=True) for m in frag_mols]
    frags_display = [Chem.MolToSmiles(strip_pair_maps_keep_hub(m), canonical=True, isomericSmiles=True)
                      for m in frag_mols]

    # 4) Fragments with pair tags -> rebuilt hub-connected pseudo-molecule
    unimol_rebuilt = rebuild_unimol_from_fragments_keep_hub(frags_with_pairs, tag2bo)

    # 5) Hub-connected pseudo-molecule -> mixture SMILES
    ab_back = unimol_to_ab(unimol_rebuilt)

    return unimol_smiles, frags_display, unimol_rebuilt, ab_back

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Mixture SMILES -> hub-connected pseudo-molecule -> BRICS fragments -> rebuilt hub-connected pseudo-molecule -> mixture SMILES")
    ap.add_argument("--smiles", required=True, help="Dot-separated mixture SMILES, such as A.B")
    args = ap.parse_args()

    print(">>> Input mixture SMILES:")
    print(args.smiles)

    unimol_smi, frags_display, unimol_rebuilt, ab_back = full_pipeline_ab_to_ab(args.smiles)

    print("\n>>> Hub-connected pseudo-molecule SMILES:")
    print(unimol_smi)

    print("\n>>> BRICS display fragments with hub tags retained and pair tags removed:")
    for i, s in enumerate(sorted(frags_display, key=lambda x: (len(x), x)), 1):
        print(f"{i:02d}. {s}")

    print("\n>>> Rebuilt hub-connected pseudo-molecule SMILES:")
    print(unimol_rebuilt)

    print("\n>>> Recovered mixture SMILES:")
    print(ab_back)

if __name__ == "__main__":
    main()









