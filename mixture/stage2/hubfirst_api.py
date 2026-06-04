# hubfirst_api.py
# -*- coding: utf-8 -*-
"""
hubfirst_api.py

Hub-first assembly utilities for BRICS-labeled fragments.

This module builds hub-containing pseudo-molecules by starting from a hub
fragment and growing it with non-hub fragments. It uses MoleculeBuilder for
fragment connection, screens candidate fragments by BRICS compatibility, and
returns completed products that retain the hub and have no unresolved non-hub
attachment points.
"""


from rdkit import Chem
from rdkit.Chem import rdchem
from typing import List, Optional, Dict, Any, Sequence
import random
import re

# MoleculeBuilder provides the fragment connection rules.
from molecule_builder import MoleculeBuilder  # type: ignore

# ----------------- Regex helpers -----------------
_RE_HUB_MAP   = re.compile(r"\[\*:(\d+)\]")   # [*:9000] hub/arm dummy
_RE_BRICS_LBL = re.compile(r"\[(\d+)\*\]")    # [14*] BRICS typed dummy


# ----------------- Atom-level helpers -----------------
def _mol_brics_types(m: Chem.Mol) -> List[int]:
    """Return BRICS types present in a molecule as isotopes on dummy atoms (* with isotope>0)."""
    types = []
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0:
            iso = a.GetIsotope()
            if iso and iso > 0:
                types.append(int(iso))
    return types


def _mol_has_hub(m: Chem.Mol, map_base: int = 9000) -> bool:
    """Return True if the molecule contains at least one hub/arm dummy [*:k] with k >= map_base."""
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 0 and a.GetAtomMapNum() >= map_base:
            return True
    return False


def _mol_has_nonhub_connectors(m: Chem.Mol, map_base: int = 9000) -> bool:
    """
    Non-hub connectors include:
      - bare [*]         -> atomMap=0 & isotope=0
      - BRICS typed [n*] -> isotope > 0
      - [*:k] with 1 <= k < map_base
    """
    for a in m.GetAtoms():
        if a.GetAtomicNum() != 0:
            continue
        amap = a.GetAtomMapNum()
        iso  = a.GetIsotope()
        if (amap == 0 and iso == 0) or (iso > 0) or (1 <= amap < map_base):
            return True
    return False


# ----------------- String-level helpers -----------------
def smiles_has_star(smi: str) -> bool:
    return "*" in smi


def smiles_has_hub(smi: str, map_base: int = 9000) -> bool:
    return any(int(x) >= map_base for x in _RE_HUB_MAP.findall(str(smi)) if x.isdigit())


def smiles_has_brics_label(smi: str) -> bool:
    return _RE_BRICS_LBL.search(str(smi)) is not None


def _smi_to_mol(smi: str) -> Optional[Chem.Mol]:
    """Robust SMILES to Mol (with gentle sanitization)."""
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is not None:
        return m
    try:
        m = Chem.MolFromSmiles(smi, sanitize=False)
        if m is None:
            return None
        Chem.SanitizeMol(
            m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
        try:
            Chem.Kekulize(m, clearAromaticFlags=True)
        except Exception:
            pass
        return m
    except Exception:
        return None


def _sanitize_gently(m: Chem.Mol) -> Chem.Mol:
    """Try sanitize without strict kekulization; return the same Mol object (best-effort)."""
    try:
        Chem.SanitizeMol(
            m, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
        )
    except Exception:
        pass
    return m


# ----------------- Candidate screening using MoleculeBuilder -----------------
def _get_builder_frag_types(builder: MoleculeBuilder, frag_smiles: str) -> List[int]:
    """
    Retrieve BRICS types for a fragment using the builder's metadata if available.
    Fallback: parse from the fragment molecule (isotopes on dummy atoms).
    """
    tlist: List[int] = []
    try:
        # MoleculeBuilder may precompute frag_types: Dict[str, List[int]].
        tlist = list(builder.frag_types.get(frag_smiles, []))  # type: ignore[attr-defined]
    except Exception:
        tlist = []
    if tlist:
        return [int(t) for t in tlist]

    m = _smi_to_mol(frag_smiles)
    return _mol_brics_types(m) if m is not None else []


def _brics_compatible(builder: MoleculeBuilder, t1: int, t2: int) -> bool:
    """Check if BRICS type t1 can connect to t2 using MoleculeBuilder rules; fall back to symmetric equality."""
    try:
        return bool(builder.brics_connection_allowed(t1, t2))  # type: ignore[attr-defined]
    except Exception:
        # Fallback heuristic if the builder does not expose the rule.
        return t1 == t2


# ----------------- Public API -----------------
def build_hubseed_with_nohub(
    hub_fragment: str,
    nohub_fragments: Sequence[str],
    *,
    map_base: int = 9000,
    max_steps: int = 10,
    per_step_trials: int = 200,
    attempts: int = 50,
    unique_use: bool = False,
    random_seed: Optional[int] = None,
    builder: Optional[MoleculeBuilder] = None,
    all_fragments_for_builder: Optional[Sequence[str]] = None,
    return_intermediate: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Compose a molecule starting from a hub fragment and grow it only with non-hub fragments.
    Fragment connection is performed by MoleculeBuilder.connect_fragments.
    The final molecule MUST still contain the hub and have NO non-hub connectors.

    Parameters
    ----------
    hub_fragment : str
        A fragment SMILES containing a hub/arm dummy [*:k] with k >= map_base.
        Prefer it to also include BRICS typed dummies ([n*]) so that connecting can start.
    nohub_fragments : Sequence[str]
        A list of fragment SMILES WITHOUT hub dummies; preferably containing BRICS labels ([n*]).
    map_base : int, default 9000
        Threshold for hub mapping numbers: [*:k], k >= map_base is considered hub.
    max_steps : int, default 10
        Maximum growth steps.
    per_step_trials : int, default 200
        Maximum trials to connect a candidate in each step.
    attempts : int, default 50
        Maximum global retries with the same input pools.
    unique_use : bool, default False
        If True, each non-hub fragment can be used at most once.
    random_seed : Optional[int]
        If provided, set the RNG seed for reproducibility.
    builder : Optional[MoleculeBuilder]
        An existing MoleculeBuilder instance. If None, a new one is created.
    all_fragments_for_builder : Optional[Sequence[str]]
        If creating a new builder, pass a complete fragment library here. If None,
        we will use [hub_fragment] + list(nohub_fragments).
    return_intermediate : bool, default False
        If True, include intermediate SMILES in the returned dict.

    Returns
    -------
    Optional[Dict[str, Any]]
        On success: {
            "final_smiles": str,
            "used_fragments": List[str],  # order of use, hub first
            "steps": List[str],           # optional, if return_intermediate=True
        }
        On failure: None
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Basic validation
    if not hub_fragment or not isinstance(hub_fragment, str):
        raise ValueError("hub_fragment must be a non-empty SMILES string.")
    if not nohub_fragments:
        raise ValueError("nohub_fragments must be a non-empty list of SMILES strings.")
    if not smiles_has_star(hub_fragment):
        raise ValueError("hub_fragment must contain '*' dummies.")
    if not smiles_has_hub(hub_fragment, map_base=map_base):
        raise ValueError(f"hub_fragment must contain a hub dummy [*:{map_base}+].")

    # Build or reuse MoleculeBuilder.
    if builder is None:
        lib = list(all_fragments_for_builder) if all_fragments_for_builder else [hub_fragment] + list(nohub_fragments)
        builder = MoleculeBuilder(fragment_list=lib, max_steps=max_steps)  # type: ignore

    # Prepare mols
    hub_mol = _smi_to_mol(hub_fragment)
    if hub_mol is None:
        raise ValueError("hub_fragment cannot be parsed by RDKit.")

    # Filter and prepare candidate pools
    pool: List[str] = [s for s in nohub_fragments if s and isinstance(s, str)]
    pool = [s for s in pool if smiles_has_star(s) and not smiles_has_hub(s, map_base)]
    if not pool:
        raise ValueError("nohub_fragments must contain at least one non-hub fragment with '*' dummies.")

    # Prefer BRICS-labeled candidates for easier connections
    pool_labeled = [s for s in pool if smiles_has_brics_label(s)]
    if not pool_labeled:
        # If no labeled fragments are available, still try the full pool; the success rate may be low.
        pool_labeled = pool[:]

    # If the hub has no BRICS labels, connection may not start until a labeled fragment is attached.

    for _ in range(max(1, attempts)):
        current = Chem.Mol(hub_mol)
        _sanitize_gently(current)
        used: List[str] = [hub_fragment]
        steps_smiles: List[str] = [Chem.MolToSmiles(current)]

        # Local pool copy for unique-use mode.
        local_pool = list(pool_labeled)

        step = 0
        while step < max_steps and _mol_has_nonhub_connectors(current, map_base):
            step += 1

            cur_types = _mol_brics_types(current)
            # Screen candidates by BRICS compatibility using MoleculeBuilder rules when available.
            if cur_types:
                candidates = []
                for frag in local_pool:
                    tlist = _get_builder_frag_types(builder, frag)
                    if not tlist:
                        # If types cannot be inferred, keep the fragment as a weak candidate.
                        candidates.append(frag)
                        continue
                    if any(_brics_compatible(builder, t1, t2) for t1 in cur_types for t2 in tlist):
                        candidates.append(frag)
            else:
                # If the current molecule has no BRICS types, try any labeled candidate.
                candidates = local_pool[:]

            if not candidates:
                break  # Dead end for this attempt.

            success = False
            trial = 0
            while trial < max(1, per_step_trials):
                trial += 1
                frag = random.choice(candidates)
                fm = _smi_to_mol(frag)
                if fm is None:
                    # Try another fragment.
                    continue

                # Use MoleculeBuilder to connect fragments.
                try:
                    new_mol = builder.connect_fragments(current, fm)  # type: ignore[attr-defined]
                except Exception:
                    new_mol = None

                if new_mol is None:
                    continue

                current = new_mol
                _sanitize_gently(current)
                used.append(frag)
                if return_intermediate:
                    steps_smiles.append(Chem.MolToSmiles(current))
                success = True

                if unique_use:
                    # Remove once-used fragment from local pool
                    try:
                        local_pool.remove(frag)
                    except ValueError:
                        pass
                break  # Proceed to the next step.

            if not success:
                break  # Give up this attempt.

        # Success condition: retain the hub and remove all non-hub connectors.
        if _mol_has_hub(current, map_base) and not _mol_has_nonhub_connectors(current, map_base):
            final_smi = Chem.MolToSmiles(current, canonical=True, isomericSmiles=True)
            out: Dict[str, Any] = {
                "final_smiles": final_smi,
                "used_fragments": used,
            }
            if return_intermediate:
                out["steps"] = steps_smiles
            return out

    # Failure after all attempts.
    return None


__all__ = [
    "build_hubseed_with_nohub",
    "smiles_has_hub",
    "smiles_has_brics_label",
    "smiles_has_star",
]
