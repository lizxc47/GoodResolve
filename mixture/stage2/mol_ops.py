# -*- coding: utf-8 -*-
"""
mol_ops.py

Molecular fragmentation, mutation, validity checking, fragment selection, and
BRICS-based crossover utilities for the GA workflow.

Main functions:
- mol_to_smiles(mol)
- fragment_with_brics(mol)
- conservative_mutation(smiles, atom_choices)
- is_valid_smiles(smiles, max_components=None, treat_single_as_invalid=False)
- select_based_on_fitness_of_fragments(...)
- partial_crossover(parent1, parent2, fragment_fitness_df, time_limit=1, ...)

This module focuses on structure-generation operations. Spectrum prediction and
SIS scoring are handled by the caller through the GA fitness function.
"""

import time
import random
import numpy as np
import pandas as pd

from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import BRICS
from brics_build_api import time_limited_build
import re

RDLogger.DisableLog('rdApp.*')

k_total      = 40
k_hub        = 2
k_nohub      = k_total - k_hub
k_nohub_tour = k_nohub // 2
k_nohub_rand = k_nohub - k_nohub_tour
k_hub_tour   = k_hub // 2
k_hub_rand   = k_hub - k_hub_tour

# -----------------------------
# Utility functions
# -----------------------------
def mol_to_smiles(mol):
    """Convert an RDKit Mol object to canonical SMILES."""
    return Chem.MolToSmiles(mol)

def fragment_with_brics(mol):
    """Perform BRICS decomposition and return a list of fragment SMILES."""
    return list(BRICS.BRICSDecompose(mol))

def conservative_mutation(smiles, atom_choices):
    """
    Conservative mutation: randomly remove one atom, add a new atom, and
    try to reconnect it to neighboring atoms with single bonds.

    This operation can generate disconnected or invalid structures, which are
    handled by downstream validity checks.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    atom_idx = random.choice(range(mol.GetNumAtoms()))
    editable_mol = Chem.RWMol(mol)
    editable_mol.RemoveAtom(atom_idx)

    new_atom_symbol = random.choice(atom_choices)
    new_atom = Chem.Atom(Chem.GetPeriodicTable().GetAtomicNumber(new_atom_symbol))
    new_atom_idx = editable_mol.AddAtom(new_atom)

    neighbors = [bond.GetOtherAtomIdx(atom_idx) for bond in editable_mol.GetBonds()
                 if bond.GetBeginAtomIdx() == atom_idx or bond.GetEndAtomIdx() == atom_idx]
    for neighbor in neighbors:
        editable_mol.AddBond(neighbor, new_atom_idx, Chem.BondType.SINGLE)

    new_smiles = Chem.MolToSmiles(editable_mol, canonical=True)
    return new_smiles if new_smiles else None

def is_valid_smiles(
    smiles: str,
    max_components: int | None = None,
    treat_single_as_invalid: bool = False,
):
    """
    Validate a SMILES string with optional component-count constraints:
      - max_components: maximum allowed number of dot-separated components.
        None means no upper limit.
          * 1 allows only single-component SMILES.
          * 2 allows up to two components.
          * Larger values follow the same rule.
      - treat_single_as_invalid: if True, single-component SMILES are rejected.
      - the SMILES must be parseable by RDKit.
      - each component must have length >= 3.
    """
    if not isinstance(smiles, str):
        return False

    s = smiles.strip()
    if not s:
        return False

    # Split into components
    components = s.split('.')
    n_comp = len(components)

    # Apply the single-component rejection option first
    if treat_single_as_invalid and n_comp == 1:
        return False

    # Check the maximum number of components
    if (max_components is not None) and (n_comp > max_components):
        return False

    # RDKit parsing; RDKit can parse dot-separated mixtures as disconnected molecules.
    if Chem.MolFromSmiles(s) is None:
        return False

    # Component-length heuristic
    for comp in components:
        if len(comp) < 3:
            return False

    return True

import re
import numpy as np
from typing import List, Dict

_HUB_MAPNUM_RE = re.compile(r"\[\*:(\d+)\]")

def _has_hub(smi: str, map_base: int) -> bool:
    """Return whether the SMILES contains a hub or arm dummy [*:map_base+]."""
    for m in _HUB_MAPNUM_RE.finditer(smi):
        try:
            if int(m.group(1)) >= map_base:
                return True
        except ValueError:
            continue
    return False

def _tournament_pick(pool: List[str], fitness_scores: Dict[str, float], k: int, tournament_size: int) -> List[str]:
    """Run k tournament selections from a pool and return the highest-scoring fragment from each tournament."""
    if not pool or k <= 0:
        return []
    picks = []
    for _ in range(k):
        size = min(tournament_size, len(pool))
        contestants = np.random.choice(pool, size=size, replace=False)
        best = max(contestants, key=lambda frag: fitness_scores.get(frag, 0.0))
        picks.append(best)
    return picks

def _random_pick(pool: List[str], k: int) -> List[str]:
    """Randomly sample k fragments from a pool with replacement."""
    if not pool or k <= 0:
        return []
    return list(np.random.choice(pool, size=k, replace=True))

def select_based_on_fitness_of_fragments(
    smiles_list: List[str],
    fitness_scores: Dict[str, float],
    *,
    # Sampling quota for fragments without a hub
    k_nohub_tournament: int = 0,
    k_nohub_random: int = 0,
    # Sampling quota for fragments with a hub
    k_hub_tournament: int = 0,
    k_hub_random: int = 0,
    tournament_size: int = 300,
    map_base: int = 9000,
    deduplicate: bool = False,
) -> List[str]:
    """
    Split the fragment pool into non-hub and hub-containing fragments, sample
    each pool using tournament selection and random sampling, and merge the
    selected fragments.

    Parameters
    ----------
    smiles_list : list of fragment SMILES.
    fitness_scores : dict mapping SMILES to scores used by tournament selection.
    k_nohub_tournament, k_nohub_random : quotas sampled from the non-hub pool.
    k_hub_tournament, k_hub_random : quotas sampled from the hub-containing pool.
    tournament_size : maximum number of tournament contestants.
    map_base : hub/arm atom-map threshold; default 9000 matches [*:9000], [*:9001], ...
    deduplicate : if True, deduplicate the final result while preserving order.

    Returns
    -------
    List[str] : selected fragment list. The length may be smaller than the
    requested quota if one of the pools is empty.
    """
    if not smiles_list:
        return []

    # Split into two pools
    pool_hub   = [s for s in smiles_list if _has_hub(s, map_base)]
    pool_nohub = [s for s in smiles_list if not _has_hub(s, map_base)]

    # Non-hub pool: tournament sampling plus random sampling
    out_nohub = []
    out_nohub += _tournament_pick(pool_nohub, fitness_scores, k_nohub_tournament, tournament_size)
    out_nohub += _random_pick(pool_nohub, k_nohub_random)

    # Hub-containing pool: tournament sampling plus random sampling
    out_hub = []
    out_hub += _tournament_pick(pool_hub, fitness_scores, k_hub_tournament, tournament_size)
    out_hub += _random_pick(pool_hub, k_hub_random)

    out = out_nohub + out_hub

    if deduplicate:
        seen = set()
        out = [x for x in out if not (x in seen or seen.add(x))]

    return out

# -----------------------------
# BRICS crossover logic
# ----------------------------
_BRICS_LABEL = re.compile(r"\[\d+\*\]")  # BRICS labels such as [14*]
import re
_MAPPED_DUMMY = re.compile(r"\[\*:(\d+)\]")

def _has_hub(s: str, map_base: int = 9000) -> bool:
    """Return whether the SMILES contains a hub or arm dummy [*:map_base+]."""
    return any(int(x) >= map_base for x in _MAPPED_DUMMY.findall(str(s)) if x.isdigit())


def _is_complete_molecule(s: str) -> bool:
    """Return whether a SMILES has no attachment points, either [*] or [n*]."""
    s = str(s)
    return ("[*]" not in s) and (not _BRICS_LABEL.search(s))

def partial_crossover(parent1, parent2, fragment_fitness_df, time_limit=1,
                      log_fragments_path=None, log_products_path=None):
    """
    BRICS-based partial crossover from two parent molecules:
      1) decompose both parents into BRICS fragments;
      2) sample additional high-scoring fragments from fragment_fitness_df;
      3) directly add sampled complete molecules to new_smiles;
      4) assemble the remaining BRICS-labeled fragments with time_limited_build;
      5) return new_smiles, including direct complete molecules and assembled products.

    Optional CSV logs can be written through log_fragments_path and
    log_products_path.
    """
    start_time = time.time()

    mol1 = Chem.MolFromSmiles(parent1)
    mol2 = Chem.MolFromSmiles(parent2)
    if mol1 is None or mol2 is None:
        return []

    # BRICS decomposition and fragment selection
    fragments1 = fragment_with_brics(mol1)
    fragments2 = fragment_with_brics(mol2)
    combined_fragments = fragments1 + fragments2
    # print(f'length_combined_fragments={len(combined_fragments)}')

    # Validate the fragment fitness table
    if (not isinstance(fragment_fitness_df, pd.DataFrame)
        or 'smiles' not in fragment_fitness_df.columns
        or 'sis_values' not in fragment_fitness_df.columns):
        raise ValueError("fragment_fitness_df must contain columns: ['smiles', 'sis_values']")

    fitness_scores_of_frag = dict(zip(fragment_fitness_df['smiles'], fragment_fitness_df['sis_values']))
    smiles_list_combined = fragment_fitness_df['smiles'].astype(str).tolist()

    new_smiles = []          # Final return list
    selected_fragments = []  # BRICS-labeled fragments available for assembly
    fragment = []

    # Fragment sampling using the configured quotas
    fragment.extend(select_based_on_fitness_of_fragments(
        smiles_list=smiles_list_combined,
        fitness_scores=fitness_scores_of_frag,
        k_nohub_tournament=k_nohub_tour,
        k_nohub_random=k_nohub_rand,
        k_hub_tournament=k_hub_tour,
        k_hub_random=k_hub_rand,
        tournament_size=300,
        map_base=9000,
        deduplicate=False,
    ))

    # Complete molecules are directly added; the remaining fragments enter the assembly pool.
    for f in fragment:
        f = str(f)
        if _is_complete_molecule(f):
            new_smiles.append(f)  # Directly add complete molecules.
        else:
            selected_fragments.append(f)

    # Merge the fragment library and call the BRICS assembly API.
    # Only digit-labeled BRICS fragments enter assembly; deduplicate the library.
    building_pool = [f for f in (fragments1 + fragments2 + selected_fragments) if _BRICS_LABEL.search(str(f))]
    building_pool = sorted(set(building_pool))

    # # Log the fragment library used for assembly.
    # print(f"[build] fragment library size: {len(building_pool)}")
    # for i, frag in enumerate(building_pool, 1):
    #     print(f"  [FRAG {i:04d}] {frag}")

    # # Optional CSV log for the fragment library.
    # if log_fragments_path:
    #     try:
    #         pd.DataFrame({"fragment": building_pool}).to_csv(log_fragments_path, index=False)
    #         print(f"[build] fragment library saved to: {log_fragments_path}")
    #     except Exception as e:
    #         print(f"[warn] failed to save fragment library: {e}")

    if not building_pool:
        # print("No labeled BRICS fragments available for building; returning directly collected molecules.")
        elapsed_time = time.time() - start_time
        # print(f"SMILES generation time: {elapsed_time:.2f} s, generated new_smiles count: {len(new_smiles)}")
        return new_smiles

    # Time-limited assembly through brics_build_api.
    result = time_limited_build(
        fragment_library=building_pool,
        max_steps=10,
        sa_threshold=6.0,
        seconds=time_limit,
        per_call_max_attempts=10000,
        seed=None,
        reject_open=False,
    )

    # Extract assembled products; time_limited_build already deduplicates them.
    built_smiles = [it["smiles"] for it in result.get("items", [])]
    
    # # Optional CSV log for assembled products.
    # if log_products_path:
    #     try:
    #         pd.DataFrame({"smiles": built_smiles}).to_csv(log_products_path, index=False)
    #         print(f"[build] built molecules saved to: {log_products_path}")
    #     except Exception as e:
    #         print(f"[warn] failed to save built molecules: {e}")

    # Append assembled products to the return list.
    new_smiles.extend(built_smiles)

    elapsed_time = time.time() - start_time
    
    # Final filtering: keep products that contain hub dummies [*:9000+] or dot-separated mixtures.
    before_n = len(new_smiles)
    new_smiles = [
    s for s in new_smiles
    if _has_hub(s, map_base=9000) or (isinstance(s, str) and "." in s)
]

    # # Log assembled products after hub filtering.
    # print(f"[build] built molecules (hub-only): {len(new_smiles)} / {before_n}")
    # for i, smi in enumerate(new_smiles, 1):
    #     print(f"  [PROD {i:04d}] {smi}")
    # print(f"SMILES generation time: {elapsed_time:.2f} s, generated new_smiles count: {len(new_smiles)}")
    return new_smiles



# Additional crossover implementations can be added here if needed.

__all__ = [
    "mol_to_smiles",
    "fragment_with_brics",
    "conservative_mutation",
    "is_valid_smiles",
    "select_based_on_fitness_of_fragments",
    "partial_crossover",
]
