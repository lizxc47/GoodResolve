# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:26:42 2025

@author: Lizxc
"""

# -*- coding: utf-8 -*-
"""
mol_ops.py

Molecular fragmentation, recombination, mutation, and validity-check utilities
used by the GA workflow.

Main functions:
- mol_to_smiles(mol)
- fragment_with_brics(mol)
- conservative_mutation(smiles, atom_choices)
- is_valid_smiles(smiles, treat_mixtures_as_invalid=True)
- select_based_on_fitness_of_fragments(smiles_list, fitness_scores, k=1, tournament_size=300)
- partial_crossover(parent1, parent2, fragment_fitness_df, time_limit=10)

This file focuses only on structure-generation operations. Spectrum prediction
and SIS scoring are handled elsewhere.
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
    Conservative mutation: randomly remove one atom, add a new atom, and try to reconnect it to neighbors with single bonds.
    This operation may produce mixtures or disconnected structures; downstream validity checks handle invalid candidates.
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

def is_valid_smiles(smiles, treat_mixtures_as_invalid=True):
    """
    Validity check:
      - if treat_mixtures_as_invalid is True, SMILES containing '.' are treated as invalid;
      - the SMILES must be parseable by RDKit;
      - each component must have length >= 3.
    """
    if treat_mixtures_as_invalid and ('.' in smiles):
        return False
    if Chem.MolFromSmiles(smiles) is None:
        return False
    components = smiles.split('.')
    for component in components:
        if len(component) < 3:
            return False
    return True

def select_based_on_fitness_of_fragments(smiles_list, fitness_scores, k=1, tournament_size=300):
    """
    Fragment sampling using a mixture of tournament selection and random selection.
    smiles_list: list of selectable fragment SMILES.
    fitness_scores: dict mapping SMILES to scores.
    k: number of fragments to select.
    tournament_size: maximum number of tournament contestants.
    """
    selected_fragments = []
    if not smiles_list:
        return selected_fragments
    sampling_method = np.random.choice(["tournament", "random"])
    for _ in range(k):
        if sampling_method == "tournament":
            size = min(tournament_size, len(smiles_list))
            tournament_contestants = np.random.choice(smiles_list, size=size, replace=False)
            best_fragment = max(tournament_contestants, key=lambda frag: fitness_scores.get(frag, 0.0))
            selected_fragments.append(best_fragment)
        else:
            random_fragment = np.random.choice(smiles_list)
            selected_fragments.append(random_fragment)
    return selected_fragments

# -----------------------------
# BRICS crossover logic
# ----------------------------
_BRICS_LABEL = re.compile(r"\[\d+\*\]")  # BRICS labels such as [14*]

def _is_complete_molecule(s: str) -> bool:
    """A molecule is considered complete if it has no attachment points, either [*] or [n*]."""
    s = str(s)
    return ("[*]" not in s) and (not _BRICS_LABEL.search(s))

def partial_crossover(parent1, parent2, fragment_fitness_df, time_limit=1):
    """
    BRICS-based partial crossover from two parent molecules:
      1) decompose both parents into BRICS fragments;
      2) sample additional high-scoring fragments from fragment_fitness_df;
      3) directly add sampled complete molecules to new_smiles;
      4) assemble the remaining BRICS-labeled fragments through brics_build_api.time_limited_build;
      5) return the generated SMILES list, including directly collected complete molecules and assembled products.
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
    print(f'length_combined_fragments={len(combined_fragments)}')

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

    # Number of fragments to sample
    selection_count = len(combined_fragments) // 2 if len(combined_fragments) // 2 > 4 else 4
    fragment.extend(
        select_based_on_fitness_of_fragments(smiles_list_combined, fitness_scores_of_frag, selection_count)
    )

    # Complete molecules are directly added; the remaining fragments enter the assembly pool.
    for f in fragment:
        f = str(f)
        if _is_complete_molecule(f):
            # Directly add complete molecules without deduplication or scoring.
            new_smiles.append(f)
        else:
            selected_fragments.append(f)

    # Merge the fragment library and call the BRICS assembly API.
    # Only digit-labeled BRICS fragments enter assembly.
    building_pool = [f for f in (fragments1 + fragments2 + selected_fragments) if _BRICS_LABEL.search(str(f))]
    building_pool = sorted(set(building_pool))  # Deduplicate the fragment library only; this does not affect output deduplication.

    if not building_pool:
        print("No labeled BRICS fragments available for building; returning directly collected molecules.")
        elapsed_time = time.time() - start_time
        print(f"SMILES generation time: {elapsed_time:.2f} s; generated new_smiles count: {len(new_smiles)}")
        return new_smiles

    # Time-limited assembly using brics_build_api:
    # - a high sa_threshold effectively disables strict SA filtering;
    # - reject_open=True discards products that still contain '*';
    # - time_limited_build deduplicates assembled products.
    result = time_limited_build(
        fragment_library=building_pool,
        max_steps=5,
        sa_threshold=5,
        seconds=time_limit,
        per_call_max_attempts=80,
        seed=None,
        reject_open=True,
    )

    # Append assembled products to the return list. These products are deduplicated by time_limited_build.
    new_smiles.extend([it["smiles"] for it in result["items"]])

    elapsed_time = time.time() - start_time
    print(f"SMILES generation time: {elapsed_time:.2f} s; generated new_smiles count: {len(new_smiles)}")
    return new_smiles


# Alternative crossover implementations can be added here if needed.

__all__ = [
    "mol_to_smiles",
    "fragment_with_brics",
    "conservative_mutation",
    "is_valid_smiles",
    "select_based_on_fitness_of_fragments",
    "partial_crossover",
]
