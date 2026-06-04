# -*- coding: utf-8 -*-
"""
ga_ops.py

Genetic-operation utilities for the component-wise GA/BRICS workflow.

This module provides:
- select_based_on_fitness(pop_smiles, fitness_scores)
- genetic_operations_with_population_update(...)

The GA update uses an externally supplied fitness function, performs
fitness-weighted parent selection, hub-style single-molecule conversion for
crossover, conservative mutation, validity checking, and elite retention.
"""

import time
import random
from typing import Callable, Dict, List, Tuple, Optional

import numpy as np
from tqdm import tqdm

from mol_ops import (
    is_valid_smiles,
    conservative_mutation,
    partial_crossover,
)

from hub_brics_canonical import ab_to_unimol, unimol_to_ab
from rdkit import Chem
import pandas as pd



__all__ = [
    "select_based_on_fitness",
    "genetic_operations_with_population_update",
]

def select_based_on_fitness(pop_smiles: List[str], fitness_scores: Dict[str, float]) -> str:
    """
    Select one individual by fitness-proportional sampling; fall back to uniform sampling if the total fitness is non-positive.
    """
    total_fitness = sum(fitness_scores[s] for s in pop_smiles)
    if total_fitness <= 0 or not np.isfinite(total_fitness):
        probs = [1.0 / len(pop_smiles)] * len(pop_smiles)
    else:
        probs = [fitness_scores[s] / total_fitness for s in pop_smiles]
    return np.random.choice(pop_smiles, p=probs)

def genetic_operations_with_population_update(
    smiles_list: List[str],
    fitness_fn: Callable[[str], float],
    fragment_fitness_df: pd.DataFrame,   # must contain ['smiles', 'sis_values']
    num_to_generate: int,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    elite_fraction: float = 0.1,
    treat_single_as_invalid: bool = False,
    max_components: Optional[int] = None,
    atom_choices: Optional[List[str]] = None,
    progress: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    Perform one round of genetic operations and update the population.

    The function:
      - computes fitness for the current population;
      - keeps elite candidates;
      - selects parent pairs by fitness;
      - converts parent mixtures to hub-style single-molecule representations;
      - applies crossover and optional conservative mutation;
      - converts offspring back to component-separated SMILES;
      - applies validity filtering and fitness scoring;
      - returns the next population and the fitness-score dictionary.
    """
    if atom_choices is None:
        atom_choices = ['C', 'N', 'P', 'S', 'O', 'Si', 'F', 'Cl', 'Br', 'I']

    # Unified validity check
    def _is_valid(sm: str) -> bool:
        return is_valid_smiles(
            sm,
            max_components=max_components,
            treat_single_as_invalid=treat_single_as_invalid,
        )

    # 1) Compute fitness for the current population
    print("Computing fitness for the current population using the external fitness_fn")
    it = tqdm(smiles_list, disable=not progress)
    fitness_scores: Dict[str, float] = {s: float(fitness_fn(s)) for s in it}

    # 2) Keep elites
    num_elites = max(1, int(len(smiles_list) * elite_fraction))
    elites = sorted(fitness_scores, key=fitness_scores.get, reverse=True)[:num_elites]

    # 3) Generate new individuals
    new_smiles_set = set()
    pbar = tqdm(total=num_to_generate, desc='Generating new SMILES', unit='SMILES', disable=not progress)
    try:
        while len(new_smiles_set) < num_to_generate:
            parent1 = select_based_on_fitness(smiles_list, fitness_scores)
            parent2 = select_based_on_fitness(smiles_list, fitness_scores)

            # Convert parents to hub-style single-molecule representations for crossover and mutation.
            try:
                p1_u_mol = ab_to_unimol(parent1)
                p2_u_mol = ab_to_unimol(parent2)
                parent1_unimol = Chem.MolToSmiles(p1_u_mol, canonical=True, isomericSmiles=True)
                parent2_unimol = Chem.MolToSmiles(p2_u_mol, canonical=True, isomericSmiles=True)
            except Exception:
                # Fallback: use the original strings if conversion fails.
                parent1_unimol = parent1
                parent2_unimol = parent2

            offspring_smiles_list: List[str] = []   # component-separated form used for population update and scoring
            offspring_scores: List[float] = []

            if random.random() < crossover_rate:
                # partial_crossover receives hub-style single-molecule parents and returns hub-style offspring.
                offspring_unimol_list = partial_crossover(parent1_unimol, parent2_unimol, fragment_fitness_df)
                start_time = time.time()

                for off_u in offspring_unimol_list:
                    # Apply mutation in the hub-style single-molecule space.
                    if random.random() < mutation_rate:
                        try:
                            mutated_u = conservative_mutation(off_u, atom_choices)
                            if mutated_u:
                                off_u = mutated_u
                        except Exception:
                            pass  # If mutation fails, keep the original offspring.

                    # Convert hub-style offspring back to component-separated SMILES before scoring.
                    try:
                        off_ab = unimol_to_ab(off_u)
                    except Exception:
                        continue

                    # Validity filtering: allow at most max_components components; single-component candidates can optionally be rejected.
                    if not _is_valid(off_ab):
                        continue

                    # Fitness scoring on the component-separated representation.
                    try:
                        score = float(fitness_fn(off_ab))
                    except Exception:
                        continue

                    offspring_smiles_list.append(off_ab)
                    offspring_scores.append(score)

                if offspring_scores:
                    # Add only the best offspring from this parent-pair attempt.
                    sorted_indices = sorted(range(len(offspring_scores)), key=lambda i: offspring_scores[i], reverse=True)
                    for idx in sorted_indices:
                        cand_ab = offspring_smiles_list[idx]  # component-separated representation
                        if cand_ab not in new_smiles_set:
                            new_smiles_set.add(cand_ab)
                            pbar.update(1)
                            break

                elapsed = time.time() - start_time
                # print(f"Population update time: {elapsed:.2f} s; current new population size: {len(new_smiles_set)}")

            if len(new_smiles_set) >= num_to_generate:
                break
    finally:
        pbar.close()

    # 4) Add fitness scores for new individuals, avoiding duplicate computation.
    it2 = tqdm(new_smiles_set, desc="Computing fitness for new individuals", disable=not progress)
    for s in it2:
        if s not in fitness_scores:
            fitness_scores[s] = float(fitness_fn(s))

    # 5) Merge new individuals with elites to form the next generation.
    new_population = list(new_smiles_set)[:max(0, num_to_generate - num_elites)] + elites
    return new_population, fitness_scores

