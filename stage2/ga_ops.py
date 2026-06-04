# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:34:53 2025

@author: Lizxc
"""

# -*- coding: utf-8 -*-
"""
ga_ops.py

Standalone module for genetic operations:
- select_based_on_fitness(pop_smiles, fitness_scores)
- genetic_operations_with_population_update(...)

Notes:
- Fitness calculation is injected through a callback: fitness_fn(smiles) -> float.
  It is recommended to pass the calculate_fitness function defined in the GA script.
- Crossover, mutation, and validity checks rely on mol_ops.py, preserving the
  original behavior.
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
__all__ = [
    "select_based_on_fitness",
    "genetic_operations_with_population_update",
]

def select_based_on_fitness(pop_smiles: List[str], fitness_scores: Dict[str, float]) -> str:
    """
    Select one individual with fitness-proportional sampling. If the total fitness is <= 0, fall back to uniform sampling.
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
    fragment_fitness_df,                # pandas.DataFrame; must contain ['smiles', 'sis_values']
    num_to_generate: int,
    crossover_rate: float = 0.6,
    mutation_rate: float = 0.3,
    elite_fraction: float = 0.1,
    treat_mixtures_as_invalid: bool = True,
    atom_choices: Optional[List[str]] = None,
    progress: bool = True,
) -> Tuple[List[str], Dict[str, float]]:
    """
    One generation of genetic operations plus population update, preserving the original behavior:
      - compute fitness for the current population;
      - select parents -> BRICS crossover -> conservative mutation -> validity check -> scoring;
      - add the best offspring from each attempt to new_smiles_set until the target size is reached;
      - keep elite candidates.

    Parameters:
      - smiles_list: current population as SMILES strings; mixtures may be included, but validity is controlled by treat_mixtures_as_invalid.
      - fitness_fn: callback that takes a SMILES string and returns a fitness score. Higher is better. The calculate_fitness function from the GA script can be passed directly.
      - fragment_fitness_df: externally provided fragment fitness table.
      - num_to_generate: target number of new individuals.
      - treat_mixtures_as_invalid: when True, SMILES containing '.' are filtered out, matching the original logic.
      - atom_choices: atom symbols available for mutation; defaults follow the original script.
      - progress: whether to show tqdm progress bars.

    Returns:
      - new_population: next-generation population, including new individuals and elites.
      - fitness_scores: dict mapping SMILES to fitness scores, including newly generated individuals.
    """
    if atom_choices is None:
        atom_choices = ['C', 'N', 'P', 'S', 'O', 'Si', 'F', 'Cl', 'Br', 'I']

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

            offspring_smiles_list: List[str] = []
            offspring_scores: List[float] = []

            if random.random() < crossover_rate:
                offspring_smiles = partial_crossover(parent1, parent2, fragment_fitness_df)
                start_time = time.time()

                for off in offspring_smiles:
                    # Validity filtering, preserving the original behavior
                    if not is_valid_smiles(off, treat_mixtures_as_invalid=treat_mixtures_as_invalid):
                        continue

                    # Conservative mutation
                    if random.random() < mutation_rate:
                        mutated = conservative_mutation(off, atom_choices)
                        if mutated and is_valid_smiles(mutated, treat_mixtures_as_invalid=treat_mixtures_as_invalid):
                            off = mutated

                    # Fitness scoring
                    score = float(fitness_fn(off))
                    offspring_smiles_list.append(off)
                    offspring_scores.append(score)

                if offspring_scores:
                    # Add only the best offspring, consistent with the original script
                    sorted_indices = sorted(range(len(offspring_scores)), key=lambda i: offspring_scores[i], reverse=True)
                    for idx in sorted_indices:
                        cand = offspring_smiles_list[idx]
                        if cand not in new_smiles_set:
                            new_smiles_set.add(cand)
                            pbar.update(1)
                            break

                elapsed = time.time() - start_time
                print(f"Population update time: {elapsed:.2f} s; current new population size: {len(new_smiles_set)}")

            if len(new_smiles_set) >= num_to_generate:
                break
    finally:
        pbar.close()

    # 4) Add fitness scores for new individuals, avoiding duplicate computation
    it2 = tqdm(new_smiles_set, desc="Computing fitness for new individuals", disable=not progress)
    for s in it2:
        if s not in fitness_scores:
            fitness_scores[s] = float(fitness_fn(s))

    # 5) Merge new individuals with elites to form the next generation
    new_population = list(new_smiles_set)[:max(0, num_to_generate - num_elites)] + elites
    return new_population, fitness_scores
