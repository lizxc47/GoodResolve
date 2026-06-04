# -*- coding: utf-8 -*-
"""
all_prediction_GA_small_molecular.py

GA/BRICS optimization for SMILES candidates. All spectrum prediction and SIS
calculation are delegated to ir_sis.py.

This version is adapted for the Graphormer-IR-inspired forward predictor:
  - ir_sis.load_models(checkpoint, device=None, local_pytorch_dir=...) -> (graphormer_model, model_meta, device)
  - ir_sis.predict_infrared(smiles_list, graphormer_model, model_meta, device, phase=...) -> List[np.ndarray(1801,)]
  - ir_sis.calculate_sis(pred, target, sigma=10) -> float
  - ir_sis.best_sis_for_mixture(mixture, ir_matrix, graphormer_model, model_meta, device, sigma, phase=...) -> float

Phase handling:
  - The phase is read from the phase column of the IR CSV specified by --ir.
  - For example, KBr disc is mapped to KBr inside ir_sis.py.
  - To override the phase manually, use --target-phase.

Compatible IR CSV formats:
  - smiles, phase, 400, 402, ..., 4000
  - smiles, phase, sp_400, sp_402, ..., sp_4000
  - smiles, phase, sp_0, ..., sp_1800
  - legacy sp_50..sp_1750 format

Outputs:
  - generation_<idx>_elites.csv is written for each GA generation.
"""

import csv
import random
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d  # Kept for compatibility with previous versions

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch

# ==== Prediction / SIS utilities from ir_sis.py ====
from ir_sis import (
    load_models as ir_load_models,
    load_ir_matrix as ir_load_ir_matrix,
    read_phase_from_ir_csv as ir_read_phase_from_ir_csv,
    predict_infrared as ir_predict_infrared,
    calculate_sis as ir_calculate_sis,
    best_sis_for_mixture as ir_best_sis_for_mixture,
)

# ==== Genetic operations from ga_ops.py, which internally depends on mol_ops.py ====
from ga_ops import (
    select_based_on_fitness,
    genetic_operations_with_population_update,
)

# -----------------------------
# Global model handles initialized in main
# -----------------------------
graphormer_model = None
model_meta = None
device = None
target_phase = None


def predict_infrared(smiles_list):
    """
    Compatibility wrapper around ir_sis.predict_infrared.
    The phase is taken from target_phase, which is read from IR_data.csv.
    """
    return ir_predict_infrared(
        smiles_list,
        gnn=graphormer_model,
        kan=model_meta,
        device=device,
        phase=target_phase,
    )


def _align_prediction_to_target(pred_1801, ir_spectra):
    """
    Align the 1801-point predicted spectrum to the target spectrum length.
    """
    pred_1801 = np.asarray(pred_1801, dtype=np.float64).reshape(-1)
    n = int(np.asarray(ir_spectra).shape[1])

    if n == len(pred_1801):
        return pred_1801
    if n == 1701 and len(pred_1801) >= 1751:
        return pred_1801[50:1751]
    if n < len(pred_1801):
        return pred_1801[:n]
    raise ValueError(f"Predicted spectrum length {len(pred_1801)} cannot be aligned to target spectrum length {n}")


def calculate_fitness(smiles, ir_spectra, sigma=10):
    """
    Fitness score based on SIS. Higher is better.
      - If the SMILES contains '.', it is treated as a mixture and scored by ir_best_sis_for_mixture.
      - Otherwise, the compound spectrum is predicted and compared with each target spectrum; the maximum SIS is used.
    """
    try:
        if '.' in smiles:
            return float(ir_best_sis_for_mixture(
                smiles,
                ir_spectra,
                graphormer_model,
                model_meta,
                device,
                sigma=sigma,
                phase=target_phase,
            ))

        pred = predict_infrared([smiles])[0]
        y_pred = _align_prediction_to_target(pred, ir_spectra)

        best = -1.0
        for tgt in ir_spectra:
            sis = ir_calculate_sis(y_pred, np.asarray(tgt, dtype=np.float64), sigma=sigma)
            if sis > best:
                best = sis
        return float(best)

    except Exception:
        # Treat any failed candidate as very low fitness to keep GA running.
        return -1.0


def calculate_and_save_fragment_fitness(input_file="data_fragments.csv",
                                        output_file="fragment_fitness.csv",
                                        ir_spectra=None,
                                        sigma=10):
    """
    Read the 'smiles' column from a fragment CSV and score each fragment as a molecule.
    Spectra are predicted in batch with the Graphormer-IR-inspired predictor, then SIS is computed against each target spectrum and the maximum value is used.
    """
    df = pd.read_csv(input_file)
    if 'smiles' not in df.columns:
        raise ValueError(f"{input_file} is missing the required column: smiles")

    smiles_list = df['smiles'].astype(str).tolist()
    preds = predict_infrared(smiles_list)

    ir_spectra = np.asarray(ir_spectra, dtype=np.float64)

    sis_vals = []
    for pred in tqdm(preds, desc="Fragment SIS via Graphormer-IR-inspired ir_sis"):
        y_pred = _align_prediction_to_target(pred, ir_spectra)
        best = -1.0
        for tgt in ir_spectra:
            sis = ir_calculate_sis(y_pred, tgt, sigma=sigma)
            if sis > best:
                best = sis
        sis_vals.append(float(best))

    out = pd.DataFrame({'smiles': smiles_list, 'sis_values': sis_vals})
    out.to_csv(output_file, index=False)
    return out


# -----------------------------
# I/O and main workflow
# -----------------------------
def read_smiles_with_sis(file_path, threshold=0.6, smiles_col="SMILES", sis_col="SIS", top_n=1000):
    """
    Read `smiles_col` and `sis_col` from the input CSV.

    Current logic:
      - the initial population is no longer filtered by SIS > threshold;
      - rows are sorted by SIS in descending order;
      - only the top_n initial SMILES are used.

    Notes:
      The threshold argument is kept for backward compatibility with old commands and scripts, but it is no longer used for initial filtering.
      If top_n <= 0, all rows are used.
    """
    df = pd.read_csv(file_path)
    if smiles_col not in df.columns or sis_col not in df.columns:
        raise ValueError(f"{file_path} must contain columns: {smiles_col} and {sis_col}")

    df = df.copy()
    df[sis_col] = pd.to_numeric(df[sis_col], errors="coerce")
    df = df.dropna(subset=[smiles_col, sis_col])
    df = df.sort_values(sis_col, ascending=False)

    if top_n is not None and int(top_n) > 0:
        df = df.head(int(top_n))

    return df[smiles_col].astype(str).tolist()


def save_elites_to_csv(elites, fitness_scores, generation_idx):
    output_file = f'generation_{generation_idx + 1}_elites.csv'
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Generation', 'SMILES', 'SIS'])
        for s in elites:
            csvwriter.writerow([generation_idx + 1, s, fitness_scores.get(s, None)])
    print(f"Generation {generation_idx + 1} elites saved to file: {output_file}")


def parse_args():
    ap = argparse.ArgumentParser(description="GA + BRICS optimization using Graphormer-IR-inspired ir_sis for spectrum prediction and SIS scoring")
    ap.add_argument("--checkpoint", default="../../best_model.pt",
                    help="Graphormer-IR-inspired checkpoint, e.g., checkpoints_sdbs_graphormer_ir_1minusT/best_model.pt")
    ap.add_argument("--local-pytorch-dir", default="local_pytorch",
                    help="Directory containing graphormer_ir_local.py")
    ap.add_argument("--ir", default="IR_data.csv",
                    help="IR CSV; expected to contain smiles, phase, and 400..4000 or sp_* spectral columns")
    ap.add_argument("--phase-col", default="phase", help="Phase column name in the IR CSV")
    ap.add_argument("--target-phase", default=None,
                    help="Manually specify the target phase; if omitted, the phase is read from the phase column in --ir")
    ap.add_argument("--input-smiles", default="smiles_sis_predictions.csv",
                    help="Initial SMILES candidate file; must contain SMILES and SIS columns")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Kept for backward compatibility; the initial population is now selected by SIS top-N ranking rather than threshold filtering")
    ap.add_argument("--initial-top-n", type=int, default=1000,
                    help="Number of initial candidates read from --input-smiles; default is top 1000 by SIS; <=0 means using all candidates")
    ap.add_argument("--fragments", default='../../data_fragments.csv', help="Fragment CSV; must contain a smiles column")
    ap.add_argument("--frag-out", default='fragment_fitness.csv', help="Output file for fragment fitness scores")
    ap.add_argument("--num-generate", type=int, default=1000, help="Number of molecules to generate per generation")
    ap.add_argument("--generations", type=int, default=20, help="Number of GA generations")
    ap.add_argument("--crossover-rate", type=float, default=0.6, help="Crossover probability")
    ap.add_argument("--mutation-rate", type=float, default=0.02, help="Mutation probability")
    ap.add_argument("--elite-fraction", type=float, default=0.2, help="Elite fraction")
    ap.add_argument("--sigma", type=int, default=10, help="Gaussian smoothing sigma for SIS, in points")
    ap.add_argument("--device", default=None, help="Device: cuda / cuda:0 / cpu; default is automatic")
    ap.add_argument("--seed", type=int, default=None, help="Random seed")
    # Optional: allow mixture candidates to pass the validity check during GA.
    ap.add_argument("--allow-mixtures", action="store_true",
                    help="Allow A.B-style mixture candidates during genetic operations; mol_ops must also permit them")
    return ap.parse_args()


def main():
    global graphormer_model, model_meta, device, target_phase

    args = parse_args()

    # Random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Device
    _dev = None if args.device is None else args.device

    # Read target phase
    if args.target_phase is not None:
        target_phase = args.target_phase
    else:
        target_phase = ir_read_phase_from_ir_csv(args.ir, phase_col=args.phase_col, default_phase="KBr")
    print(f"[ir_sis] Target phase: {target_phase}")

    # Load the Graphormer-IR-inspired forward predictor once
    graphormer_model, model_meta, device = ir_load_models(
        args.checkpoint,
        device=_dev,
        local_pytorch_dir=args.local_pytorch_dir,
    )
    print(f"[ir_sis] Using device: {device}")

    # Load target IR matrix
    ir_spectra = ir_load_ir_matrix(args.ir)
    print(f"[ir_sis] Loaded target IR matrix: {ir_spectra.shape}")

    # Score fragments by treating each fragment as a molecule
    fragment_fitness_df = calculate_and_save_fragment_fitness(
        input_file=args.fragments,
        output_file=args.frag_out,
        ir_spectra=ir_spectra,
        sigma=args.sigma
    )

    # Initial population
    smiles_list = read_smiles_with_sis(
        args.input_smiles,
        threshold=args.threshold,
        smiles_col="SMILES",
        sis_col="SIS",
        top_n=args.initial_top_n,
    )
    if args.initial_top_n is not None and args.initial_top_n > 0:
        print(f"Loaded initial SMILES: {len(smiles_list)} candidates selected by SIS top-{args.initial_top_n} ranking")
    else:
        print(f"Loaded initial SMILES: {len(smiles_list)} candidates using all rows")

    best_smiles = None
    best_sis = -np.inf

    for gen in range(args.generations):
        print(f"\nStarting generation {gen + 1} ...")

        # One GA generation plus population update. Mixture validity is controlled by --allow-mixtures.
        smiles_list, fitness_scores = genetic_operations_with_population_update(
            smiles_list=smiles_list,
            fitness_fn=lambda s: calculate_fitness(s, ir_spectra, sigma=args.sigma),
            fragment_fitness_df=fragment_fitness_df,
            num_to_generate=args.num_generate,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            elite_fraction=args.elite_fraction,
            treat_mixtures_as_invalid=not args.allow_mixtures,
        )

        # Elites for the current generation
        elites = sorted(fitness_scores, key=fitness_scores.get, reverse=True)[:max(1, int(len(smiles_list) * args.elite_fraction))]
        save_elites_to_csv(elites, fitness_scores, gen)

        # Track global best candidate
        for s in smiles_list:
            cur = fitness_scores.get(s, None)
            if cur is None:
                cur = calculate_fitness(s, ir_spectra, sigma=args.sigma)
            if cur > best_sis:
                best_sis = cur
                best_smiles = s

        if np.isfinite(best_sis):
            print(f"Best SIS in generation {gen + 1}: {best_sis:.6f}; SMILES: {best_smiles}")
        else:
            print(f"Best SIS in generation {gen + 1}: {best_sis}; SMILES: {best_smiles}")

    # Final result
    with open('optimized_smiles.txt', 'w') as f:
        f.write(f"Best SMILES: {best_smiles}\nBest SIS: {best_sis}\n")
    print("Best result written to optimized_smiles.txt")


if __name__ == "__main__":
    main()
