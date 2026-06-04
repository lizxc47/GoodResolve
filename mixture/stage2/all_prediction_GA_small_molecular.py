# -*- coding: utf-8 -*-
"""
all_prediction_GA_small_molecular_componentwise.py

Component-wise GA/BRICS optimization for molecular and mixture candidates.

This script runs GA-based candidate optimization separately for different
component counts. It reads initial candidates from component-specific folders,
uses the corresponding fragment library for each component count, evaluates
candidate fitness by IR prediction and SIS scoring, and writes optimized
candidates back to each component folder.

Default data_* folder layout
----------------------------
Run this script inside one data_* folder:

    data_0/
    ├── IR_data.csv
    ├── components_1/
    │   ├── smiles_sis_predictions.csv
    │   └── data_fragments.csv
    ├── components_2/
    │   ├── smiles_sis_predictions.csv
    │   └── data_fragments.csv
    └── components_3/
        ├── smiles_sis_predictions.csv
        └── data_fragments.csv

If fragment libraries are outside components_k/, specify:
    --fragment-template "/path/to/data_fragments_{k}.csv"

For each component count k, this script enforces exactly k components:
    k=1: only single-component SMILES are valid.
    k=2: only A.B mixtures are valid.
    k=3: only A.B.C mixtures are valid.
"""

import os
import csv
import argparse
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import torch

from ir_sis import (
    load_models as ir_load_models,
    load_ir_matrix as ir_load_ir_matrix,
    read_phase_from_ir_csv as ir_read_phase_from_ir_csv,
    predict_infrared as ir_predict_infrared,
    calculate_sis as ir_calculate_sis,
    best_sis_for_mixture as ir_best_sis_for_mixture,
)

from ga_ops import (
    select_based_on_fitness,
    genetic_operations_with_population_update,
)


graphormer_model = None
model_meta = None
device = None
target_phase = None


def count_components(smiles: str) -> int:
    parts = [p.strip() for p in str(smiles).split(".") if p.strip()]
    return len(parts)


def is_exact_component_count(smiles: str, k: int) -> bool:
    return count_components(smiles) == int(k)


def predict_infrared(smiles_list):
    try:
        return ir_predict_infrared(
            smiles_list,
            graphormer_model,
            model_meta,
            device,
            phase=target_phase,
        )
    except TypeError:
        return ir_predict_infrared(
            smiles_list,
            graphormer_model,
            model_meta,
            device,
        )


def _align_prediction_to_target(pred_1801, ir_spectra):
    pred_1801 = np.asarray(pred_1801, dtype=np.float64).reshape(-1)
    n = int(np.asarray(ir_spectra).shape[1])

    if n == len(pred_1801):
        return pred_1801
    if n == 1701 and len(pred_1801) >= 1751:
        return pred_1801[50:1751]
    if n < len(pred_1801):
        return pred_1801[:n]

    raise ValueError(f"Predicted spectrum length {len(pred_1801)} cannot be aligned to target spectrum length {n}")


def calculate_fitness(smiles, ir_spectra, sigma=10, exact_components=None):
    try:
        if exact_components is not None and not is_exact_component_count(smiles, exact_components):
            return -1.0

        if "." in str(smiles):
            try:
                return float(ir_best_sis_for_mixture(
                    smiles,
                    ir_spectra,
                    graphormer_model,
                    model_meta,
                    device,
                    sigma=sigma,
                    phase=target_phase,
                ))
            except TypeError:
                return float(ir_best_sis_for_mixture(
                    smiles,
                    ir_spectra,
                    graphormer_model,
                    model_meta,
                    device,
                    sigma=sigma,
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
        return -1.0


def calculate_and_save_fragment_fitness(input_file, output_file, ir_spectra, sigma=10):
    df = pd.read_csv(input_file)
    if "smiles" not in df.columns:
        raise ValueError(f"{input_file} is missing the required column: smiles")

    smiles_list = df["smiles"].astype(str).tolist()
    preds = predict_infrared(smiles_list)
    ir_spectra = np.asarray(ir_spectra, dtype=np.float64)

    sis_vals = []
    for pred in tqdm(preds, desc=f"Fragment SIS: {os.path.basename(input_file)}"):
        y_pred = _align_prediction_to_target(pred, ir_spectra)
        best = -1.0
        for tgt in ir_spectra:
            sis = ir_calculate_sis(y_pred, tgt, sigma=sigma)
            if sis > best:
                best = sis
        sis_vals.append(float(best))

    out = pd.DataFrame({"smiles": smiles_list, "sis_values": sis_vals})
    out.to_csv(output_file, index=False)
    return out


def read_smiles_with_sis(
    file_path,
    threshold=0.6,
    smiles_col="SMILES",
    sis_col="SIS",
    exact_components=None,
    top_n=1000,
):
    """
    Read initial candidates from smiles_sis_predictions.csv.

    Selection behavior:
      - sort candidates by SIS in descending order;
      - use the top_n rows for each component input file;
      - apply exact component-count filtering and de-duplication.

    The threshold argument is accepted, while initial selection is controlled
    by top_n ranking.
    """
    df = pd.read_csv(file_path)

    if smiles_col not in df.columns or sis_col not in df.columns:
        raise ValueError(
            f"{file_path} must contain columns: {smiles_col} and {sis_col}."
            f"Available columns: {list(df.columns)}"
        )

    df = df.copy()
    df[sis_col] = pd.to_numeric(df[sis_col], errors="coerce")
    df = df.dropna(subset=[smiles_col, sis_col])
    df = df.sort_values(sis_col, ascending=False)

    if top_n is not None and int(top_n) > 0:
        before_topn = len(df)
        df = df.head(int(top_n))
        print(f"[Initial filter] top_n={int(top_n)} by {sis_col}: {before_topn} -> {len(df)}")
    else:
        print(f"[Initial filter] top_n disabled; using all rows sorted by {sis_col}: {len(df)}")

    smiles_list = df[smiles_col].astype(str).tolist()

    if exact_components is not None:
        before = len(smiles_list)
        smiles_list = [s for s in smiles_list if is_exact_component_count(s, exact_components)]
        print(f"[Initial filter] exact_components={exact_components}: {before} -> {len(smiles_list)}")

    seen = set()
    out = []
    for s in smiles_list:
        s = str(s).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)

    return out


def save_elites_to_csv(elites, fitness_scores, generation_idx, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"generation_{generation_idx + 1}_elites.csv")

    with open(output_file, "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["Generation", "SMILES", "SIS", "Num_Components"])
        for s in elites:
            csvwriter.writerow([
                generation_idx + 1,
                s,
                fitness_scores.get(s, None),
                count_components(s),
            ])

    print(f"Generation {generation_idx + 1} elites saved to file: {output_file}")


def format_template(template: str, k: int) -> str:
    return str(template).format(k=k)


def resolve_fragment_path(args, k: int, comp_dir: str) -> str:
    candidates = []

    if args.fragment_template:
        candidates.append(format_template(args.fragment_template, k))

    candidates.extend([
        os.path.join(comp_dir, "data_fragments.csv"),
        os.path.join(comp_dir, f"data_fragments_{k}.csv"),
        f"data_fragments_{k}.csv",
        f"data_fragments_components_{k}.csv",
    ])

    for path in candidates:
        if path and os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"No fragment library found for component count {k}. Tried:\n  " + "\n  ".join(candidates)
    )


def run_component_ga(args, k: int, ir_spectra):
    comp_dir = format_template(args.component_dir_template, k)
    input_smiles_path = os.path.join(comp_dir, format_template(args.input_filename_template, k))

    if not os.path.isdir(comp_dir):
        raise FileNotFoundError(f"Component directory does not exist: {comp_dir}")
    if not os.path.exists(input_smiles_path):
        raise FileNotFoundError(f"Initial candidate file does not exist: {input_smiles_path}")

    fragment_path = resolve_fragment_path(args, k, comp_dir)
    frag_out = os.path.join(comp_dir, args.frag_out)

    print("\n" + "=" * 80)
    print(f"[Component {k}] Start GA")
    print(f"[Component {k}] component dir : {comp_dir}")
    print(f"[Component {k}] input smiles  : {input_smiles_path}")
    print(f"[Component {k}] fragments     : {fragment_path}")
    print(f"[Component {k}] fragment out  : {frag_out}")
    print("=" * 80)

    fragment_fitness_df = calculate_and_save_fragment_fitness(
        input_file=fragment_path,
        output_file=frag_out,
        ir_spectra=ir_spectra,
        sigma=args.sigma,
    )

    smiles_list = read_smiles_with_sis(
        input_smiles_path,
        threshold=args.threshold,
        smiles_col=args.smiles_col,
        sis_col=args.sis_col,
        exact_components=k,
        top_n=args.initial_top_n,
    )

    if args.initial_top_n is not None and args.initial_top_n > 0:
        print(f"[Component {k}] Initial SMILES: {len(smiles_list)} candidates selected by SIS top-{args.initial_top_n} ranking")
    else:
        print(f"[Component {k}] Initial SMILES: {len(smiles_list)} candidates selected from all rows sorted by SIS")

    if len(smiles_list) == 0:
        print(f"[Component {k}] No available initial population; skipping.")
        with open(os.path.join(comp_dir, "optimized_smiles.txt"), "w") as f:
            f.write("Best SMILES: None\nBest SIS: -inf\nReason: empty_initial_population\n")
        return {
            "component_count": k,
            "best_smiles": None,
            "best_sis": -np.inf,
            "num_initial": 0,
            "status": "empty_initial_population",
        }

    max_components = k
    treat_single_as_invalid = bool(k > 1)

    print(
        f"[Component {k}] Validity settings: exact_components={k}, "
        f"max_components={max_components}, "
        f"treat_single_as_invalid={treat_single_as_invalid}"
    )

    best_smiles = None
    best_sis = -np.inf

    for gen in range(args.generations):
        print(f"\n[Component {k}] Starting generation {gen + 1} ...")

        smiles_list, fitness_scores = genetic_operations_with_population_update(
            smiles_list=smiles_list,
            fitness_fn=lambda s: calculate_fitness(
                s,
                ir_spectra,
                sigma=args.sigma,
                exact_components=k,
            ),
            fragment_fitness_df=fragment_fitness_df,
            num_to_generate=args.num_generate,
            crossover_rate=args.crossover_rate,
            mutation_rate=args.mutation_rate,
            elite_fraction=args.elite_fraction,
            treat_single_as_invalid=treat_single_as_invalid,
            max_components=max_components,
        )

        exact_population = [s for s in smiles_list if is_exact_component_count(s, k)]
        if len(exact_population) == 0:
            print(f"[Component {k}] No exact {k}-component candidates after generation {gen + 1}; keeping the current population.")
        else:
            smiles_list = exact_population

        exact_fitness = {
            s: fitness_scores.get(s, calculate_fitness(s, ir_spectra, sigma=args.sigma, exact_components=k))
            for s in smiles_list
            if is_exact_component_count(s, k)
        }

        n_elite = max(1, int(max(len(exact_fitness), 1) * args.elite_fraction))
        elites = sorted(exact_fitness, key=exact_fitness.get, reverse=True)[:n_elite]
        save_elites_to_csv(elites, exact_fitness, gen, output_dir=comp_dir)

        for s, cur in exact_fitness.items():
            if cur > best_sis:
                best_sis = cur
                best_smiles = s

        print(f"[Component {k}] Best SIS in generation {gen + 1}: {best_sis:.6f}; SMILES: {best_smiles}")

    optimized_path = os.path.join(comp_dir, "optimized_smiles.txt")
    with open(optimized_path, "w") as f:
        f.write(f"Best SMILES: {best_smiles}\n")
        f.write(f"Best SIS: {best_sis}\n")
        f.write(f"Num_Components: {k}\n")

    print(f"[Component {k}] Best result written to {optimized_path}")

    return {
        "component_count": k,
        "best_smiles": best_smiles,
        "best_sis": float(best_sis),
        "num_initial": len(smiles_list),
        "status": "ok",
    }


def parse_component_counts(text: str):
    out = []
    for x in str(text).split(","):
        x = x.strip()
        if not x:
            continue
        k = int(x)
        if k not in (1, 2, 3):
            raise ValueError("--component-counts supports only 1,2,3")
        if k not in out:
            out.append(k)
    return out


def parse_args():
    ap = argparse.ArgumentParser(
        description="Component-wise GA/BRICS with separated 1/2/3-component inputs and fragment libraries."
    )

    ap.add_argument("--checkpoint", default="../../../best_model.pth",
                    help="Graphormer-IR-inspired checkpoint")
    ap.add_argument("--local-pytorch-dir", default="local_pytorch",
                    help="Directory containing graphormer_ir_local.py")
    ap.add_argument("--ir", default="IR_data.csv",
                    help="IR CSV; expected to contain smiles, phase, and 400..4000 or sp_* spectral columns")
    ap.add_argument("--phase-col", default="phase", help="Phase column name in the IR CSV")
    ap.add_argument("--target-phase", default=None,
                    help="Manually specify the target phase; if omitted, the phase is read from the phase column in --ir")

    ap.add_argument("--component-counts", default="1,2,3",
                    help="Component counts to run separately, e.g., 1,2,3 or 2 or 2,3")
    ap.add_argument("--component-dir-template", default="components_{k}",
                    help="Component directory template; {k} is replaced by the component count. Default: components_{k}")
    ap.add_argument("--input-filename-template", default="smiles_sis_predictions.csv",
                    help="Initial candidate filename inside each component directory. Default: smiles_sis_predictions.csv")
    ap.add_argument("--smiles-col", default="SMILES",
                    help="SMILES or mixture column name in the initial candidate file")
    ap.add_argument("--sis-col", default="SIS",
                    help="SIS column name in the initial candidate file")
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="Accepted parameter; initial selection is based on SIS top-N ranking")
    ap.add_argument("--initial-top-n", type=int, default=1000,
                    help="Number of initial candidates read from each components_k/smiles_sis_predictions.csv; default is top 1000 by SIS; <=0 means using all candidates")

    ap.add_argument("--fragment-template", default=None,
                    help=(
                        "Fragment-library path template; {k} is replaced by the component count. For example: "
                        "'data_fragments_{k}.csv' or 'components_{k}/data_fragments.csv'. "
                        "If omitted, paths such as components_k/data_fragments.csv are tried automatically."
                    ))
    ap.add_argument("--frag-out", default="fragment_fitness.csv",
                    help="Fragment-fitness output filename inside each components_k directory")

    ap.add_argument("--num-generate", type=int, default=1000,
                    help="Number of molecules to generate per generation")
    ap.add_argument("--generations", type=int, default=20,
                    help="Number of GA generations")
    ap.add_argument("--crossover-rate", type=float, default=1.0,
                    help="Crossover probability")
    ap.add_argument("--mutation-rate", type=float, default=0.02,
                    help="Mutation probability")
    ap.add_argument("--elite-fraction", type=float, default=0.2,
                    help="Elite fraction")
    ap.add_argument("--sigma", type=int, default=10,
                    help="Gaussian smoothing sigma for SIS")
    ap.add_argument("--device", default=None,
                    help="Device: cuda / cuda:0 / cpu")
    ap.add_argument("--seed", type=int, default=None,
                    help="Random seed")

    ap.add_argument("--summary-out", default="componentwise_ga_summary.csv",
                    help="Summary output CSV")

    return ap.parse_args()


def main():
    global graphormer_model, model_meta, device, target_phase

    args = parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    if args.target_phase is not None:
        target_phase = args.target_phase
    else:
        target_phase = ir_read_phase_from_ir_csv(
            args.ir,
            phase_col=args.phase_col,
            default_phase="KBr",
        )
    print(f"[ir_sis] Target phase: {target_phase}")

    _dev = None if args.device is None else args.device

    try:
        graphormer_model, model_meta, device = ir_load_models(
            args.checkpoint,
            device=_dev,
            local_pytorch_dir=args.local_pytorch_dir,
        )
    except TypeError:
        graphormer_model, model_meta, device = ir_load_models(
            args.checkpoint,
            device=_dev,
        )
    print(f"[ir_sis] Using device: {device}")

    ir_spectra = ir_load_ir_matrix(args.ir)
    print(f"[ir_sis] Loaded target IR matrix: {ir_spectra.shape}")

    component_counts = parse_component_counts(args.component_counts)

    summary = []
    for k in component_counts:
        result = run_component_ga(args, k, ir_spectra)
        summary.append(result)

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.summary_out, index=False)
    print("=" * 80)
    print(f"[SAVE] Summary: {args.summary_out}")
    print(summary_df)
    print("=" * 80)


if __name__ == "__main__":
    main()
