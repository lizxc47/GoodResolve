#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# GoodResolve mixture Stage 1 -> Stage 2 -> Stage 3 pipeline
#
# This script is intended for mixture prediction.
# The script is expected to be placed in the project root as:
#   ./run_stage123_prediction_mixture.sh
#
# Expected project tree:
#   .
#   ├── run_stage123_prediction_mixture.sh
#   ├── mixture/
#   │   ├── stage1/
#   │   ├── stage2/
#   │   ├── stage3/
#   │   ├── test_data/
#   │   ├── mixture_by_fragment_count/
#   │   └── ternary_frag3/
#   ├── stage2/
#   │   ├── all_prediction_GA_small_molecular.py
#   │   ├── data_fragments.csv
#   │   ├── local_pytorch/
#   │   └── checkpoints_sdbs_graphormer_ir_1minusT/
#   ├── environment.yml
#   ├── requirements.txt
#   ├── README.md
#   └── LICENSE
#
# Execution logic:
#   data_i:
#       Stage 1 generates only data_i.
#       component_1: single-component Stage 2 -> Stage 3
#       component_2: mixture Stage 2 -> Stage 3
#       component_3: mixture Stage 2 -> Stage 3
#
# Notes:
#   component_1 uses the single-component Stage 2 logic under ./stage2.
#   component_2 and component_3 use the mixture Stage 2 logic under ./mixture/stage2.
# ============================================================


# ============================================================
# 0. Global switches
# ============================================================

RUN_STAGE1=true
RUN_STAGE2=true
RUN_STAGE3=true


# ============================================================
# 1. Basic paths
# ============================================================

PROJECT_ROOT="$(pwd)"
ROOT="${PROJECT_ROOT}/mixture"

STAGE1_DIR="${ROOT}/stage1"
STAGE2_MIXTURE_DIR="${ROOT}/stage2"
STAGE3_DIR="${ROOT}/stage3"

# Single-component Stage 2 directory
STAGE2_SINGLE_DIR="${PROJECT_ROOT}/stage2"

STAGE1_SCRIPT="${STAGE1_DIR}/all_prediction_graphormer_rnn_sdbs_mixture_weighted_irsis.py"

# Single-component GA reconstruction script
STAGE2_GA_SCRIPT_SINGLE="${STAGE2_SINGLE_DIR}/all_prediction_GA_small_molecular.py"

# Binary/ternary mixture GA reconstruction script
STAGE2_GA_SCRIPT_MIXTURE="${STAGE2_MIXTURE_DIR}/all_prediction_GA_small_molecular.py"

STAGE3_SCRIPT="${STAGE3_DIR}/stage3_final_rerank.py"

# Input data
IR_CSV="${ROOT}/test_data/binary_frag2_with_sis.csv"

# Stage 1 output directory; Stage 2/3 read data_<row_index> folders from here
OUT_ROOT="${PROJECT_ROOT}/result/mixture/"


# ============================================================
# 2. Stage 1 models and paths
# ============================================================

STAGE1_GRAPHORMER_CHECKPOINT="${STAGE1_DIR}/checkpoints_sdbs_graphormer_ir_1minusT/best_model.pt"
STAGE1_RNN_CHECKPOINT="${STAGE1_DIR}/smiles_rnn_token_checkpoint.pth"
STAGE1_IR_SIS_PATH="${STAGE1_DIR}/ir_sis.py"
STAGE1_LOCAL_PYTORCH_DIR="${STAGE1_DIR}/local_pytorch"


# ============================================================
# 3. Stage 2 models, local_pytorch, and fragment libraries
# ============================================================

# ---------- component_1: single-component logic ----------
STAGE2_SINGLE_GRAPHORMER_CHECKPOINT="${STAGE2_SINGLE_DIR}/checkpoints_sdbs_graphormer_ir_1minusT/best_model.pt"
STAGE2_SINGLE_LOCAL_PYTORCH_DIR="${STAGE2_SINGLE_DIR}/local_pytorch"
FRAG_LIB_1="${STAGE2_SINGLE_DIR}/data_fragments.csv"

# ---------- component_2/3: mixture logic ----------
STAGE2_MIXTURE_GRAPHORMER_CHECKPOINT="${STAGE2_MIXTURE_DIR}/checkpoints_sdbs_graphormer_ir_1minusT/best_model.pt"
STAGE2_MIXTURE_LOCAL_PYTORCH_DIR="${STAGE2_MIXTURE_DIR}/local_pytorch"

FRAG_DIR="${ROOT}/mixture_by_fragment_count/fragment_libraries_per_file"

# Fragment-count group. The default here is frag3.
# To run another fragment-count group, modify the following variables together:
#   IR_CSV
#   OUT_ROOT
#   FRAG_COUNT
FRAG_COUNT=2

FRAG_LIB_2="${FRAG_DIR}/binary_frag${FRAG_COUNT}_with_sis_data_fragments.csv"
FRAG_LIB_3="${FRAG_DIR}/ternary_frag${FRAG_COUNT}_with_sis_data_fragments.csv"


# ============================================================
# 4. Stage 3 models and parameters
# ============================================================

MODEL3_CHECKPOINT="${STAGE3_DIR}/model3_stage3_runs/phase_aware_gnn_400_4000/best_model3_gnn.pt"
SIGNATURE_VOCAB="${STAGE3_DIR}/model3_stage3_data/signature_vocab.csv"

# True-structure column in IR_data.csv.
# The current IR_CSV uses the Mixture column as the true-structure column.
TRUE_SMILES_COL="Mixture"

CANDIDATE_SMILES_COL="SMILES"
SIS_COL="SIS"

TOP_K_STAGE3=200
FEATURE="heavy_asc"

W_SIS=0.15
W_MODEL3=0.15
W_FEATURE=0.70

RERANK_BATCH_SIZE=256
DEFAULT_PHASE_STAGE3="unknown"


# ============================================================
# 5. Input column names and phase settings
# ============================================================

# True structure / mixture column in the input CSV
SMILES_COL="Mixture"

# To force DEFAULT_PHASE_STAGE1 instead of reading the phase column, set:
# PHASE_COL=""
PHASE_COL="phase"
DEFAULT_PHASE_STAGE1="KBr"


# ============================================================
# 6. Target selection
# ============================================================

# Specify row indices manually, separated by commas, for example:
# TARGET_INDICES="0,5,10"
# Leave empty to use START / LIMIT / NUM_TARGETS.
TARGET_INDICES=""

START=0

# Candidate range length starting from START.
# For example, LIMIT="10" means the candidate range contains 10 rows.
# Empty string means from START to the end of the file.
LIMIT=""

# When TARGET_INDICES is empty:
#   NUM_TARGETS=0 means running the full START/LIMIT range;
#   NUM_TARGETS>0 means randomly sampling NUM_TARGETS rows from the START/LIMIT range.
NUM_TARGETS=0


# ============================================================
# 7. Stage 1 generation parameters
# ============================================================

COMPONENT_COUNTS="1,2,3"

NUM_GENERATE=10000
NUM_MIXTURE_CANDIDATES=10000

# Number of single-component candidates; empty means using NUM_MIXTURE_CANDIDATES
NUM_SINGLE_CANDIDATES=""

ROUNDS=1
TEMPERATURE=1.0
TOP_K=30
MAX_ATTEMPT_FACTOR=20

GENERATE_VALID_ONLY=true
GENERATE_UNIQUE_CANONICAL=true

STAGE1_PREDICT_BATCH_SIZE=64
NUM_WORKERS=2
STAGE1_SIGMA=10
NO_SAVE_PREDICTED_SPECTRA=true


# ============================================================
# 8. Component switches
# ============================================================

RUN_COMPONENT_1=true
RUN_COMPONENT_2=true
RUN_COMPONENT_3=true


# ============================================================
# 9. Stage 2 GA parameters
# ============================================================

GA_NUM_GENERATE=1000
GA_GENERATIONS=5
INITIAL_TOP_N=1000

THRESHOLD=0.60
CROSSOVER_RATE=1.0
MUTATION_RATE=0.02
ELITE_FRACTION=0.2
STAGE2_SIGMA=10

# Whether to skip Stage 2 for this component if generation_${GA_GENERATIONS}_elites.csv already exists
SKIP_STAGE2_EXISTING=false

# Whether to skip Stage 3 for this component if the corresponding Stage 3 output already exists
SKIP_STAGE3_EXISTING=false

# Whether to stop the entire run when a sample or component fails
STOP_ON_ERROR=false

# Whether to skip a component if components_k/smiles_sis_predictions.csv is missing
SKIP_MISSING=true


# ============================================================
# 10. Device and runtime settings
# ============================================================

PYTHON_EXE="python"

STAGE1_GPU=0
STAGE2_GPU=0
STAGE3_GPU=0

DEVICE="cuda"
SEED=42
USE_AMP=true


# ============================================================
# 11. Log directories
# ============================================================

LOG_ROOT="${OUT_ROOT}/logs_stage123_one_by_one_componentwise"
STAGE1_LOG_DIR="${LOG_ROOT}/stage1"
STAGE2_LOG_DIR="${LOG_ROOT}/stage2"
STAGE3_LOG_DIR="${LOG_ROOT}/stage3"

mkdir -p "${LOG_ROOT}"
mkdir -p "${STAGE1_LOG_DIR}"
mkdir -p "${STAGE2_LOG_DIR}"
mkdir -p "${STAGE3_LOG_DIR}"


# ============================================================
# 12. Utility functions
# ============================================================

check_file() {
  local f="$1"
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] File not found: $f"
    exit 1
  fi
}

check_dir() {
  local d="$1"
  if [[ ! -d "$d" ]]; then
    echo "[ERROR] Directory not found: $d"
    exit 1
  fi
}

print_header() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}

get_fragment_lib() {
  local k="$1"

  if [[ "$k" == "1" ]]; then
    echo "${FRAG_LIB_1}"
  elif [[ "$k" == "2" ]]; then
    echo "${FRAG_LIB_2}"
  elif [[ "$k" == "3" ]]; then
    echo "${FRAG_LIB_3}"
  else
    echo "[ERROR] Unsupported component count: $k" >&2
    exit 1
  fi
}

component_enabled() {
  local k="$1"

  if [[ "$k" == "1" && "${RUN_COMPONENT_1}" == "true" ]]; then
    return 0
  fi
  if [[ "$k" == "2" && "${RUN_COMPONENT_2}" == "true" ]]; then
    return 0
  fi
  if [[ "$k" == "3" && "${RUN_COMPONENT_3}" == "true" ]]; then
    return 0
  fi

  return 1
}

build_target_indices() {
  if [[ -n "${TARGET_INDICES}" ]]; then
    echo "${TARGET_INDICES}" | tr ',' ' '
    return 0
  fi

  "${PYTHON_EXE}" - <<PY
import pandas as pd
import random

csv = "${IR_CSV}"
start = int("${START}")
limit_text = "${LIMIT}"
num_targets = int("${NUM_TARGETS}")
seed = int("${SEED}")

df = pd.read_csv(csv)
n = len(df)

if start < 0:
    start = 0
if start >= n:
    raise SystemExit(f"START={start} out of range; dataset rows={n}")

end = n if limit_text.strip() == "" else min(n, start + int(limit_text))
pool = list(range(start, end))

if num_targets <= 0 or num_targets >= len(pool):
    selected = pool
else:
    rng = random.Random(seed)
    selected = rng.sample(pool, num_targets)

print(" ".join(map(str, selected)))
PY
}


# ============================================================
# 13. Preflight checks
# ============================================================

print_header "Preflight checks"

check_file "${IR_CSV}"

check_file "${STAGE1_SCRIPT}"
check_file "${STAGE1_GRAPHORMER_CHECKPOINT}"
check_file "${STAGE1_RNN_CHECKPOINT}"
check_file "${STAGE1_IR_SIS_PATH}"
check_dir  "${STAGE1_LOCAL_PYTORCH_DIR}"

check_file "${STAGE2_GA_SCRIPT_SINGLE}"
check_file "${STAGE2_GA_SCRIPT_MIXTURE}"

check_file "${STAGE2_SINGLE_GRAPHORMER_CHECKPOINT}"
check_dir  "${STAGE2_SINGLE_LOCAL_PYTORCH_DIR}"

check_file "${STAGE2_MIXTURE_GRAPHORMER_CHECKPOINT}"
check_dir  "${STAGE2_MIXTURE_LOCAL_PYTORCH_DIR}"

check_file "${STAGE3_SCRIPT}"
check_file "${MODEL3_CHECKPOINT}"
check_file "${SIGNATURE_VOCAB}"

if [[ "${RUN_COMPONENT_1}" == "true" ]]; then
  check_file "${FRAG_LIB_1}"
fi

if [[ "${RUN_COMPONENT_2}" == "true" ]]; then
  check_file "${FRAG_LIB_2}"
fi

if [[ "${RUN_COMPONENT_3}" == "true" ]]; then
  check_file "${FRAG_LIB_3}"
fi

mkdir -p "${OUT_ROOT}"


# ============================================================
# 14. Stage 1 for one target
# ============================================================

run_stage1_one_target() {
  local target_idx="$1"
  local log_file="${STAGE1_LOG_DIR}/data_${target_idx}.log"

  print_header "Stage 1 | target data_${target_idx}"

  STAGE1_CMD=(
    "${PYTHON_EXE}" "${STAGE1_SCRIPT}"
    --ir-csv "${IR_CSV}"
    --smiles-col "${SMILES_COL}"
    --default-phase "${DEFAULT_PHASE_STAGE1}"
    --graphormer-checkpoint "${STAGE1_GRAPHORMER_CHECKPOINT}"
    --rnn-checkpoint "${STAGE1_RNN_CHECKPOINT}"
    --ir-sis-path "${STAGE1_IR_SIS_PATH}"
    --local-pytorch-dir "${STAGE1_LOCAL_PYTORCH_DIR}"
    --out-root "${OUT_ROOT}"
    --component-counts "${COMPONENT_COUNTS}"
    --target-indices "${target_idx}"
    --num-generate "${NUM_GENERATE}"
    --num-mixture-candidates "${NUM_MIXTURE_CANDIDATES}"
    --rounds "${ROUNDS}"
    --temperature "${TEMPERATURE}"
    --top-k "${TOP_K}"
    --max-attempt-factor "${MAX_ATTEMPT_FACTOR}"
    --predict-batch-size "${STAGE1_PREDICT_BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --sigma "${STAGE1_SIGMA}"
    --device "${DEVICE}"
    --seed "${SEED}"
  )

  if [[ -n "${PHASE_COL}" ]]; then
    STAGE1_CMD+=(--phase-col "${PHASE_COL}")
  fi

  if [[ -n "${NUM_SINGLE_CANDIDATES}" ]]; then
    STAGE1_CMD+=(--num-single-candidates "${NUM_SINGLE_CANDIDATES}")
  fi

  if [[ "${GENERATE_VALID_ONLY}" == "true" ]]; then
    STAGE1_CMD+=(--generate-valid-only)
  fi

  if [[ "${GENERATE_UNIQUE_CANONICAL}" == "true" ]]; then
    STAGE1_CMD+=(--generate-unique-canonical)
  fi

  if [[ "${USE_AMP}" == "true" ]]; then
    STAGE1_CMD+=(--amp)
  fi

  if [[ "${NO_SAVE_PREDICTED_SPECTRA}" == "true" ]]; then
    STAGE1_CMD+=(--no-save-predicted-spectra)
  fi

  echo "Target index    : ${target_idx}"
  echo "Output data dir : ${OUT_ROOT}/data_${target_idx}"
  echo "Log             : ${log_file}"
  echo "[CMD]"
  printf ' %q' "${STAGE1_CMD[@]}"
  echo

  set +e
  CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" "${STAGE1_CMD[@]}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] Stage 1 failed for data_${target_idx}, status=${status}"
    return "${status}"
  fi

  return 0
}


# ============================================================
# 15A. Stage 2 for component 1
# ============================================================

run_stage2_component1_single() {
  local data_dir="$1"

  local component_count="1"
  local data_name
  data_name="$(basename "${data_dir}")"

  local comp_dir="${data_dir}/components_1"

  local frag_out="fragment_fitness_component1.csv"
  local summary_out="componentwise_ga_summary_component1.csv"
  local final_elite="${comp_dir}/generation_${GA_GENERATIONS}_elites.csv"
  local log_file="${STAGE2_LOG_DIR}/${data_name}_component1.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "[Stage 2 SINGLE] ${data_name} | component 1"
  echo "Data dir    : ${data_dir}"
  echo "Comp dir    : ${comp_dir}"
  echo "GA script   : ${STAGE2_GA_SCRIPT_SINGLE}"
  echo "Checkpoint  : ${STAGE2_SINGLE_GRAPHORMER_CHECKPOINT}"
  echo "Local PT    : ${STAGE2_SINGLE_LOCAL_PYTORCH_DIR}"
  echo "Fragment lib: ${FRAG_LIB_1}"
  echo "Log         : ${log_file}"
  echo "------------------------------------------------------------"

  if [[ ! -f "${data_dir}/IR_data.csv" ]]; then
    echo "[WARN] Missing IR_data.csv: ${data_dir}/IR_data.csv"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  if [[ ! -f "${comp_dir}/smiles_sis_predictions.csv" ]]; then
    echo "[WARN] Missing initial candidates: ${comp_dir}/smiles_sis_predictions.csv"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  if [[ "${SKIP_STAGE2_EXISTING}" == "true" && -f "${final_elite}" ]]; then
    echo "[SKIP] Existing Stage2 final elite found: ${final_elite}"
    return 0
  fi

  # The single-component script usually uses the current directory as its working directory and reads IR_data.csv and smiles_sis_predictions.csv from there.
  # Therefore, copy the parent IR_data.csv into components_1 and run the script inside components_1.
  cp "${data_dir}/IR_data.csv" "${comp_dir}/IR_data.csv"

  cd "${comp_dir}"

  STAGE2_CMD=(
    "${PYTHON_EXE}" "${STAGE2_GA_SCRIPT_SINGLE}"
    --checkpoint "${STAGE2_SINGLE_GRAPHORMER_CHECKPOINT}"
    --local-pytorch-dir "${STAGE2_SINGLE_LOCAL_PYTORCH_DIR}"
    --ir IR_data.csv
    --phase-col phase
    --input-smiles smiles_sis_predictions.csv
    --threshold "${THRESHOLD}"
    --initial-top-n "${INITIAL_TOP_N}"
    --fragments "${FRAG_LIB_1}"
    --frag-out "${frag_out}"
    --num-generate "${GA_NUM_GENERATE}"
    --generations "${GA_GENERATIONS}"
    --crossover-rate "${CROSSOVER_RATE}"
    --mutation-rate "${MUTATION_RATE}"
    --elite-fraction "${ELITE_FRACTION}"
    --sigma "${STAGE2_SIGMA}"
    --device "${DEVICE}"
    --seed "${SEED}"
  )

  echo "[CMD]"
  printf ' %q' "${STAGE2_CMD[@]}"
  echo

  set +e
  CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" "${STAGE2_CMD[@]}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] ${data_name} component 1 Stage2 SINGLE failed with status ${status}"
    return "${status}"
  fi

  # Compatibility note: if the single-component script writes outputs to the current directory, they are already in comp_dir.
  # If it writes a differently named summary file, this script does not enforce it; only generation_${GA_GENERATIONS}_elites.csv is expected.
  if [[ ! -f "${final_elite}" ]]; then
    echo "[WARN] Expected final elite not found after single Stage2: ${final_elite}"
  fi

  echo "[DONE] ${data_name} component 1 Stage2 SINGLE"
  return 0
}


# ============================================================
# 15B. Stage 2 for component 2/3
# ============================================================

run_stage2_component_mixture() {
  local data_dir="$1"
  local component_count="$2"

  local data_name
  data_name="$(basename "${data_dir}")"

  local comp_dir="${data_dir}/components_${component_count}"

  local fragment_lib
  fragment_lib="$(get_fragment_lib "${component_count}")"

  local frag_out="fragment_fitness_component${component_count}.csv"
  local summary_out="componentwise_ga_summary_component${component_count}.csv"
  local final_elite="${comp_dir}/generation_${GA_GENERATIONS}_elites.csv"
  local log_file="${STAGE2_LOG_DIR}/${data_name}_component${component_count}.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "[Stage 2 MIXTURE] ${data_name} | component ${component_count}"
  echo "Data dir    : ${data_dir}"
  echo "Comp dir    : ${comp_dir}"
  echo "GA script   : ${STAGE2_GA_SCRIPT_MIXTURE}"
  echo "Checkpoint  : ${STAGE2_MIXTURE_GRAPHORMER_CHECKPOINT}"
  echo "Local PT    : ${STAGE2_MIXTURE_LOCAL_PYTORCH_DIR}"
  echo "Fragment lib: ${fragment_lib}"
  echo "Log         : ${log_file}"
  echo "------------------------------------------------------------"

  if [[ ! -f "${data_dir}/IR_data.csv" ]]; then
    echo "[WARN] Missing IR_data.csv: ${data_dir}/IR_data.csv"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  if [[ ! -f "${comp_dir}/smiles_sis_predictions.csv" ]]; then
    echo "[WARN] Missing initial candidates: ${comp_dir}/smiles_sis_predictions.csv"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  if [[ ! -f "${fragment_lib}" ]]; then
    echo "[ERROR] Missing fragment library: ${fragment_lib}"
    return 1
  fi

  if [[ "${SKIP_STAGE2_EXISTING}" == "true" && -f "${final_elite}" ]]; then
    echo "[SKIP] Existing Stage2 final elite found: ${final_elite}"
    return 0
  fi

  cd "${data_dir}"

  STAGE2_CMD=(
    "${PYTHON_EXE}" "${STAGE2_GA_SCRIPT_MIXTURE}"
    --checkpoint "${STAGE2_MIXTURE_GRAPHORMER_CHECKPOINT}"
    --local-pytorch-dir "${STAGE2_MIXTURE_LOCAL_PYTORCH_DIR}"
    --ir IR_data.csv
    --phase-col phase
    --component-counts "${component_count}"
    --component-dir-template "components_{k}"
    --input-filename-template smiles_sis_predictions.csv
    --smiles-col SMILES
    --sis-col SIS
    --threshold "${THRESHOLD}"
    --initial-top-n "${INITIAL_TOP_N}"
    --fragment-template "${fragment_lib}"
    --frag-out "${frag_out}"
    --num-generate "${GA_NUM_GENERATE}"
    --generations "${GA_GENERATIONS}"
    --crossover-rate "${CROSSOVER_RATE}"
    --mutation-rate "${MUTATION_RATE}"
    --elite-fraction "${ELITE_FRACTION}"
    --sigma "${STAGE2_SIGMA}"
    --device "${DEVICE}"
    --seed "${SEED}"
    --summary-out "${summary_out}"
  )

  echo "[CMD]"
  printf ' %q' "${STAGE2_CMD[@]}"
  echo

  set +e
  CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" "${STAGE2_CMD[@]}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] ${data_name} component ${component_count} Stage2 MIXTURE failed with status ${status}"
    return "${status}"
  fi

  echo "[DONE] ${data_name} component ${component_count} Stage2 MIXTURE"
  return 0
}


run_stage2_one_component() {
  local data_dir="$1"
  local component_count="$2"

  if [[ "${component_count}" == "1" ]]; then
    run_stage2_component1_single "${data_dir}"
  else
    run_stage2_component_mixture "${data_dir}" "${component_count}"
  fi
}


# ============================================================
# 16. Stage 3 for one component
# ============================================================

run_stage3_one_component() {
  local data_dir="$1"
  local component_count="$2"

  local data_name
  data_name="$(basename "${data_dir}")"

  local comp_dir="${data_dir}/components_${component_count}"
  local candidate_file="generation_${GA_GENERATIONS}_elites.csv"

  local output_file="stage3_final_component${component_count}.csv"
  local summary_file="stage3_final_component${component_count}_summary.csv"
  local log_file="${STAGE3_LOG_DIR}/${data_name}_component${component_count}.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "[Stage 3] ${data_name} | component ${component_count}"
  echo "Data dir      : ${data_dir}"
  echo "Comp dir      : ${comp_dir}"
  echo "Candidate file: ${candidate_file}"
  echo "Output file   : ${output_file}"
  echo "Summary file  : ${summary_file}"
  echo "Log           : ${log_file}"
  echo "------------------------------------------------------------"

  if [[ ! -d "${comp_dir}" ]]; then
    echo "[WARN] Missing component directory: ${comp_dir}"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  if [[ ! -f "${comp_dir}/${candidate_file}" ]]; then
    echo "[WARN] Missing Stage2 candidate file: ${comp_dir}/${candidate_file}"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  if [[ ! -f "${data_dir}/IR_data.csv" ]]; then
    echo "[WARN] Missing parent IR_data.csv: ${data_dir}/IR_data.csv"
    if [[ "${SKIP_MISSING}" == "true" ]]; then return 0; else return 1; fi
  fi

  # stage3_final_rerank.py is run with root=data_dir and pattern=components_k.
  # Therefore, copy the parent IR_data.csv into the components_k directory.
  cp "${data_dir}/IR_data.csv" "${comp_dir}/IR_data.csv"

  if [[ "${SKIP_STAGE3_EXISTING}" == "true" && -f "${comp_dir}/${output_file}" ]]; then
    echo "[SKIP] Existing Stage3 output found: ${comp_dir}/${output_file}"
    return 0
  fi

  STAGE3_CMD=(
    "${PYTHON_EXE}" "${STAGE3_SCRIPT}"
    --root "${data_dir}"
    --pattern "components_${component_count}"
    --checkpoint "${MODEL3_CHECKPOINT}"
    --signature-vocab "${SIGNATURE_VOCAB}"
    --candidate-filename "${candidate_file}"
    --ir-filename "IR_data.csv"
    --true-smiles-col "${TRUE_SMILES_COL}"
    --phase-col "${PHASE_COL:-phase}"
    --default-phase "${DEFAULT_PHASE_STAGE3}"
    --candidate-smiles-col "${CANDIDATE_SMILES_COL}"
    --sis-col "${SIS_COL}"
    --top-k "${TOP_K_STAGE3}"
    --feature "${FEATURE}"
    --w-sis "${W_SIS}"
    --w-model3 "${W_MODEL3}"
    --w-feature "${W_FEATURE}"
    --candidate-batch-size "${RERANK_BATCH_SIZE}"
    --device "${DEVICE}"
    --output-filename "${output_file}"
    --summary-filename "${summary_file}"
  )

  if [[ "${USE_AMP}" == "true" ]]; then
    STAGE3_CMD+=(--amp)
  fi

  echo "[CMD]"
  printf ' %q' "${STAGE3_CMD[@]}"
  echo

  set +e
  CUDA_VISIBLE_DEVICES="${STAGE3_GPU}" "${STAGE3_CMD[@]}" 2>&1 | tee "${log_file}"
  local status=${PIPESTATUS[0]}
  set -e

  if [[ "${status}" -ne 0 ]]; then
    echo "[ERROR] ${data_name} component ${component_count} Stage3 failed with status ${status}"
    return "${status}"
  fi

  echo "[DONE] ${data_name} component ${component_count} Stage3"
  return 0
}


# ============================================================
# 17. Stage 2 -> Stage 3 for one component
# ============================================================

run_stage23_one_component() {
  local data_dir="$1"
  local component_count="$2"

  local failed_local=0

  if [[ "${RUN_STAGE2}" == "true" ]]; then
    if run_stage2_one_component "${data_dir}" "${component_count}"; then
      :
    else
      failed_local=$((failed_local + 1))
      if [[ "${STOP_ON_ERROR}" == "true" ]]; then exit 1; fi
    fi
  else
    echo "[SKIP] Stage 2 disabled for $(basename "${data_dir}") component ${component_count}"
  fi

  if [[ "${RUN_STAGE3}" == "true" ]]; then
    if run_stage3_one_component "${data_dir}" "${component_count}"; then
      :
    else
      failed_local=$((failed_local + 1))
      if [[ "${STOP_ON_ERROR}" == "true" ]]; then exit 1; fi
    fi
  else
    echo "[SKIP] Stage 3 disabled for $(basename "${data_dir}") component ${component_count}"
  fi

  if [[ "${failed_local}" -eq 0 ]]; then
    return 0
  else
    return 1
  fi
}


# ============================================================
# 18. Main loop: run Stage 1 for one target, then Stage 2 -> Stage 3 component-wise
# ============================================================

print_header "Build target list"

TARGET_LIST="$(build_target_indices)"
echo "Targets:"
echo "  ${TARGET_LIST}"

success=0
failed=0

for target_idx in ${TARGET_LIST}; do
  print_header "Pipeline target data_${target_idx}"

  data_dir="${OUT_ROOT}/data_${target_idx}"

  if [[ "${RUN_STAGE1}" == "true" ]]; then
    if run_stage1_one_target "${target_idx}"; then
      :
    else
      failed=$((failed + 1))
      if [[ "${STOP_ON_ERROR}" == "true" ]]; then exit 1; fi
      continue
    fi
  else
    echo "[SKIP] Stage 1 disabled for data_${target_idx}"
  fi

  if [[ ! -d "${data_dir}" ]]; then
    echo "[ERROR] Stage 1 output data dir not found: ${data_dir}"
    failed=$((failed + 1))
    if [[ "${STOP_ON_ERROR}" == "true" ]]; then exit 1; fi
    continue
  fi

  sample_failed=0

  for component_count in 1 2 3; do
    if component_enabled "${component_count}"; then
      print_header "data_${target_idx} | component_${component_count}: Stage2 -> Stage3"
      if run_stage23_one_component "${data_dir}" "${component_count}"; then
        :
      else
        sample_failed=$((sample_failed + 1))
      fi
    else
      echo "[SKIP] data_${target_idx} component ${component_count} disabled"
    fi
  done

  if [[ "${sample_failed}" -eq 0 ]]; then
    success=$((success + 1))
  else
    failed=$((failed + 1))
  fi

done


# ============================================================
# 19. Finish
# ============================================================

print_header "One-by-one component-wise Stage 1 -> Stage 2 -> Stage 3 pipeline finished"

echo "Success targets: ${success}"
echo "Failed targets : ${failed}"

echo ""
echo "Output root:"
echo "  ${OUT_ROOT}"

echo ""
echo "Logs:"
echo "  ${LOG_ROOT}"

echo ""
echo "Data folders:"
find "${OUT_ROOT}" -maxdepth 1 -type d -name "data*" | sort -V
