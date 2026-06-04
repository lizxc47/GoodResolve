#!/usr/bin/env bash
set -euo pipefail

# NOTE:
#   This script is intended for pure-compound prediction only.
#   It is not the mixture-component pipeline.
#   For mixture prediction, use:
#       ./run_stage123_prediction_mixture.sh
#
# ============================================================
# GoodResolve per-sample Stage 1 -> Stage 2 -> Stage 3 pipeline
#
# Execution logic:
#   sample 0: Stage1 -> Stage2 -> Stage3
#   sample 1: Stage1 -> Stage2 -> Stage3
#   sample 2: Stage1 -> Stage2 -> Stage3
#   ...
#
# Directory structure:
#   .
#   ├── stage1/
#   ├── stage2/
#   │   └── all_prediction_GA_small_molecular.py
#   ├── stage3/
#   ├── test_data/test_data.csv
#   └── result/
#
# Output for each sample:
#   result/sample_<idx>/
#   ├── data_<idx>/
#   │   ├── IR_data.csv
#   │   ├── generated_smiles_round_1.csv
#   │   ├── smiles_sis_predictions.csv
#   │   ├── generation_20_elites.csv
#   │   └── stage3_final_*.csv
#   └── logs/
# ============================================================


# ============================================================
# 0. Global switches
# ============================================================

RUN_STAGE1=true
RUN_STAGE2=true
RUN_STAGE3=true

PYTHON_EXE="python"

STAGE1_GPU=0
STAGE2_GPU=0
STAGE3_GPU=0

DEVICE="cuda"
SEED=42


# ============================================================
# 1. Global directories
# ============================================================

ROOT_DIR="$(pwd)"

STAGE1_DIR="${ROOT_DIR}/stage1"
STAGE2_DIR="${ROOT_DIR}/stage2"
STAGE3_DIR="${ROOT_DIR}/stage3"

TEST_CSV="${ROOT_DIR}/test_data/test_data.csv"

RESULT_DIR="${ROOT_DIR}/result/pure_compounds"
mkdir -p "${RESULT_DIR}"


# ============================================================
# 2. Target sample selection
# ============================================================

# Option A: manually specify row indices, for example:
# TARGET_INDICES="0 1 2 5 10"
#
# Option B: leave empty to automatically select the first N samples according to NUM_TARGETS;
# NUM_TARGETS=0 means running the full test_data.csv.
TARGET_INDICES=""

# Effective only when TARGET_INDICES is empty.
# 0 means running the full test set.
NUM_TARGETS=1

# Start index. Used together with NUM_TARGETS.
START_INDEX=0


# ============================================================
# 3. Stage 1 parameters
# ============================================================

STAGE1_SCRIPT="${STAGE1_DIR}/all_prediction_graphormer_rnn_sdbs.py"

STAGE1_GRAPHORMER_CHECKPOINT="${STAGE1_DIR}/checkpoints_sdbs_graphormer_ir_1minusT/best_model.pt"
STAGE1_RNN_CHECKPOINT="${STAGE1_DIR}/smiles_rnn_token_checkpoint.pth"
STAGE1_LOCAL_PYTORCH_DIR="${STAGE1_DIR}/local_pytorch"

NUM_GENERATE=10000
ROUNDS=1
TEMPERATURE=1.0
TOP_K=30
MAX_ATTEMPT_FACTOR=20

PREDICT_BATCH_SIZE=32
NUM_WORKERS=2
SIS_GAUSSIAN_STD_CM=15.0

NO_SAVE_PREDICTED_SPECTRA=true


# ============================================================
# 4. Stage 2 parameters
# ============================================================

STAGE2_GA_SCRIPT="${STAGE2_DIR}/all_prediction_GA_small_molecular.py"

STAGE2_GRAPHORMER_CHECKPOINT="${STAGE2_DIR}/checkpoints_sdbs_graphormer_ir_1minusT/best_model.pt"
STAGE2_LOCAL_PYTORCH_DIR="${STAGE2_DIR}/local_pytorch"
STAGE2_FRAGMENTS="${STAGE2_DIR}/data_fragments.csv"

IR_FILE="IR_data.csv"
INPUT_SMILES="smiles_sis_predictions.csv"

PHASE_COL="phase"
TARGET_PHASE=""

THRESHOLD=0.60
INITIAL_TOP_N=1000

GA_NUM_GENERATE=1000
GA_GENERATIONS=5
CROSSOVER_RATE=1.0
MUTATION_RATE=0.02
ELITE_FRACTION=0.2

SIGMA=10
ALLOW_MIXTURES=false

SKIP_EXISTING=false
STOP_ON_ERROR=true


# ============================================================
# 5. Stage 3 parameters
# ============================================================

STAGE3_SCRIPT="${STAGE3_DIR}/stage3_final_rerank.py"

MODEL3_CHECKPOINT="${STAGE3_DIR}/model3_stage3_runs/phase_aware_gnn_400_4000/best_model3_gnn.pt"
SIGNATURE_VOCAB="${STAGE3_DIR}/model3_stage3_data/signature_vocab.csv"

TRUE_SMILES_COL="smiles"
CANDIDATE_SMILES_COL="SMILES"
SIS_COL="SIS"

TOP_K_STAGE3=200
FEATURE="heavy_asc"

W_SIS=0.15
W_MODEL3=0.15
W_FEATURE=0.70

RERANK_BATCH_SIZE=256
DEFAULT_PHASE="unknown"

USE_AMP=true


# ============================================================
# 6. Utility functions
# ============================================================

check_file() {
  local f="$1"
  if [[ ! -f "${f}" ]]; then
    echo "[ERROR] File not found: ${f}"
    exit 1
  fi
}

check_dir() {
  local d="$1"
  if [[ ! -d "${d}" ]]; then
    echo "[ERROR] Directory not found: ${d}"
    exit 1
  fi
}

print_header() {
  echo ""
  echo "============================================================"
  echo "$1"
  echo "============================================================"
}


# ============================================================
# 7. Check required files
# ============================================================

check_file "${TEST_CSV}"

check_file "${STAGE1_SCRIPT}"
check_file "${STAGE1_GRAPHORMER_CHECKPOINT}"
check_file "${STAGE1_RNN_CHECKPOINT}"
check_dir  "${STAGE1_LOCAL_PYTORCH_DIR}"

check_file "${STAGE2_GA_SCRIPT}"
check_file "${STAGE2_GRAPHORMER_CHECKPOINT}"
check_dir  "${STAGE2_LOCAL_PYTORCH_DIR}"
check_file "${STAGE2_FRAGMENTS}"

check_file "${STAGE3_SCRIPT}"
check_file "${MODEL3_CHECKPOINT}"
check_file "${SIGNATURE_VOCAB}"


# ============================================================
# 8. Build the target sample list
# ============================================================

if [[ -z "${TARGET_INDICES}" ]]; then
  N_ROWS=$("${PYTHON_EXE}" - <<PY
import pandas as pd
df = pd.read_csv("${TEST_CSV}")
print(len(df))
PY
)

  if [[ "${NUM_TARGETS}" -le 0 ]]; then
    END_INDEX="${N_ROWS}"
  else
    END_INDEX=$((START_INDEX + NUM_TARGETS))
    if [[ "${END_INDEX}" -gt "${N_ROWS}" ]]; then
      END_INDEX="${N_ROWS}"
    fi
  fi

  TARGET_INDICES=$(seq "${START_INDEX}" $((END_INDEX - 1)))
fi

echo "============================================================"
echo "GoodResolve per-sample Stage123 pipeline"
echo "TEST_CSV       = ${TEST_CSV}"
echo "RESULT_DIR     = ${RESULT_DIR}"
echo "TARGET_INDICES = ${TARGET_INDICES}"
echo "RUN_STAGE1     = ${RUN_STAGE1}"
echo "RUN_STAGE2     = ${RUN_STAGE2}"
echo "RUN_STAGE3     = ${RUN_STAGE3}"
echo "============================================================"


# ============================================================
# 9. Run Stage 1 -> Stage 2 -> Stage 3 sample by sample
# ============================================================

for IDX in ${TARGET_INDICES}; do
  SAMPLE_DIR="${RESULT_DIR}/sample_${IDX}"
  SAMPLE_LOG_DIR="${SAMPLE_DIR}/logs"

  mkdir -p "${SAMPLE_DIR}"
  mkdir -p "${SAMPLE_LOG_DIR}"

  STAGE1_LOG="${SAMPLE_LOG_DIR}/stage1.log"
  STAGE2_LOG="${SAMPLE_LOG_DIR}/stage2.log"
  STAGE3_LOG="${SAMPLE_LOG_DIR}/stage3.log"

  print_header "Sample ${IDX}: Stage 1 -> Stage 2 -> Stage 3"

  # ------------------------------------------------------------
  # Stage 1
  # ------------------------------------------------------------
  if [[ "${RUN_STAGE1}" == "true" ]]; then
    print_header "Sample ${IDX} | Stage 1"

    STAGE1_CMD=(
      "${PYTHON_EXE}" "${STAGE1_SCRIPT}"
      --ir-csv "${TEST_CSV}"
      --graphormer-checkpoint "${STAGE1_GRAPHORMER_CHECKPOINT}"
      --rnn-checkpoint "${STAGE1_RNN_CHECKPOINT}"
      --local-pytorch-dir "${STAGE1_LOCAL_PYTORCH_DIR}"
      --out-root "${SAMPLE_DIR}"
      --target-indices "${IDX}"
      --num-generate "${NUM_GENERATE}"
      --rounds "${ROUNDS}"
      --temperature "${TEMPERATURE}"
      --top-k "${TOP_K}"
      --max-attempt-factor "${MAX_ATTEMPT_FACTOR}"
      --generate-valid-only
      --generate-unique-canonical
      --predict-batch-size "${PREDICT_BATCH_SIZE}"
      --num-workers "${NUM_WORKERS}"
      --sis-gaussian-std-cm "${SIS_GAUSSIAN_STD_CM}"
      --device "${DEVICE}"
      --seed "${SEED}"
    )

    if [[ "${USE_AMP}" == "true" ]]; then
      STAGE1_CMD+=(--amp)
    fi

    if [[ "${NO_SAVE_PREDICTED_SPECTRA}" == "true" ]]; then
      STAGE1_CMD+=(--no-save-predicted-spectra)
    fi

    echo "[CMD]"
    printf ' %q' "${STAGE1_CMD[@]}"
    echo

    CUDA_VISIBLE_DEVICES="${STAGE1_GPU}" "${STAGE1_CMD[@]}" 2>&1 | tee "${STAGE1_LOG}"
  fi

  DATA_DIR="${SAMPLE_DIR}/data_${IDX}"

  if [[ ! -d "${DATA_DIR}" ]]; then
    echo "[ERROR] Stage 1 output not found: ${DATA_DIR}"
    exit 1
  fi

  # ------------------------------------------------------------
  # Stage 2
  # ------------------------------------------------------------
  if [[ "${RUN_STAGE2}" == "true" ]]; then
    print_header "Sample ${IDX} | Stage 2"

    if [[ "${SKIP_EXISTING}" == "true" && -f "${DATA_DIR}/optimized_smiles.txt" ]]; then
      echo "[SKIP] Existing Stage 2 output found: ${DATA_DIR}/optimized_smiles.txt"
    else
      STAGE2_CMD=(
        "${PYTHON_EXE}" "${STAGE2_GA_SCRIPT}"
        --checkpoint "${STAGE2_GRAPHORMER_CHECKPOINT}"
        --local-pytorch-dir "${STAGE2_LOCAL_PYTORCH_DIR}"

        --ir "${IR_FILE}"
        --phase-col "${PHASE_COL}"

        --input-smiles "${INPUT_SMILES}"
        --threshold "${THRESHOLD}"
        --initial-top-n "${INITIAL_TOP_N}"

        --fragments "${STAGE2_FRAGMENTS}"
        --frag-out "fragment_fitness.csv"

        --num-generate "${GA_NUM_GENERATE}"
        --generations "${GA_GENERATIONS}"
        --crossover-rate "${CROSSOVER_RATE}"
        --mutation-rate "${MUTATION_RATE}"
        --elite-fraction "${ELITE_FRACTION}"

        --sigma "${SIGMA}"
        --device "${DEVICE}"
        --seed "${SEED}"
      )

      if [[ -n "${TARGET_PHASE}" ]]; then
        STAGE2_CMD+=(--target-phase "${TARGET_PHASE}")
      fi

      if [[ "${ALLOW_MIXTURES}" == "true" ]]; then
        STAGE2_CMD+=(--allow-mixtures)
      fi

      echo "[CMD]"
      printf ' %q' "${STAGE2_CMD[@]}"
      echo

      set +e
      (
        cd "${DATA_DIR}"
        CUDA_VISIBLE_DEVICES="${STAGE2_GPU}" "${STAGE2_CMD[@]}"
      ) 2>&1 | tee "${STAGE2_LOG}"
      STAGE2_STATUS=${PIPESTATUS[0]}
      set -e

      if [[ "${STAGE2_STATUS}" -ne 0 ]]; then
        echo "[ERROR] Stage 2 failed for sample ${IDX}, status=${STAGE2_STATUS}"
        if [[ "${STOP_ON_ERROR}" == "true" ]]; then
          exit "${STAGE2_STATUS}"
        fi
        continue
      fi
    fi
  fi

  # ------------------------------------------------------------
  # Stage 3
  # ------------------------------------------------------------
  if [[ "${RUN_STAGE3}" == "true" ]]; then
    print_header "Sample ${IDX} | Stage 3"

    STAGE3_CMD=(
      "${PYTHON_EXE}" "${STAGE3_SCRIPT}"
      --root "${SAMPLE_DIR}"
      --checkpoint "${MODEL3_CHECKPOINT}"
      --signature-vocab "${SIGNATURE_VOCAB}"
      --ir-filename "${IR_FILE}"
      --true-smiles-col "${TRUE_SMILES_COL}"
      --phase-col "${PHASE_COL}"
      --default-phase "${DEFAULT_PHASE}"
      --candidate-smiles-col "${CANDIDATE_SMILES_COL}"
      --sis-col "${SIS_COL}"
      --top-k "${TOP_K_STAGE3}"
      --feature "${FEATURE}"
      --w-sis "${W_SIS}"
      --w-model3 "${W_MODEL3}"
      --w-feature "${W_FEATURE}"
      --candidate-batch-size "${RERANK_BATCH_SIZE}"
      --device "${DEVICE}"
    )

    if [[ "${USE_AMP}" == "true" ]]; then
      STAGE3_CMD+=(--amp)
    fi

    echo "[CMD]"
    printf ' %q' "${STAGE3_CMD[@]}"
    echo

    CUDA_VISIBLE_DEVICES="${STAGE3_GPU}" "${STAGE3_CMD[@]}" 2>&1 | tee "${STAGE3_LOG}"

    # Copy the summary files from each sample folder to the global result directory for easier aggregation.
    find "${SAMPLE_DIR}" -maxdepth 1 -type f -name "stage3_final_*_summary.csv" \
      -exec cp {} "${RESULT_DIR}/sample_${IDX}_{}" \; 2>/dev/null || true
  fi

  print_header "Sample ${IDX} finished"
  echo "Sample result:"
  echo "  ${SAMPLE_DIR}"
  echo "Data folder:"
  echo "  ${DATA_DIR}"
  echo "Logs:"
  echo "  ${SAMPLE_LOG_DIR}"

done


# ============================================================
# 10. Finish
# ============================================================

print_header "All selected samples finished"

echo "Result root:"
echo "  ${RESULT_DIR}"

echo ""
echo "Sample directories:"
find "${RESULT_DIR}" -maxdepth 1 -type d -name "sample_*" | sort

echo ""
echo "Done."
