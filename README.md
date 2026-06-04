# GoodResolve Stage 1–2–3 Prediction Pipeline

> **Confidential review package.** This repository is provided solely for confidential peer-review, editorial evaluation, and manuscript verification. It is not currently released as public open-source software, but is provided to reviewers for inspection, execution, testing, and verification of the associated manuscript. The associated methods are under pending patent application, including Chinese patent application No. CN202610751112.3. A separate public, research-use, or other license may be provided after completion of the review, publication, or patent-related process. See `LICENSE` for details.

GoodResolve is an infrared-spectrum-based molecular and mixture structure identification workflow. Two root-level execution scripts are provided:

```text
./run_stage123_prediction_pure_compounds.sh   # pure-compound prediction
./run_stage123_prediction_mixture.sh          # mixture prediction
```

Both scripts can be executed directly from the project root. By default, both pipelines start from row index 0 of their configured input CSV files.

The provided example configuration is intentionally lightweight. It is designed for quick execution and functional verification of the Stage 1–2–3 workflow, rather than for reproducing the full-scale production. For the provided mixture example, successful prediction can be completed in approximately 9 hours on our hardware using single-threaded CPU execution together with one GPU. The hardware used for this test was an Intel(R) Xeon(R) Gold 6136 CPU @ 3.00 GHz and an NVIDIA RTX 4090 GPU. In addition, we provide a smaller quick-test parameter setting to verify that the full pipeline can run successfully; under the same hardware configuration, this quick test can be completed in approximately 10 minutes. The quick-test setting is intended only to check whether the workflow executes correctly and is not expected to guarantee accurate prediction results.


---

## 1. Directory structure

The expected project layout is:

```text
./
├── README.md
├── LICENSE
├── REVIEWER_ONLY_NOTICE.md
├── environment.yml
├── requirements.txt
├── run_stage123_prediction_pure_compounds.sh
├── run_stage123_prediction_mixture.sh
│
├── stage1/
│   ├── all_prediction_graphormer_rnn_sdbs.py
│   ├── local_pytorch/
│   ├── checkpoints_sdbs_graphormer_ir_1minusT/
│   └── smiles_rnn_token_checkpoint.pth
│
├── stage2/
│   ├── all_prediction_GA_small_molecular.py
│   ├── data_fragments.csv
│   ├── local_pytorch/
│   └── checkpoints_sdbs_graphormer_ir_1minusT/
│
├── stage3/
│   ├── stage3_final_rerank.py
│   ├── model3_stage3_data/
│   └── model3_stage3_runs/
│
├── test_data/
│   └── test_data.csv
│
├── result/
│
└── mixture/
    ├── stage1/
    │   ├── all_prediction_graphormer_rnn_sdbs_mixture_weighted_irsis.py
    │   ├── ir_sis.py
    │   ├── local_pytorch/
    │   ├── checkpoints_sdbs_graphormer_ir_1minusT/
    │   └── smiles_rnn_token_checkpoint.pth
    │
    ├── stage2/
    │   ├── all_prediction_GA_small_molecular.py
    │   ├── local_pytorch/
    │   └── checkpoints_sdbs_graphormer_ir_1minusT/
    │
    ├── stage3/
    │   ├── stage3_final_rerank.py
    │   ├── model3_stage3_data/
    │   └── model3_stage3_runs/
    │
    ├── mixture_by_fragment_count/
    │   └── fragment_libraries_per_file/
    │       ├── binary_frag*_with_sis_data_fragments.csv
    │       └── ternary_frag*_with_sis_data_fragments.csv
    │
    ├── test_data/
    │   └── *_with_sis.csv
    │
    └── ternary_frag3/
        └── result_mixture/
```

The root-level `./stage1`, `./stage2`, and `./stage3` folders are used by the pure-compound prediction script. The `./mixture/stage1`, `./mixture/stage2`, and `./mixture/stage3` folders are used by the mixture prediction script. The root-level `./stage2` folder is also used by the mixture script for the single-component reconstruction logic.

---

## 2. Environment setup

The recommended installation method is `conda` with `environment.yml`:

```bash
conda env create -f environment.yml
conda activate goodresolve
```

The tested core environment is:

```text
Python        3.12.2
NumPy         1.26.4
Pandas        2.2.2
SciPy         1.13.1
tqdm          4.66.5
RDKit         2023.09.1
PyTorch       2.4.0
Torch CUDA    11.8
TorchVision   0.19.0
scikit-learn  1.5.1
Matplotlib    3.9.2
```

To check whether PyTorch can access CUDA:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

---

## 3. Input data

### Pure-compound prediction

The default pure-compound input file is:

```text
./test_data/test_data.csv
```

The input CSV should contain the molecular structure column and the target IR spectral columns, for example:

```text
smiles
phase
400, 402, 404, ..., 4000
```

### Mixture prediction

The default mixture input file is configured in:

```text
./run_stage123_prediction_mixture.sh
```

For example, the current mixture script may use:

```text
./mixture/test_data/ternary_frag3_with_sis.csv
```

The mixture input CSV should contain:

```text
Mixture
phase
400, 402, 404, ..., 4000
```

`Mixture` is the true mixture SMILES column. The mixture script uses `START=0` by default. With the default `LIMIT="10"` and `NUM_TARGETS=0`, it processes rows 0–9 unless these parameters are modified.

---

## 4. Main execution scripts

The execution scripts are located in the project root:

```text
./run_stage123_prediction_pure_compounds.sh
./run_stage123_prediction_mixture.sh
```

Before running, make sure both scripts have Unix line endings and execute permission:

```bash
sed -i 's/\r$//' run_stage123_prediction_pure_compounds.sh
sed -i 's/\r$//' run_stage123_prediction_mixture.sh

chmod +x run_stage123_prediction_pure_compounds.sh
chmod +x run_stage123_prediction_mixture.sh
```

Run pure-compound prediction:

```bash
./run_stage123_prediction_pure_compounds.sh
```

Run mixture prediction:

```bash
./run_stage123_prediction_mixture.sh
```

---

## 5. Pipeline stages

### Stage 1: RNN generation and Graphormer-IR SIS scoring

For pure-compound prediction, Stage 1 calls:

```text
./stage1/all_prediction_graphormer_rnn_sdbs.py
```

For mixture prediction, Stage 1 calls:

```text
./mixture/stage1/all_prediction_graphormer_rnn_sdbs_mixture_weighted_irsis.py
```

Stage 1 performs candidate generation, forward-IR prediction, and SIS scoring. For mixture prediction, Stage 1 additionally constructs 1-, 2-, and 3-component candidates and evaluates predefined mixture ratios.

### Stage 2: fragment-based GA reconstruction

For pure-compound prediction, Stage 2 uses:

```text
./stage2/all_prediction_GA_small_molecular.py
```

For mixture prediction:

```text
component_1:
  ./stage2/all_prediction_GA_small_molecular.py

component_2 and component_3:
  ./mixture/stage2/all_prediction_GA_small_molecular.py
```

Stage 2 outputs elite candidates such as:

```text
generation_${GA_GENERATIONS}_elites.csv
```

### Stage 3: final reranking

For pure-compound prediction, Stage 3 calls:

```text
./stage3/stage3_final_rerank.py
```

For mixture prediction, Stage 3 calls:

```text
./mixture/stage3/stage3_final_rerank.py
```

Stage 3 reranks Stage 2 candidates using:

```text
SIS score
Model III-based signature-consistency score
A candidate-intrinsic molecular feature prior
```

The default reranking weights are:

```text
w_sis     = 0.15
w_model3  = 0.15
w_feature = 0.70
feature   = heavy_asc
```

---

## 6. Important script parameters

### Pure-compound script

Edit the following parameters in:

```text
./run_stage123_prediction_pure_compounds.sh
```

Default sample selection:

```bash
TARGET_INDICES=""
START_INDEX=0
NUM_TARGETS=1000
```

This means the script starts from row 0 and processes the first 1000 rows by default. To run the full test set, set:

```bash
NUM_TARGETS=0
```

Common runtime parameters:

```bash
NUM_GENERATE=10000
GA_NUM_GENERATE=1000
GA_GENERATIONS=20

STAGE1_GPU=0
STAGE2_GPU=0
STAGE3_GPU=0
```

### Mixture script

Edit the following parameters in:

```text
./run_stage123_prediction_mixture.sh
```

Default target selection:

```bash
TARGET_INDICES=""
START=0
LIMIT="10"
NUM_TARGETS=0
```

This means the script starts from row 0 and processes rows 0–9 by default. To run from row 0 to the end of the input file, set:

```bash
LIMIT=""
NUM_TARGETS=0
```

Common runtime parameters:

```bash
FRAG_COUNT=3
COMPONENT_COUNTS="1,2,3"

NUM_GENERATE=10000
NUM_MIXTURE_CANDIDATES=10000

GA_NUM_GENERATE=1000
GA_GENERATIONS=5

RUN_COMPONENT_1=true
RUN_COMPONENT_2=true
RUN_COMPONENT_3=true

STAGE1_GPU=0
STAGE2_GPU=0
STAGE3_GPU=0
```

---

## 7. Quick tests

Because both scripts start from row 0 by default, the provided example run is a lightweight configuration for quick execution and functional verification. For a faster smoke test, further reduce the generation scale as shown below.

### Pure-compound quick test

```bash
sed -i 's/^NUM_TARGETS=.*/NUM_TARGETS=1/' run_stage123_prediction_pure_compounds.sh
sed -i 's/^GA_GENERATIONS=.*/GA_GENERATIONS=1/' run_stage123_prediction_pure_compounds.sh
sed -i 's/^GA_NUM_GENERATE=.*/GA_NUM_GENERATE=50/' run_stage123_prediction_pure_compounds.sh
sed -i 's/^NUM_GENERATE=.*/NUM_GENERATE=1000/' run_stage123_prediction_pure_compounds.sh

./run_stage123_prediction_pure_compounds.sh
```

### Mixture quick test

```bash
sed -i 's/^LIMIT=.*/LIMIT="1"/' run_stage123_prediction_mixture.sh
sed -i 's/^GA_GENERATIONS=.*/GA_GENERATIONS=1/' run_stage123_prediction_mixture.sh
sed -i 's/^GA_NUM_GENERATE=.*/GA_NUM_GENERATE=50/' run_stage123_prediction_mixture.sh
sed -i 's/^NUM_GENERATE=.*/NUM_GENERATE=1000/' run_stage123_prediction_mixture.sh
sed -i 's/^NUM_MIXTURE_CANDIDATES=.*/NUM_MIXTURE_CANDIDATES=1000/' run_stage123_prediction_mixture.sh

./run_stage123_prediction_mixture.sh
```

Expected mixture execution order:

```text
data_0:
  Stage 1
  component_1: Stage 2 SINGLE -> Stage 3
  component_2: Stage 2 MIXTURE -> Stage 3
  component_3: Stage 2 MIXTURE -> Stage 3
```

---

## 8. Output structure

### Pure-compound output

Pure-compound results are saved under:

```text
./result/
```

For sample `i`:

```text
./result/pure_compounds/sample_i/
├── data_i/
│   ├── IR_data.csv
│   ├── generated_smiles_round_1.csv
│   ├── smiles_sis_predictions.csv
│   ├── generation_${GA_GENERATIONS}_elites.csv
│   └── stage3_final_*.csv
└── logs/
    ├── stage1.log
    ├── stage2.log
    └── stage3.log
```

### Mixture output

Mixture results are saved under the output directory configured in:

```text
./run_stage123_prediction_mixture.sh
```

For example:

```text
./result/mixture/data_0/
├── IR_data.csv
├── generated_smiles_round_1.csv
├── metadata.json
├── smiles_sis_predictions.csv
│
├── components_1/
│   ├── smiles_sis_predictions.csv
│   ├── generation_${GA_GENERATIONS}_elites.csv
│   ├── stage3_final_component1.csv
│   └── stage3_final_component1_summary.csv
│
├── components_2/
│   ├── smiles_sis_predictions.csv
│   ├── generation_${GA_GENERATIONS}_elites.csv
│   ├── stage3_final_component2.csv
│   └── stage3_final_component2_summary.csv
│
└── components_3/
    ├── smiles_sis_predictions.csv
    ├── generation_${GA_GENERATIONS}_elites.csv
    ├── stage3_final_component3.csv
    └── stage3_final_component3_summary.csv
```

---

## 9. Switching mixture fragment-count datasets

The mixture script is configured through:

```text
./run_stage123_prediction_mixture.sh
```

To run another fragment-count group, modify the following variables together:

```bash
IR_CSV="${ROOT}/test_data/binary_frag1_with_sis.csv"
OUT_ROOT="${ROOT}/binary_frag1/result_mixture"
FRAG_COUNT=1
```

For `frag3`:

```bash
IR_CSV="${ROOT}/test_data/ternary_frag3_with_sis.csv"
OUT_ROOT="${ROOT}/ternary_frag3/result_mixture"
FRAG_COUNT=3
```

Make sure the corresponding fragment libraries exist:

```text
binary_frag${FRAG_COUNT}_with_sis_data_fragments.csv
ternary_frag${FRAG_COUNT}_with_sis_data_fragments.csv
```

---

## 10. Troubleshooting

### The wrong script is being used

Use:

```text
./run_stage123_prediction_pure_compounds.sh
```

for pure-compound prediction, and use:

```text
./run_stage123_prediction_mixture.sh
```

for mixture prediction.

### Component 1 calls the wrong Stage 2 script in mixture mode

The correct mixture log should show:

```text
[Stage 2 SINGLE] data_i | component 1
GA script   : ./stage2/all_prediction_GA_small_molecular.py
```

For `component_2` and `component_3`, the log should show:

```text
[Stage 2 MIXTURE] data_i | component 2
GA script   : ./mixture/stage2/all_prediction_GA_small_molecular.py
```

### Component 2 or 3 candidate generation is slow

This is expected. The GA accepts only candidates that pass validity, uniqueness, component-count, and SIS-scoring constraints. If the progress bar shows nonzero progress, the program is working.

### Stage 3 cannot find candidates

Stage 3 expects:

```text
generation_${GA_GENERATIONS}_elites.csv
```

If `GA_GENERATIONS=5`, it will look for:

```text
generation_5_elites.csv
```

Make sure Stage 2 produced that file.

### CUDA is not available

Check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

If CUDA is unavailable, recreate the environment using `environment.yml` or install PyTorch with CUDA 11.8.

---

## 11. License and review status

This repository is provided under the `GoodResolve Confidential Review License`.

Current status:

```text
Confidential review use only
Not currently released as public open-source software
No redistribution
No commercial use
No patent license granted
Pending patent application: CN202610751112.3
```

Reviewers may inspect, run, test, and locally modify the code only as reasonably necessary to verify the associated manuscript. Public release, redistribution, independent research reuse, commercial use, or integration into other projects is not permitted without separate written permission.

A separate public, research-use, or other license may be provided after completion of the review, publication, or patent-related process.

---

## 12. Notes

The example settings included in this repository are lightweight settings intended to help reviewers quickly run the code and verify the workflow. Larger-scale experiments can be performed by increasing the number of targets, generated candidates, GA generations, and candidate counts in the execution scripts.
