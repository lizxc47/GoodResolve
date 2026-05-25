# GoodResolve

GoodResolve is an infrared-spectrum-based molecular resolution framework for pure compounds and mixtures. It integrates spectral retrieval, BRICS-based molecular building-block assembly, molecular-structure-to-spectrum prediction, spectrum encoding, and functional-group-aware reranking to infer candidate molecular structures from infrared spectra without requiring molecular formula constraints.

This repository provides the source code, pretrained model weights, configuration files, synthetic demonstration spectra, candidate libraries, data manifests, and scripts required to run the main GoodResolve inference workflow.

> **Note:** GoodResolve is the subject of pending patent applications. This repository is provided for Science editorial assessment, scholarly peer review, reproducibility, and non-commercial academic research only. Commercial use is not permitted, and no patent license is granted. See [LICENSE](LICENSE) for details.

---

## Overview

GoodResolve is designed to address molecular identification and mixture resolution from infrared spectra. Unlike formula-constrained approaches, GoodResolve does not require molecular formula information as input.

The workflow contains three major stages:

### Stage 1: Spectral retrieval

Candidate molecular structures are retrieved from a molecular candidate library according to infrared spectral similarity.

### Stage 2: Molecular building-block assembly

Retrieved candidates are decomposed into BRICS fragments. These fragments, together with a global fragment library, are recombined to generate chemically plausible candidate structures beyond direct spectral-library retrieval.

### Stage 3: Functional-group-aware reranking

Candidate structures are reranked according to spectral consistency, molecular validity, predicted infrared spectra, spectrum-encoding similarity, and functional-group-level agreement with the target infrared spectrum.

GoodResolve supports:

- Pure-compound molecular resolution from IR spectra
- Binary-mixture resolution
- Ternary-mixture resolution
- Candidate generation beyond direct database retrieval
- Inference without molecular formula constraints
- Reproducible evaluation using top-k ranking metrics

---

## Repository structure

```text
GoodResolve/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
│   ├── pure.yaml
│   ├── binary_mixture.yaml
│   └── ternary_mixture.yaml
├── goodresolve/
│   ├── preprocessing/
│   ├── stage1_retrieval/
│   ├── stage2_building_blocks/
│   ├── stage3_reranking/
│   ├── mixture/
│   └── metrics/
├── scripts/
│   ├── run_goodresolve_demo.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── run_stage3.py
│   ├── evaluate_topk.py
│   └── download_models.py
├── examples/
│   ├── pure_compound/
│   ├── binary_mixture/
│   └── ternary_mixture/
├── models/
│   └── README.md
├── data_manifests/
│   ├── training_sources.csv
│   └── third_party_spectral_sources.csv
└── docs/
    ├── input_format.md
    ├── model_availability.md
    ├── data_availability.md
    ├── reproduce_main_results.md
    └── third_party_data.md
```

---

## Installation

We recommend creating a new conda environment.

```bash
conda env create -f environment.yml
conda activate goodresolve
```

Alternatively, dependencies can be installed using pip:

```bash
pip install -r requirements.txt
```

The recommended environment is:

```text
Python >= 3.10
PyTorch >= 2.0
RDKit
NumPy
Pandas
SciPy
scikit-learn
tqdm
PyYAML
```

---

## Pretrained model weights

The repository includes pretrained GoodResolve model weights or scripts for accessing them.

The pretrained components include:

```text
models/
├── graphormer_ir_predictor.pt
├── spectrum_encoder.pt
├── model_iii_reranker.pt
├── rnn_generator.pt
└── fragment_library.csv
```

The released model weights include:

- Molecular-structure-to-spectrum prediction models
- Spectrum-encoding models
- Candidate-generation models
- Functional-group-aware candidate reranking models

The pretrained weights were trained using spectral data from public and third-party spectral databases as described in the manuscript. Raw spectra, bulk processed spectra, and retrieval indices derived from restricted third-party spectral databases are not redistributed in this repository.

---

## Quick start

### Pure-compound example

```bash
python scripts/run_goodresolve_demo.py \
  --input examples/pure_compound/input_ir_synthetic.csv \
  --mode pure \
  --config configs/pure.yaml \
  --out outputs/demo_pure
```

Expected outputs:

```text
outputs/demo_pure/
├── stage1_retrieval_topk.csv
├── stage2_generated_candidates.csv
├── stage3_reranked_candidates.csv
└── summary.json
```

### Binary-mixture example

```bash
python scripts/run_goodresolve_demo.py \
  --input examples/binary_mixture/input_ir_synthetic.csv \
  --mode binary \
  --config configs/binary_mixture.yaml \
  --out outputs/demo_binary
```

### Ternary-mixture example

```bash
python scripts/run_goodresolve_demo.py \
  --input examples/ternary_mixture/input_ir_synthetic.csv \
  --mode ternary \
  --config configs/ternary_mixture.yaml \
  --out outputs/demo_ternary
```

---

## Input format

GoodResolve expects an absorbance-like infrared spectrum.

The recommended input format is:

```text
wavenumber,absorbance
4000,0.012
3998,0.015
3996,0.018
...
400,0.034
```

GoodResolve also supports vectorized spectral columns:

```text
sample_id,sp_400,sp_402,...,sp_4000
sample_001,0.034,0.035,...,0.012
```

The default wavenumber grid is:

```text
400–4000 cm^-1 with a step size of 2 cm^-1
```

If the input spectrum is provided on a different grid, it is interpolated to the predefined grid before inference.

Transmittance spectra should be converted to absorbance-like spectra before running GoodResolve.

See [docs/input_format.md](docs/input_format.md) for details.

---

## Output format

The final ranked candidate file contains:

```text
rank
candidate_smiles
canonical_smiles
mode
stage1_sis
stage2_source
predicted_spectrum_score
spectrum_encoder_score
model3_score
functional_group_score
final_score
is_valid_smiles
```

For mixture cases, additional columns are provided:

```text
component_1_smiles
component_2_smiles
component_3_smiles
mixture_pseudo_smiles
mixture_ratio
```

---

## Reproducing manuscript results

Scripts and configuration files for reproducing the main GoodResolve inference workflow are provided in this repository.

Example:

```bash
python scripts/evaluate_topk.py \
  --pred outputs/demo_pure/stage3_reranked_candidates.csv \
  --target examples/pure_compound/target_smiles.txt \
  --topk 1 3 5 10
```

For detailed instructions, see:

```text
docs/reproduce_main_results.md
```

The full benchmark experiments may require access to third-party spectral databases. Raw spectra and bulk processed spectra from restricted third-party databases are not redistributed. Source identifiers, preprocessing scripts, and data manifests are provided where permitted.

---

## Data availability

This repository provides:

- Source code
- Pretrained model weights
- Configuration files
- Synthetic demonstration spectra
- Candidate libraries
- Data manifests
- Scripts for inference and evaluation
- Documentation for reproducing the main workflow

Raw spectra from third-party spectral databases are not redistributed when database-specific terms restrict redistribution. Where permitted, this repository provides source identifiers, preprocessing scripts, and data manifests.

Users should obtain third-party spectral data directly from the corresponding database providers and follow their terms of use.

See:

```text
docs/data_availability.md
docs/third_party_data.md
data_manifests/
```

---

## Third-party data

GoodResolve may use spectral data from public and third-party spectral databases as described in the associated manuscript.

This repository does not redistribute:

- Raw spectra from restricted third-party databases
- Bulk processed spectra derived from restricted third-party databases
- Retrieval indices derived from restricted third-party spectral databases
- Embedding databases that may substitute for restricted third-party databases

The released model weights are not intended to redistribute, reconstruct, or substitute for any third-party spectral database.

---

## License

GoodResolve is released under the **GoodResolve Research Use License**.

The source code, pretrained model weights, configuration files, example materials, and documentation are provided for:

- Science editorial assessment
- Scholarly peer review
- Reproducibility
- Non-commercial academic research

Commercial use is not permitted. GoodResolve is the subject of pending patent applications, and no patent license is granted by this repository release.

See [LICENSE](LICENSE) for details.

---

## Citation

If you use GoodResolve, the pretrained model weights, or any materials from this repository in academic publications, presentations, or derivative research, please cite the associated manuscript and this repository.

```bibtex
@misc{goodresolve2026,
  title = {GoodResolve: Infrared-spectrum-based molecular resolution of pure compounds and mixtures},
  author = {Li, Ruitao and co-workers},
  year = {2026},
  url = {https://github.com/YOUR_ACCOUNT/GoodResolve},
  note = {Version v1.0.0-submission}
}
```

The citation information will be updated after publication.

---

## Contact

For questions about academic use, reproducibility, or manuscript review, please contact:

```text
[Contact name]
[Institution]
[Email address]
```

For patent licensing, commercial use, or collaboration inquiries, please contact the corresponding rights holder or institutional technology transfer office.

---

## Disclaimer

GoodResolve is provided for research and scholarly reproducibility only. It is not intended for medical diagnosis, clinical decision-making, regulatory compliance, forensic use, safety-critical systems, or any use where incorrect molecular identification may cause harm, legal consequences, financial loss, or safety risks.

The software, model weights, and outputs are provided "as is", without warranty of any kind.# GoodResolve

GoodResolve is an infrared-spectrum-based molecular resolution framework for pure compounds and mixtures. It integrates spectral retrieval, BRICS-based molecular building-block assembly, molecular-structure-to-spectrum prediction, spectrum encoding, and functional-group-aware reranking to infer candidate molecular structures from infrared spectra without requiring molecular formula constraints.

This repository provides the source code, pretrained model weights, configuration files, synthetic demonstration spectra, candidate libraries, data manifests, and scripts required to run the main GoodResolve inference workflow.

> **Note:** GoodResolve is the subject of pending patent applications. This repository is provided for Science editorial assessment, scholarly peer review, reproducibility, and non-commercial academic research only. Commercial use is not permitted, and no patent license is granted. See [LICENSE](LICENSE) for details.

---

## Overview

GoodResolve is designed to address molecular identification and mixture resolution from infrared spectra. Unlike formula-constrained approaches, GoodResolve does not require molecular formula information as input.

The workflow contains three major stages:

### Stage 1: Spectral retrieval

Candidate molecular structures are retrieved from a molecular candidate library according to infrared spectral similarity.

### Stage 2: Molecular building-block assembly

Retrieved candidates are decomposed into BRICS fragments. These fragments, together with a global fragment library, are recombined to generate chemically plausible candidate structures beyond direct spectral-library retrieval.

### Stage 3: Functional-group-aware reranking

Candidate structures are reranked according to spectral consistency, molecular validity, predicted infrared spectra, spectrum-encoding similarity, and functional-group-level agreement with the target infrared spectrum.

GoodResolve supports:

- Pure-compound molecular resolution from IR spectra
- Binary-mixture resolution
- Ternary-mixture resolution
- Candidate generation beyond direct database retrieval
- Inference without molecular formula constraints
- Reproducible evaluation using top-k ranking metrics

---

## Repository structure

```text
GoodResolve/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── environment.yml
├── configs/
│   ├── pure.yaml
│   ├── binary_mixture.yaml
│   └── ternary_mixture.yaml
├── goodresolve/
│   ├── preprocessing/
│   ├── stage1_retrieval/
│   ├── stage2_building_blocks/
│   ├── stage3_reranking/
│   ├── mixture/
│   └── metrics/
├── scripts/
│   ├── run_goodresolve_demo.py
│   ├── run_stage1.py
│   ├── run_stage2.py
│   ├── run_stage3.py
│   ├── evaluate_topk.py
│   └── download_models.py
├── examples/
│   ├── pure_compound/
│   ├── binary_mixture/
│   └── ternary_mixture/
├── models/
│   └── README.md
├── data_manifests/
│   ├── training_sources.csv
│   └── third_party_spectral_sources.csv
└── docs/
    ├── input_format.md
    ├── model_availability.md
    ├── data_availability.md
    ├── reproduce_main_results.md
    └── third_party_data.md
```

---

## Installation

We recommend creating a new conda environment.

```bash
conda env create -f environment.yml
conda activate goodresolve
```

Alternatively, dependencies can be installed using pip:

```bash
pip install -r requirements.txt
```

The recommended environment is:

```text
Python >= 3.10
PyTorch >= 2.0
RDKit
NumPy
Pandas
SciPy
scikit-learn
tqdm
PyYAML
```

---

## Pretrained model weights

The repository includes pretrained GoodResolve model weights or scripts for accessing them.

The pretrained components include:

```text
models/
├── graphormer_ir_predictor.pt
├── spectrum_encoder.pt
├── model_iii_reranker.pt
├── rnn_generator.pt
└── fragment_library.csv
```

The released model weights include:

- Molecular-structure-to-spectrum prediction models
- Spectrum-encoding models
- Candidate-generation models
- Functional-group-aware candidate reranking models

The pretrained weights were trained using spectral data from public and third-party spectral databases as described in the manuscript. Raw spectra, bulk processed spectra, and retrieval indices derived from restricted third-party spectral databases are not redistributed in this repository.

---

## Quick start

### Pure-compound example

```bash
python scripts/run_goodresolve_demo.py \
  --input examples/pure_compound/input_ir_synthetic.csv \
  --mode pure \
  --config configs/pure.yaml \
  --out outputs/demo_pure
```

Expected outputs:

```text
outputs/demo_pure/
├── stage1_retrieval_topk.csv
├── stage2_generated_candidates.csv
├── stage3_reranked_candidates.csv
└── summary.json
```

### Binary-mixture example

```bash
python scripts/run_goodresolve_demo.py \
  --input examples/binary_mixture/input_ir_synthetic.csv \
  --mode binary \
  --config configs/binary_mixture.yaml \
  --out outputs/demo_binary
```

### Ternary-mixture example

```bash
python scripts/run_goodresolve_demo.py \
  --input examples/ternary_mixture/input_ir_synthetic.csv \
  --mode ternary \
  --config configs/ternary_mixture.yaml \
  --out outputs/demo_ternary
```

---

## Input format

GoodResolve expects an absorbance-like infrared spectrum.

The recommended input format is:

```text
wavenumber,absorbance
4000,0.012
3998,0.015
3996,0.018
...
400,0.034
```

GoodResolve also supports vectorized spectral columns:

```text
sample_id,sp_400,sp_402,...,sp_4000
sample_001,0.034,0.035,...,0.012
```

The default wavenumber grid is:

```text
400–4000 cm^-1 with a step size of 2 cm^-1
```

If the input spectrum is provided on a different grid, it is interpolated to the predefined grid before inference.

Transmittance spectra should be converted to absorbance-like spectra before running GoodResolve.

See [docs/input_format.md](docs/input_format.md) for details.

---

## Output format

The final ranked candidate file contains:

```text
rank
candidate_smiles
canonical_smiles
mode
stage1_sis
stage2_source
predicted_spectrum_score
spectrum_encoder_score
model3_score
functional_group_score
final_score
is_valid_smiles
```

For mixture cases, additional columns are provided:

```text
component_1_smiles
component_2_smiles
component_3_smiles
mixture_pseudo_smiles
mixture_ratio
```

---

## Reproducing manuscript results

Scripts and configuration files for reproducing the main GoodResolve inference workflow are provided in this repository.

Example:

```bash
python scripts/evaluate_topk.py \
  --pred outputs/demo_pure/stage3_reranked_candidates.csv \
  --target examples/pure_compound/target_smiles.txt \
  --topk 1 3 5 10
```

For detailed instructions, see:

```text
docs/reproduce_main_results.md
```

The full benchmark experiments may require access to third-party spectral databases. Raw spectra and bulk processed spectra from restricted third-party databases are not redistributed. Source identifiers, preprocessing scripts, and data manifests are provided where permitted.

---

## Data availability

This repository provides:

- Source code
- Pretrained model weights
- Configuration files
- Synthetic demonstration spectra
- Candidate libraries
- Data manifests
- Scripts for inference and evaluation
- Documentation for reproducing the main workflow

Raw spectra from third-party spectral databases are not redistributed when database-specific terms restrict redistribution. Where permitted, this repository provides source identifiers, preprocessing scripts, and data manifests.

Users should obtain third-party spectral data directly from the corresponding database providers and follow their terms of use.

See:

```text
docs/data_availability.md
docs/third_party_data.md
data_manifests/
```

---

## Third-party data

GoodResolve may use spectral data from public and third-party spectral databases as described in the associated manuscript.

This repository does not redistribute:

- Raw spectra from restricted third-party databases
- Bulk processed spectra derived from restricted third-party databases
- Retrieval indices derived from restricted third-party spectral databases
- Embedding databases that may substitute for restricted third-party databases

The released model weights are not intended to redistribute, reconstruct, or substitute for any third-party spectral database.

---

## License

GoodResolve is released under the **GoodResolve Research Use License**.

The source code, pretrained model weights, configuration files, example materials, and documentation are provided for:

- Science editorial assessment
- Scholarly peer review
- Reproducibility
- Non-commercial academic research

Commercial use is not permitted. GoodResolve is the subject of pending patent applications, and no patent license is granted by this repository release.

See [LICENSE](LICENSE) for details.

---

## Citation

If you use GoodResolve, the pretrained model weights, or any materials from this repository in academic publications, presentations, or derivative research, please cite the associated manuscript and this repository.

```bibtex
@misc{goodresolve2026,
  title = {GoodResolve: Infrared-spectrum-based molecular resolution of pure compounds and mixtures},
  author = {Li, Ruitao and co-workers},
  year = {2026},
  url = {https://github.com/lizxc47/GoodResolve},
  note = {Version v1.0.0-submission}
}
```

The citation information will be updated after publication.

---

## Contact

For questions about academic use, reproducibility, or manuscript review, please contact:

```text
[Contact name]
[Institution]
[Email address]
```

For patent licensing, commercial use, or collaboration inquiries, please contact the corresponding rights holder or institutional technology transfer office.

---

## Disclaimer

GoodResolve is provided for research and scholarly reproducibility only. It is not intended for medical diagnosis, clinical decision-making, regulatory compliance, forensic use, safety-critical systems, or any use where incorrect molecular identification may cause harm, legal consequences, financial loss, or safety risks.

The software, model weights, and outputs are provided "as is", without warranty of any kind.
