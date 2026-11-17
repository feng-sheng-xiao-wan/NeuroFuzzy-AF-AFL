# NeuroFuzzy-AF-AFL

Core implementation for *Adaptive Fusion of Neural and Heuristic Fuzzy-Rule Evidence for Atrial Fibrillation versus Atrial Flutter Classification in Short Two-Channel ECG Windows* by Xiaolin Ju, Xin Zhou, Xiaodong Ji, Zhan Gao, and Haiyan Pan.

The model classifies 10-second ECG windows already identified as AF/AFL candidates. It combines a waveform CNN, an RR-interval Bi-GRU, and a seven-input signal-quality MLP with AF/AFL fuzzy-rule evidence. A gate conditioned on signal quality and neural confidence, together with class-specific adjustments, produces the fused prediction. The internal model uses the first two available ECG channels at 250 Hz, with 2,500 samples per channel.

## Manuscript results

The internal cohort combines AFDB, LTAFDB, MITDB, and JOINT. Full adaptive fusion achieved **AFL-F1 0.778**, precision 0.961, recall 0.654, AUROC 0.811, and average precision 0.758 at a validation-selected threshold of 0.522 on **6,781 original test windows**. On the separate frozen-model PTB-XL assessment, AFL-F1 was 0.680 at the same threshold. [Results and experiment definitions](docs/MANUSCRIPT_ALIGNMENT.md)

These are values reported in the manuscript. The release contains the core model and analysis utilities. Reproducing the reported experiments requires the original cohort manifests, features, and corresponding checkpoints.

## Setup

Use Python 3.10 or later. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Train and evaluate

Prepare the original 10-second windows, assign the explicit source-record splits, and compute features. The data guide documents the [cohort, input format, and split-first workflow](docs/PAPER_DATA_PROTOCOL.md). With a prepared manifest:

```powershell
.\.venv\Scripts\python.exe scripts/verify_paper_protocol.py --manifest data/train_manifest.csv
.\.venv\Scripts\python.exe src/train_af_vs_afl_afdb_ltafdb.py --combined_csv data/train_manifest.csv --output_dir outputs/internal --num_leads 2
```

| Entry point | Use |
| --- | --- |
| [Model](src/noise_aware_ecg_af_afl_simple.py) | Neural encoder, binary fuzzy rules, adaptive fusion, and training |
| [Feature computation](src/precompute_features.py) | RR sequences, signal-quality descriptors, and fuzzy-rule inputs |
| [Sensitivity analysis](scripts/evaluate_inference_sensitivity.py) | RQ2 inference interventions on a checkpoint using original validation and test windows |
| [PTB-XL evaluation](src/step3_test_ptbxl.py) | Frozen-checkpoint evaluation on a prepared II+V1 manifest at a supplied threshold |

Supply a trained checkpoint when running either evaluation entry point. The checkpoint for the four-source manuscript runs is not included in this repository.
