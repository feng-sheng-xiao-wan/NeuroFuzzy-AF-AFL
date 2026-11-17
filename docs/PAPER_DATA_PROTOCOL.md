# Data and Evaluation Protocol

## Original internal cohort

| Source | AF | AFL | Total |
| --- | ---: | ---: | ---: |
| [AFDB](https://physionet.org/content/afdb/) | 42,851 | 1,917 | 44,768 |
| [LTAFDB](https://physionet.org/content/ltafdb/) | 6,293 | 0 | 6,293 |
| [MITDB](https://physionet.org/content/mitdb/) | 1,064 | 123 | 1,187 |
| [JOINT](https://doi.org/10.13026/wgex-er52) | 1,286 | 3,247 | 4,533 |
| Total | 51,494 | 5,287 | 56,781 |

JOINT is derived from the PhysioNet 12-lead arrhythmia database. Its published mapping uses SNOMED CT 164889003 for atrial fibrillation and 164890007 for atrial flutter. The manuscript selected a target code in any diagnosis position and filtered co-occurring rhythm or ectopic diagnoses, unresolved code combinations, duplicate conflicts, and signal-check failures. Each retained JOINT ECG contributes one window.

Resample to 250 Hz and use two available channels per internal record. The paper uses 10-second non-overlapping windows. RR, quality, and fuzzy inputs are derived before the channel-wise standardization used by the morphology branch.

## Split before adding noise variants

The three long-record sources use an AFL-aware grouped split at the source-record level with seed 42 and a nominal 8:1:1 ratio. JOINT uses a separate ECG-record split stratified by AF/AFL label. The original counts reported in the manuscript are:

| Split | AF | AFL | Total | Long records | AFL-containing long records | JOINT records |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Train | 38,677 | 3,319 | 41,996 | 88 | 6 | 3,627 |
| Validation | 7,472 | 532 | 8,004 | 12 | 2 | 454 |
| Test | 5,345 | 1,436 | 6,781 | 14 | 3 | 452 |

Assign the original windows to their splits before noise augmentation. Add baseline wander, electrode-motion artifact, or muscle artifact from the [Noise Stress Test Database](https://physionet.org/content/nstdb/) to training windows at 6–18 dB SNR. RQ1 and RQ2 use original validation and test windows. JOINT grouping uses ECG record identifiers.

## Local manifest

Use one row per window with these columns:

```text
path,label_raw,record_name,dataset,feature_path,split,is_augmented_noise
```

- `label_raw` is AF or AFL. `split` is train, validation (or val), or test. `is_augmented_noise` identifies noise-augmented windows.
- `path` points to a window NPZ. Use record-prefixed names of the form `<record>_seg<window>.npz` for long recordings. For a cached variant, append `_noisy_<variant>` to its parent window stem. Keep source namespaces in `dataset`.
- `feature_path` points to the matching precomputed NPZ. Run `src/precompute_features.py --help` to see the feature options, and use the same configuration across splits.
- The training loader accepts explicit splits. The split-first helper can attach existing cached variants to an explicitly partitioned clean manifest. Its generic split fallback does not reconstruct the separate JOINT stratification or identify the exact published cohort.
- The verifier checks the disjointness of source-record groups and confines augmented rows to groups in training. The waveform annotations, patient identity, exact manuscript record list, and weight provenance require their corresponding source metadata.

The repository does not distribute the raw databases, selected JOINT/PTB-XL record lists, or the exact prediction files. [Evaluation definitions and manuscript results](MANUSCRIPT_ALIGNMENT.md)
