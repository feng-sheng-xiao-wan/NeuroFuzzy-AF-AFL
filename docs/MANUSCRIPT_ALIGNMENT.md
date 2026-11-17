# Manuscript Results and Evaluation

AFL is the positive class (AF = 0, AFL = 1). The values below are reported in the manuscript.

## RQ1 — Original internal windows

Each of the eight configurations used its own threshold selected on 8,004 original validation windows and then applied to 6,781 original test windows (5,345 AF and 1,436 AFL).

| Configuration | Threshold | TP | FP | AF specificity | AFL precision | AFL recall | AFL-F1 | AUROC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure neural | 0.525 | 885 | 35 | 0.993 | 0.962 | 0.616 | 0.751 | 0.819 | 0.751 |
| Same-descriptor concatenation | 0.514 | 905 | 34 | 0.994 | 0.964 | 0.630 | 0.762 | 0.827 | 0.760 |
| Fixed fusion | 0.517 | 918 | 37 | 0.993 | 0.961 | 0.639 | 0.768 | 0.821 | 0.760 |
| Full adaptive fusion | 0.522 | 939 | 38 | 0.993 | 0.961 | 0.654 | 0.778 | 0.811 | 0.758 |
| Without quality branch | 0.528 | 914 | 35 | 0.993 | 0.963 | 0.636 | 0.766 | 0.824 | 0.762 |
| Without flutter stability | 0.528 | 922 | 35 | 0.993 | 0.963 | 0.642 | 0.771 | 0.819 | 0.760 |
| RR gradient boosting | 0.515 | 785 | 34 | 0.994 | 0.958 | 0.547 | 0.696 | 0.757 | 0.688 |
| RR + spectral logistic regression | 0.523 | 823 | 26 | 0.995 | 0.969 | 0.573 | 0.720 | 0.774 | 0.705 |

## RQ2 — Inference-time sensitivity

All four variants use the selected checkpoint without retraining. The zeroed quality vector still passes through the quality MLP. The fixed mixture uses 0.7 times the neural logits plus 0.3 times the fuzzy logits. Each threshold maximizes AFL-F1 on the corresponding original validation predictions.

| Variant | Threshold | Accuracy | AFL precision | AFL recall | AFL-F1 | Macro-F1 | AUROC | AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Neural logits only | 0.78714 | 0.861 | 0.968 | 0.357 | 0.522 | 0.720 | 0.823 | 0.762 |
| Zeroed quality input | 0.57056 | 0.920 | 0.960 | 0.648 | 0.773 | 0.862 | 0.811 | 0.776 |
| Fixed fusion (alpha = 0.30) | 0.56664 | 0.916 | 0.967 | 0.625 | 0.759 | 0.854 | 0.825 | 0.782 |
| Fuzzy-rule logits only | 0.01490 | 0.255 | 0.221 | 0.999 | 0.362 | 0.234 | 0.777 | 0.565 |

With a prepared internal manifest and the selected checkpoint:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_inference_sensitivity.py --manifest data/train_manifest.csv --checkpoint outputs/internal/af_vs_afl_afdb_ltafdb.pt --output-dir outputs/sensitivity
```

## RQ3 — Frozen-model PTB-XL assessment

The cohort contains 104 II+V1 ECG records from 81 patients (48 AF and 56 AFL), one window per record. At the RQ1 full-adaptive threshold of 0.522, accuracy was 0.702, AFL precision 0.805, AFL recall 0.589, and AFL-F1 0.680. The 95% record-bootstrap interval for AFL-F1 was 0.562–0.783 (2,000 resamples). Rows and columns in the confusion matrix are true and predicted AF/AFL, respectively.

| | Predicted AF | Predicted AFL |
| --- | ---: | ---: |
| True AF | 40 | 8 |
| True AFL | 23 | 33 |

To score a prepared II+V1 feature manifest with this operating threshold:

```powershell
.\.venv\Scripts\python.exe src/step3_test_ptbxl.py --checkpoint outputs/internal/af_vs_afl_afdb_ltafdb.pt --test_csv data/ptbxl_rq3.csv --afl_prob_threshold 0.522
```

## RQ4 — PTB-XL lead comparison

A separately defined cohort contains 104 ECGs from 85 patients, with 90 ECGs shared with RQ3. Both lead configurations were trained from scratch with the same splits. Five seeds (42–46) each used five outer patient-grouped folds and record-stratified inner validation drawn from the development records. Reported values are mean ± sample standard deviation across 25 seed–fold evaluations. Internal AP is computed by average precision. The RQ4 PR-AUC uses trapezoidal integration.

| Metric | 12 leads | II+V1 |
| --- | ---: | ---: |
| Accuracy | 0.8650 ± 0.1087 | 0.8861 ± 0.0962 |
| AFL precision | 0.8161 ± 0.1500 | 0.8676 ± 0.1359 |
| AFL recall | 0.9710 ± 0.0655 | 0.9424 ± 0.1020 |
| AFL-F1 | 0.8787 ± 0.0968 | 0.8922 ± 0.0866 |
| Macro-F1 | 0.8497 ± 0.1200 | 0.8746 ± 0.1076 |
| AUROC | 0.9527 ± 0.0671 | 0.9892 ± 0.0181 |
| Trapezoidal PR-AUC | 0.9433 ± 0.0907 | 0.9918 ± 0.0141 |

## RQ5 — Local ECG attribution

The paper illustrates one original AF window and one noise-augmented AFL window from a fixed auxiliary list. The maps show absolute gradients of the reference-class fused logit, averaged across leads and min–max scaled within each example. These two examples describe local attribution.
