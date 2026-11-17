"""Evaluate inference interventions on the manifest's original validation and test splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import noise_aware_ecg_af_afl_simple as base
from test_saved_model import build_model, get_device, load_checkpoint, load_model_weights


MODES = ("neural_only", "zeroed_quality", "fixed_fusion", "fuzzy_only")


@torch.inference_mode()
def collect_probabilities(model, loader, device):
    labels = []
    probabilities = {mode: [] for mode in MODES}

    for batch in loader:
        ecg = batch["ecg"].to(device)
        rr_seq = batch["rr_seq"].to(device)
        quality = batch["noise_feat"].to(device)
        fuzzy = batch["fuzzy_logits"].to(device)[:, :2]

        neural, _ = model(ecg, rr_seq, quality, None)
        zeroed, _ = model(ecg, rr_seq, torch.zeros_like(quality), fuzzy)
        logits = {
            "neural_only": neural,
            "zeroed_quality": zeroed,
            "fixed_fusion": 0.7 * neural + 0.3 * fuzzy,
            "fuzzy_only": fuzzy,
        }
        labels.append(batch["label"].numpy())
        for mode, values in logits.items():
            probabilities[mode].append(values.softmax(dim=1)[:, 1].cpu().numpy())

    return np.concatenate(labels), {
        mode: np.concatenate(values) for mode, values in probabilities.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    _, validation, test = base.split_dataset_csv(
        str(args.manifest), require_predefined_split=True
    )
    for name, frame in (("validation", validation), ("test", test)):
        if set(frame["label_raw"].astype(str).str.strip().str.upper()) != {"AF", "AFL"}:
            parser.error(f"{name} must contain both AF and AFL windows and no other labels")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    validation_csv = args.output_dir / "validation_manifest.csv"
    test_csv = args.output_dir / "test_manifest.csv"
    validation.to_csv(validation_csv, index=False)
    test.to_csv(test_csv, index=False)

    device = get_device(args.device)
    checkpoint, cfg, _, _, _ = load_checkpoint(str(args.checkpoint), device)
    if cfg.num_arrhythmia_classes != 2:
        parser.error("an AF/AFL binary checkpoint is required")

    dataset_options = {
        "max_len": cfg.max_len,
        "num_leads": cfg.num_leads,
        "max_rr_intervals": cfg.max_rr_intervals,
        "use_stft_dbscan": cfg.use_stft_dbscan,
        "multi_lead_method": cfg.multi_lead_method,
        "flutter_method": cfg.flutter_method,
        "lead_names": cfg.lead_names,
        "use_precomputed": cfg.use_precomputed,
        "augment": False,
    }
    validation_ds = base.ECGDataset(str(validation_csv), **dataset_options)
    test_ds = base.ECGDataset(str(test_csv), **dataset_options)
    sample = validation_ds[0]
    model = build_model(cfg, device, sample["noise_feat"].numel(), sample["rr_seq"].numel())
    model = load_model_weights(model, checkpoint, device)

    validation_loader = DataLoader(validation_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    y_val, val_probs = collect_probabilities(model, validation_loader, device)
    y_test, test_probs = collect_probabilities(model, test_loader, device)

    metrics = []
    for mode in MODES:
        threshold, _ = base._search_best_afl_threshold(y_val, val_probs[mode], steps=1001)
        prediction = (test_probs[mode] >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        metrics.append({
            "variant": mode,
            "threshold": threshold,
            "accuracy": (prediction == y_test).mean(),
            "afl_precision": tp / (tp + fp) if tp + fp else 0.0,
            "afl_recall": tp / (tp + fn),
            "afl_f1": f1_score(y_test, prediction, zero_division=0),
            "macro_f1": f1_score(y_test, prediction, average="macro", zero_division=0),
            "auroc": roc_auc_score(y_test, test_probs[mode]),
            "ap": average_precision_score(y_test, test_probs[mode]),
        })

    pd.DataFrame(metrics).to_csv(args.output_dir / "metrics.csv", index=False)
    validation[["path"]].assign(label=y_val, **val_probs).to_csv(
        args.output_dir / "validation_predictions.csv", index=False
    )
    test[["path"]].assign(label=y_test, **test_probs).to_csv(
        args.output_dir / "test_predictions.csv", index=False
    )


if __name__ == "__main__":
    main()
