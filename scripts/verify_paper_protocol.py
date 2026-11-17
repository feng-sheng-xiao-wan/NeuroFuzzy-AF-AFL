#!/usr/bin/env python3
"""Verify the split/augmentation protocol required by the paper.

Usage:
    python scripts/verify_paper_protocol.py --manifest path/to/manifest.csv

Checks:
1. train/validation/test splits are explicitly present;
2. clean source groups are disjoint across splits;
3. no augmented rows occur in validation/test;
4. every augmented row belongs to a source group present in the training split.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def record_group(row) -> str:
    stem = Path(str(row["path"])).name
    if stem.lower().endswith(".npz"):
        stem = stem[:-4]
    if "_noisy_" in stem:
        stem = stem.split("_noisy_", 1)[0]
    if "_seg" in stem:
        stem = stem.split("_seg", 1)[0]
    dataset = str(row.get("dataset", "")).replace("_AFL_NOISE", "")
    return f"{dataset}::{stem}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    if "split" not in df.columns:
        raise SystemExit("FAIL: manifest has no split column")
    if "path" not in df.columns:
        raise SystemExit("FAIL: manifest has no path column")

    split = df["split"].astype(str).str.lower().str.strip().replace({"val": "validation"})
    if set(split.unique()) != {"train", "validation", "test"}:
        raise SystemExit(f"FAIL: unexpected splits: {sorted(split.unique())}")

    if "is_augmented_noise" in df.columns:
        augmented = df["is_augmented_noise"].astype(str).str.lower().isin(["1", "true", "yes"])
    else:
        augmented = df["path"].astype(str).str.contains("_noisy_", regex=False)
    groups = df.apply(record_group, axis=1)

    clean_groups = {
        name: set(groups[(~augmented) & split.eq(name)])
        for name in ("train", "validation", "test")
    }
    overlaps = {
        "train_validation": clean_groups["train"] & clean_groups["validation"],
        "train_test": clean_groups["train"] & clean_groups["test"],
        "validation_test": clean_groups["validation"] & clean_groups["test"],
    }
    bad_overlaps = {k: sorted(v) for k, v in overlaps.items() if v}
    if bad_overlaps:
        raise SystemExit(f"FAIL: clean source-group overlap: {bad_overlaps}")

    bad_augmented = split[augmented & split.isin(["validation", "test"])]
    if len(bad_augmented):
        raise SystemExit(f"FAIL: {len(bad_augmented)} augmented rows occur in validation/test")

    augmented_groups = set(groups[augmented])
    outside_train = augmented_groups - clean_groups["train"]
    if outside_train:
        raise SystemExit(f"FAIL: augmented parent groups outside train: {sorted(outside_train)}")

    print("PASS: split/augmentation protocol")
    print(f"rows={len(df)} augmented_rows={int(augmented.sum())}")
    for name in ("train", "validation", "test"):
        part = df[split.eq(name)]
        print(f"{name}: rows={len(part)} clean_groups={len(clean_groups[name])} augmented={int(augmented[split.eq(name)].sum())}")


if __name__ == "__main__":
    main()
