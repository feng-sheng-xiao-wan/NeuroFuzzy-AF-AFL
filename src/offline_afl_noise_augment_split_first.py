"""
Split-then-augment dataset builder (train-only noisy augmentation).

Policy:
1) Split CLEAN csv at patient level (seed=42, AFL-aware 8:1:1).
2) Attach noisy variants ONLY for train AFL segments.
3) Noisy rows inherit source patient ID for record_name / grouping.
4) Val/Test contain clean samples only (no noisy rows).

Reuses existing files under:
  data/holter/afl_noise_aug/
  data/precompute_features/afl_noise_aug_features/
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import noise_aware_ecg_af_afl_simple as base


def _patient_id_from_path(path_str: str) -> str:
    stem = Path(str(path_str)).stem
    if "_noisy_" in stem:
        stem = stem.split("_noisy_")[0]
    if "_seg" in stem:
        return stem.split("_seg")[0]
    return stem


def _index_existing_noisy(noise_dir: Path, feat_dir: Path):
    """Map orig_segment_stem -> list of (noisy_npz_path, feature_path)."""
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)
    if not noise_dir.exists():
        return buckets
    for npz in noise_dir.glob("*_noisy_*.npz"):
        stem = npz.stem
        orig = stem.split("_noisy_")[0]
        feat = feat_dir / f"{stem}_features.npz"
        if not feat.exists():
            continue
        buckets[orig].append((str(npz).replace("\\", "/"), str(feat).replace("\\", "/")))
    for k in buckets:
        buckets[k] = sorted(buckets[k])
    return buckets


def _pick_variants(entries: list[tuple[str, str]], num_aug: int | None = None):
    if not entries:
        return []
    if num_aug is None:
        return list(entries)
    v0 = [e for e in entries if e[0].endswith("_v0.npz") or "_v0." in e[0]]
    v1 = [e for e in entries if e[0].endswith("_v1.npz") or "_v1." in e[0]]
    picked = []
    if v0:
        picked.append(v0[0])
    if num_aug > 1 and v1:
        picked.append(v1[0])
    if len(picked) < num_aug:
        for e in entries:
            if e not in picked:
                picked.append(e)
            if len(picked) >= num_aug:
                break
    return picked[:num_aug]


def build_split_first_csv(
    clean_csv: str,
    output_csv: str,
    noise_dir: str,
    feat_dir: str,
    seed: int = 42,
    num_aug_per_afl: int | None = None,
):
    clean_csv = str(clean_csv)
    train_df, val_df, test_df = base.split_dataset_csv(clean_csv, 0.8, 0.1, 0.1, seed=seed)
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df["split"] = "train"
    val_df["split"] = "validation"
    test_df["split"] = "test"
    train_df["is_augmented_noise"] = False
    val_df["is_augmented_noise"] = False
    test_df["is_augmented_noise"] = False

    train_afl = train_df[train_df["label_raw"].astype(str).str.upper() == "AFL"].copy()
    train_stems = {Path(str(p)).stem for p in train_afl["path"]}

    noisy_index = _index_existing_noisy(Path(noise_dir), Path(feat_dir))

    noisy_rows = []
    missing = 0
    for _, row in train_afl.iterrows():
        orig_stem = Path(str(row["path"])).stem
        entries = noisy_index.get(orig_stem, [])
        picked = _pick_variants(entries, num_aug=num_aug_per_afl)
        if not picked:
            missing += 1
            continue
        patient_id = _patient_id_from_path(row["path"])
        for noisy_path, feat_path in picked:
            new_row = row.copy()
            new_row["path"] = noisy_path
            if "feature_path" in new_row.index:
                new_row["feature_path"] = feat_path
            new_row["record_name"] = patient_id
            new_row["split"] = "train"
            new_row["is_augmented_noise"] = True
            if "_record_group" in new_row.index:
                new_row["_record_group"] = patient_id
            if "dataset" in new_row.index:
                ds = str(new_row["dataset"])
                if not ds.endswith("_AFL_NOISE"):
                    new_row["dataset"] = ds + "_AFL_NOISE"
            noisy_rows.append(new_row)

    noisy_df = pd.DataFrame(noisy_rows) if noisy_rows else pd.DataFrame(columns=train_df.columns)
    out_df = pd.concat([train_df, noisy_df, val_df, test_df], ignore_index=True)

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    summary = {
        "clean_csv": clean_csv,
        "output_csv": str(out_path),
        "seed": seed,
        "train_clean": len(train_df),
        "train_afl_clean": len(train_afl),
        "train_noisy_added": len(noisy_df),
        "train_afl_missing_noisy": missing,
        "val_clean": len(val_df),
        "val_afl": int((val_df["label_raw"] == "AFL").sum()),
        "val_noisy": 0,
        "test_clean": len(test_df),
        "test_afl": int((test_df["label_raw"] == "AFL").sum()),
        "test_noisy": 0,
        "total_rows": len(out_df),
    }
    return summary, out_df


def verify_leakage(csv_path: str, seed: int = 42):
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = base.split_dataset_csv(csv_path, 0.8, 0.1, 0.1, seed=seed)

    def group_key(row) -> str:
        base_name = Path(str(row["path"])).stem
        if "_noisy_" in base_name:
            base_name = base_name.split("_noisy_", 1)[0]
        if "_seg" in base_name:
            base_name = base_name.split("_seg", 1)[0]
        dataset = str(row.get("dataset", "")).replace("_AFL_NOISE", "")
        return f"{dataset}::{base_name}"

    sm = {}
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        for _, row in part.iterrows():
            sm[group_key(row)] = name

    noisy = df[df["path"].astype(str).str.contains("_noisy_", regex=False)]
    noisy_splits = noisy.apply(lambda row: sm.get(group_key(row), "unknown"), axis=1)
    return {
        "noisy_total": int(len(noisy)),
        "noisy_not_in_train_split": int((noisy_splits != "train").sum()),
        "noisy_in_val": int((noisy_splits == "val").sum()),
        "noisy_in_test": int((noisy_splits == "test").sum()),
    }


def main():
    p = argparse.ArgumentParser(description="Build split-then-augment CSV (train-only noisy).")
    p.add_argument(
        "--clean_csv",
        default="data/precompute_features/multi_dataset_combined_with_features.csv",
    )
    p.add_argument(
        "--output_csv",
        default="data/precompute_features/multi_dataset_with_afl_noise_aug_split_first.csv",
    )
    p.add_argument("--noise_dir", default="data/holter/afl_noise_aug")
    p.add_argument("--feat_dir", default="data/precompute_features/afl_noise_aug_features")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_aug_per_afl", type=int, default=None, help="Maximum augmented variants per AFL window; default uses all available variants.")
    args = p.parse_args()

    summary, _ = build_split_first_csv(
        args.clean_csv,
        args.output_csv,
        args.noise_dir,
        args.feat_dir,
        seed=args.seed,
        num_aug_per_afl=args.num_aug_per_afl,
    )
    leak = verify_leakage(args.output_csv, seed=args.seed)

    print("=== split-then-augment summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("=== leakage check (should be all 0) ===")
    for k, v in leak.items():
        print(f"  {k}: {v}")
    print(f"Saved: {args.output_csv}")


if __name__ == "__main__":
    main()
