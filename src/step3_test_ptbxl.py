#!/usr/bin/env python3
"""
第三步：在PTB-XL AF/AFL数据上测试保存的模型

用法: python step3_test_ptbxl.py --checkpoint <模型路径> --test_csv <测试CSV>
"""

import argparse
import sys
import os

# 直接导入测试脚本中的函数
from test_saved_model import (
    load_checkpoint,
    build_model,
    load_model_weights,
    print_detailed_metrics,
    get_device,
)
from noise_aware_ecg_af_afl_simple import (
    ECGDataset,
    evaluate,
)
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(
        description="Test saved model on PTB-XL AF/AFL dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to saved model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--test_csv",
        type=str,
        default="data/precompute_features/ptbxl/ptbxl_af_afl_with_features.csv",
        help="PTB-XL II+V1 test manifest with matching precomputed features",
    )
    parser.add_argument(
        "--afl_prob_threshold",
        type=float,
        help="Fixed AFL probability threshold (0.522 for manuscript RQ3)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cpu, or cuda:0)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for testing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--no_use_precomputed",
        action="store_true",
        help="Disable precomputed features (will compute all features on-the-fly)",
    )
    parser.add_argument(
        "--recompute_fuzzy_only",
        action="store_true",
        help="Recompute only fuzzy_logits (other features still use precomputed files)",
    )
    
    args = parser.parse_args()
    if args.afl_prob_threshold is not None and not 0 <= args.afl_prob_threshold <= 1:
        parser.error("--afl_prob_threshold must be between 0 and 1")
    
    print("=" * 80)
    print("Step 3: Testing Model on PTB-XL AF/AFL Dataset")
    print("=" * 80)
    
    # 检查文件
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    if not os.path.exists(args.test_csv):
        print(f"Error: Test CSV not found: {args.test_csv}")
        sys.exit(1)
    
    # 获取设备
    device = get_device(args.device)
    print(f"\n[1/5] Device: {device}")
    
    # 加载checkpoint
    print(f"\n[2/5] Loading checkpoint...")
    checkpoint, cfg, _, _, _ = load_checkpoint(args.checkpoint, device)
    
    # 准备测试数据
    print(f"\n[3/5] Preparing test data...")
    print(f"  - Test CSV: {args.test_csv}")
    
    # 如果指定了 --no_use_precomputed，覆盖配置中的设置
    use_precomputed = cfg.use_precomputed if not args.no_use_precomputed else False
    if args.no_use_precomputed:
        print(f"  - Precomputed features disabled (will compute all features on-the-fly)")
    elif args.recompute_fuzzy_only:
        print(f"  - Recomputing fuzzy_logits only (other features use precomputed files)")
    
    test_ds = ECGDataset(
        args.test_csv,
        max_len=cfg.max_len,
        num_leads=cfg.num_leads,
        max_rr_intervals=cfg.max_rr_intervals,
        use_stft_dbscan=cfg.use_stft_dbscan,
        multi_lead_method=cfg.multi_lead_method,
        flutter_method=cfg.flutter_method,
        lead_names=cfg.lead_names,
        use_precomputed=use_precomputed,
        recompute_fuzzy_only=args.recompute_fuzzy_only,
    )
    
    sample = test_ds[0]
    noise_dim = sample["noise_feat"].shape[0]
    rr_len = sample["rr_seq"].shape[0]
    
    print(f"  - Test samples: {len(test_ds)}")
    print(f"  - Noise feat dim: {noise_dim}")
    print(f"  - RR seq len: {rr_len}")
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    
    # 构建模型
    print(f"\n[4/5] Building model...")
    model = build_model(cfg, device, noise_dim, rr_len)
    model = load_model_weights(model, checkpoint, device)
    
    # 评估模型
    print(f"\n[5/5] Evaluating model on test set...")
    if args.afl_prob_threshold is not None:
        print(f"  - AFL probability threshold: {args.afl_prob_threshold:g}")
    test_metrics = evaluate(
        model, test_loader, device, verbose=True,
        afl_prob_threshold=args.afl_prob_threshold,
    )
    
    # 打印详细结果
    print_detailed_metrics(test_metrics, cfg.num_arrhythmia_classes)
    
    print("\n" + "=" * 80)
    print("Step 3 Complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

