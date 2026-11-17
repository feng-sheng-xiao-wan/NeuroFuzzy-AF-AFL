# train_af_vs_afl_afdb_ltafdb.py
"""
基于 noise_aware_ecg_af_afl_simple.py 的 AF vs AFL 二分类训练脚本
专门用于 AFDB 和 LTAFDB 数据集的二分类任务

主要修改：
1. 二分类：AF=0, AFL=1（仅保留 AF/AFL）
2. 使用 AFDB 和 LTAFDB 数据集
3. 技术路线保持不变：形态+节律+噪声分支+模糊规则
4. 修改模糊规则输出，只保留AF和AFL的logits
"""

import sys
import os

# 导入基础训练脚本
import noise_aware_ecg_af_afl_simple as base

import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from typing import Optional, List
from pathlib import Path


def combine_afdb_ltafdb_csvs(
    afdb_csv: str,
    ltafdb_csv: str,
    output_csv: str,
) -> pd.DataFrame:
    """
    合并 AFDB 和 LTAFDB 的 CSV 文件
    
    Args:
        afdb_csv: AFDB segments CSV 路径
        ltafdb_csv: LTAFDB segments CSV 路径
        output_csv: 输出合并后的 CSV 路径
    
    Returns:
        合并后的 DataFrame
    """
    # 读取两个CSV
    df_afdb = pd.read_csv(afdb_csv)
    df_ltafdb = pd.read_csv(ltafdb_csv)
    
    # 确保只包含AF和AFL
    df_afdb = df_afdb[df_afdb['label_raw'].isin(['AF', 'AFL'])].copy()
    df_ltafdb = df_ltafdb[df_ltafdb['label_raw'].isin(['AF', 'AFL'])].copy()
    
    # 合并
    df_combined = pd.concat([df_afdb, df_ltafdb], ignore_index=True)
    
    # ECGDataset 会自动将 label_raw 映射为 label
    # 对于二分类，我们需要确保映射正确：AF=0, AFL=1
    # ECGDataset 的二分类映射是：AF=0, AFL=1
    
    # 保存
    df_combined.to_csv(output_csv, index=False)
    
    print(f"Combined {len(df_afdb)} AFDB segments + {len(df_ltafdb)} LTAFDB segments = {len(df_combined)} total")
    print(f"Label distribution:")
    print(df_combined['label_raw'].value_counts())
    
    return df_combined




def run_training_binary(
    afdb_csv: str,
    ltafdb_csv: str,
    output_dir: str = "outputs",
    **train_kwargs
):
    """
    运行二分类训练
    
    Args:
        afdb_csv: AFDB segments CSV 路径（如果是预计算特征CSV，ltafdb_csv可以为空）
        ltafdb_csv: LTAFDB segments CSV 路径
        output_dir: 输出目录
        **train_kwargs: 传递给训练配置的其他参数
    """
    print("=" * 80)
    print("AF vs AFL Binary Classification (AFDB + LTAFDB)")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查是否使用预计算特征CSV
    if "with_features" in afdb_csv or (ltafdb_csv == "" or ltafdb_csv is None):
        print("\n[1/7] Using precomputed features CSV...")
        combined_csv = afdb_csv  # 直接使用预计算特征CSV
        print(f"  - Using precomputed CSV: {combined_csv}")
    else:
        # 合并CSV（原始逻辑）
        combined_csv = os.path.join(output_dir, "afdb_ltafdb_combined.csv")
        print("\n[1/7] Combining AFDB and LTAFDB datasets...")
        df_combined = combine_afdb_ltafdb_csvs(afdb_csv, ltafdb_csv, combined_csv)
    
    # 创建训练配置
    print("\n[2/7] Creating training configuration...")
    cfg = base.TrainConfig(
        data_csv=combined_csv,
        num_arrhythmia_classes=2,  # 二分类：AF=0, AFL=1
        out_ckpt=os.path.join(output_dir, "af_vs_afl_afdb_ltafdb.pt"),
        **train_kwargs
    )
    
    # 注意：ECGDataset 会自动将 label_raw='AF' 映射为 label=0，label_raw='AFL' 映射为 label=1
    # 这与我们的二分类目标一致
    
    # The binary fuzzy wrapper returns two rule logits directly. The legacy
    # four-logit compatibility path remains in base code for older checkpoints.
    
    # 运行训练（使用修改后的配置）
    print("\n[3/7] Starting training...")
    
    # 临时替换 apply_fuzzy_rules 函数，使其返回2个logits
    original_apply_fuzzy_rules = base.apply_fuzzy_rules
    
    def apply_fuzzy_rules_binary_wrapper(*args, **kwargs):
        logits_4 = original_apply_fuzzy_rules(*args, **kwargs)
        return logits_4[:2]  # 只返回前两个logits
    
    # 替换函数
    base.apply_fuzzy_rules = apply_fuzzy_rules_binary_wrapper
    
    try:
        model = base.run_training(cfg)
    finally:
        # 恢复原始函数
        base.apply_fuzzy_rules = original_apply_fuzzy_rules
    
    return model


def parse_args():
    parser = argparse.ArgumentParser(
        description="AF vs AFL binary classification training (AFDB + LTAFDB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # 数据路径
    parser.add_argument("--combined_csv", type=str, help="Combined CSV path (AFDB + LTAFDB with features)")
    parser.add_argument("--afdb_csv", type=str, help="AFDB segments CSV path (legacy)")
    parser.add_argument("--ltafdb_csv", type=str, default="", help="LTAFDB segments CSV path (legacy)")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument(
        "--allow_random_resplit",
        action="store_true",
        help="Legacy escape hatch: allow re-splitting a CSV that lacks a split column. Do not use for paper reproduction.",
    )
    
    # 训练参数（继承自基础脚本）
    parser.add_argument("--num_leads", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=2500)
    parser.add_argument("--max_rr_intervals", type=int, default=32)
    parser.add_argument("--use_stft_dbscan", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--weight_conf_gamma", type=float, default=0.5)
    parser.add_argument("--min_weight", type=float, default=0.1)
    parser.add_argument("--conf_discard_th", type=float, default=0.3)
    parser.add_argument("--lambda_af_bin", type=float, default=0.0, help="AF/AFL auxiliary head loss weight (reported configuration: 0.0)")
    parser.add_argument("--validate_every_batch", action="store_true", default=False)
    parser.add_argument("--no_validate_every_batch", dest="validate_every_batch", action="store_false")
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_use_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--fuzzy_weight_init", type=float, default=0.3)
    parser.add_argument("--multi_lead_method", type=str, default="voting", choices=["max_energy", "voting"])
    parser.add_argument("--flutter_method", type=str, default="attention", choices=["fixed_lead", "attention"])
    parser.add_argument("--lead_names", type=str, nargs="+", default=None)
    parser.add_argument("--use_precomputed", action="store_true", default=True)
    parser.add_argument("--no_use_precomputed", dest="use_precomputed", action="store_false")
    parser.add_argument("--use_boundary_aware", action="store_true", default=True)
    parser.add_argument("--no_use_boundary_aware", dest="use_boundary_aware", action="store_false")
    parser.add_argument("--boundary_alpha", type=float, default=0.6)
    parser.add_argument("--boundary_threshold", type=float, default=0.3)
    parser.add_argument("--lambda_boundary", type=float, default=0.1)
    # 二分类评估：阈值化 &（可选）logit 偏置
    parser.add_argument(
        "--afl_logit_bias_inference",
        type=float,
        default=0.0,
        help="Binary inference only: add bias to AFL logit (index=1). Prefer threshold tuning over large bias.",
    )
    parser.add_argument(
        "--afl_prob_threshold",
        type=float,
        default=None,
        help="Binary inference only: classify as AFL if P(AFL)>=threshold. If not set, auto-tune on val to maximize AFL-F1.",
    )
    parser.add_argument(
        "--afl_threshold_search_steps",
        type=int,
        default=1001,
        help="Binary val auto-tuning: number of thresholds to search.",
    )
    parser.add_argument(
        "--afl_threshold_search_min",
        type=float,
        default=0.01,
        help="Binary val auto-tuning: min threshold.",
    )
    parser.add_argument(
        "--afl_threshold_search_max",
        type=float,
        default=0.99,
        help="Binary val auto-tuning: max threshold. Reported configuration scans 0.01--0.99.",
    )
    parser.add_argument(
        "--afl_min_recall_val",
        type=float,
        default=0.0,
        help="Reported configuration does not impose a minimum validation recall.",
    )
    # 方案1：record-level 聚合评估
    parser.add_argument("--enable_record_agg", action="store_true", default=False, help="Enable record-level aggregation for binary eval")
    parser.add_argument("--no_enable_record_agg", dest="enable_record_agg", action="store_false", help="Disable record-level aggregation")
    parser.add_argument("--record_agg_method", type=str, default="mean", choices=["mean", "median", "max", "topk"], help="Record aggregation method")
    parser.add_argument("--record_agg_topk", type=int, default=5, help="Top-k for record_agg_method=topk")
    # 方案2：hard example mining
    parser.add_argument("--use_hard_mining", action="store_true", default=True, help="Enable hard example mining via weighted sampling")
    parser.add_argument("--no_use_hard_mining", dest="use_hard_mining", action="store_false", help="Disable hard example mining")
    parser.add_argument("--hard_mining_start_epoch", type=int, default=1, help="Start hard mining after this epoch")
    parser.add_argument("--hard_mining_afl_fn_mult", type=float, default=4.0, help="Weight multiplier for AFL->AF false negatives")
    parser.add_argument("--hard_mining_af_fp_mult", type=float, default=2.0, help="Weight multiplier for AF->AFL false positives")
    parser.add_argument("--hard_mining_low_conf_mult", type=float, default=1.5, help="Weight multiplier for low-confidence correct samples")
    parser.add_argument("--hard_mining_conf_threshold", type=float, default=0.65, help="Low-confidence threshold for hard mining")
    # 早停机制（基于验证集 macro F1）
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=0,
        help="Early stopping patience (number of epochs with no Val F1 improvement before stopping, 0=disable)",
    )
    parser.add_argument(
        "--early_stop_min_delta",
        type=float,
        default=0.0,
        help="Minimum improvement on Val macro F1 to be considered as an improvement",
    )
    parser.add_argument("--use_low_conf_focus", action="store_true", default=True, help="Focus on low-confidence samples (give them higher weight) instead of discarding them")
    parser.add_argument("--no_use_low_conf_focus", dest="use_low_conf_focus", action="store_false", help="Use original discard strategy for low-confidence samples")
    parser.add_argument("--low_conf_alpha", type=float, default=1.5, help="Weight enhancement coefficient for low-confidence samples (reported configuration: 1.5)")
    # 过采样参数
    parser.add_argument("--use_oversampling", action="store_true", default=True, help="Use oversampling for training data")
    parser.add_argument("--no_use_oversampling", dest="use_oversampling", action="store_false", help="Disable oversampling")
    parser.add_argument("--oversample_target_ratio", type=float, default=0.5, help="Oversampling target ratio (reported configuration: 0.5)")
    parser.add_argument("--oversample_strategy", type=str, default="patient_level", choices=["random", "patient_level"], help="Oversampling strategy (reported configuration: patient_level)")
    # 预计算特征参数
    parser.add_argument("--recompute_fuzzy_only", action="store_true", default=False, help="Recompute only fuzzy features even if precomputed features exist")
    parser.add_argument("--split_seed", type=int, default=42, help="Patient-level 8:1:1 split RNG seed")
    
    args = parser.parse_args()
    
    # 构建训练参数字典
    train_kwargs = {
        'num_leads': args.num_leads,
        'max_len': args.max_len,
        'max_rr_intervals': args.max_rr_intervals,
        'use_stft_dbscan': args.use_stft_dbscan,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'epochs': args.epochs,
        'device': args.device,
        'weight_conf_gamma': args.weight_conf_gamma,
        'min_weight': args.min_weight,
        'conf_discard_th': args.conf_discard_th,
        'lambda_af_bin': args.lambda_af_bin,
        'validate_every_batch': args.validate_every_batch,
        'use_class_weights': args.use_class_weights,
        'fuzzy_weight_init': args.fuzzy_weight_init,
        'multi_lead_method': args.multi_lead_method,
        'flutter_method': args.flutter_method,
        'lead_names': args.lead_names,
        'use_precomputed': args.use_precomputed,
        'use_boundary_aware': args.use_boundary_aware,
        'boundary_alpha': args.boundary_alpha,
        'boundary_threshold': args.boundary_threshold,
        'lambda_boundary': args.lambda_boundary,
        'afl_logit_bias_inference': args.afl_logit_bias_inference,
        'afl_prob_threshold': args.afl_prob_threshold,
        'afl_threshold_search_steps': args.afl_threshold_search_steps,
        'afl_threshold_search_min': args.afl_threshold_search_min,
        'afl_threshold_search_max': args.afl_threshold_search_max,
        'afl_min_recall_val': args.afl_min_recall_val,
        'enable_record_agg': args.enable_record_agg,
        'record_agg_method': args.record_agg_method,
        'record_agg_topk': args.record_agg_topk,
        'use_hard_mining': args.use_hard_mining,
        'hard_mining_start_epoch': args.hard_mining_start_epoch,
        'hard_mining_afl_fn_mult': args.hard_mining_afl_fn_mult,
        'hard_mining_af_fp_mult': args.hard_mining_af_fp_mult,
        'hard_mining_low_conf_mult': args.hard_mining_low_conf_mult,
        'hard_mining_conf_threshold': args.hard_mining_conf_threshold,
        'use_low_conf_focus': args.use_low_conf_focus,
        'low_conf_alpha': args.low_conf_alpha,
        'early_stop_patience': args.early_stop_patience,
        'early_stop_min_delta': args.early_stop_min_delta,
        'use_oversampling': args.use_oversampling,
        'oversample_target_ratio': args.oversample_target_ratio,
        'oversample_strategy': args.oversample_strategy,
        'recompute_fuzzy_only': args.recompute_fuzzy_only,
        'split_seed': args.split_seed,
        'require_predefined_split': not args.allow_random_resplit,
    }
    
    return args, train_kwargs


def main():
    args, train_kwargs = parse_args()
    
    # 确定使用哪个CSV文件
    if args.combined_csv:
        # 使用合并的CSV（推荐）
        if not os.path.exists(args.combined_csv):
            print(f"Error: Combined CSV not found: {args.combined_csv}")
            sys.exit(1)
        
        print(f"Using combined CSV: {args.combined_csv}")
        # 运行训练（使用合并CSV作为afdb_csv，ltafdb_csv为空）
        run_training_binary(
            args.combined_csv,
            "",  # 空字符串表示不使用ltafdb_csv
            args.output_dir,
            **train_kwargs
        )
    else:
        # 使用分离的CSV（向后兼容）
        if not args.afdb_csv:
            print("Error: Either --combined_csv or --afdb_csv must be provided")
            sys.exit(1)
            
        if not os.path.exists(args.afdb_csv):
            print(f"Error: AFDB CSV not found: {args.afdb_csv}")
            sys.exit(1)
        
        if args.ltafdb_csv and not os.path.exists(args.ltafdb_csv):
            print(f"Error: LTAFDB CSV not found: {args.ltafdb_csv}")
            sys.exit(1)
        
        # 运行训练
        run_training_binary(
            args.afdb_csv,
            args.ltafdb_csv,
            args.output_dir,
            **train_kwargs
        )


if __name__ == "__main__":
    main()
