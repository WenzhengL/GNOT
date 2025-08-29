#!/usr/bin/env python3
"""
测试更新后的BZ策略平衡效果
"""

import torch
import numpy as np
from bz_strategy_scale_fix import calculate_dimension_scales, bz_query_with_dimension_scaling

def test_balance_simulation():
    """模拟测试不同维度误差的平衡效果"""
    print("=== 测试更新后的BZ策略平衡效果 ===\n")
    
    # 模拟典型的误差分布（基于实际日志数据）
    n_samples = 100
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟5个维度的误差，维度0（pressure）误差通常很大
    errors = torch.zeros((n_samples, 5), device=device)
    
    # 维度0: pressure - 大误差
    errors[:, 0] = torch.normal(700, 300, (n_samples,)).clamp(min=100)
    # 维度1: wall-shear - 中等误差  
    errors[:, 1] = torch.normal(360, 3, (n_samples,)).clamp(min=350)
    # 维度2: x-wall-shear - 小误差
    errors[:, 2] = torch.normal(10.5, 0.4, (n_samples,)).clamp(min=9)
    # 维度3: y-wall-shear - 小误差
    errors[:, 3] = torch.normal(55, 0.6, (n_samples,)).clamp(min=53)
    # 维度4: z-wall-shear - 小误差
    errors[:, 4] = torch.normal(71, 1.5, (n_samples,)).clamp(min=66)
    
    print("原始误差分布:")
    for i in range(5):
        mean_err = errors[:, i].mean().item()
        std_err = errors[:, i].std().item()
        print(f"  维度 {i}: 均值={mean_err:.2f}, 标准差={std_err:.2f}")
    
    # 测试不同的平衡策略
    strategies = ['adaptive', 'inverse', 'balanced', 'strong_balanced']
    
    for strategy in strategies:
        print(f"\n=== 测试策略: {strategy} ===")
        
        # 计算缩放因子
        scales = calculate_dimension_scales(errors, scaling_strategy=strategy)
        print(f"缩放因子: {scales.cpu().numpy()}")
        
        # 应用缩放
        if strategy == 'strong_balanced':
            # 强平衡策略：目标是让各维度贡献相等
            scaled_errors = errors / errors.mean(dim=0, keepdim=True) * scales
        else:
            scaled_errors = errors * scales
        
        # 计算各维度的贡献占比
        total_scaled = scaled_errors.sum(dim=1)
        dim_contributions = scaled_errors.sum(dim=0) / total_scaled.sum()
        
        print("各维度贡献占比:")
        for i, contrib in enumerate(dim_contributions):
            print(f"  维度 {i}: {contrib.item()*100:.1f}%")
        
        # 计算不平衡度（标准差越小越平衡）
        balance_score = dim_contributions.std().item()
        print(f"平衡得分（越小越好）: {balance_score:.4f}")
        
        # 评估平衡效果
        max_contrib = dim_contributions.max().item()
        if max_contrib < 0.3:  # 最大贡献小于30%
            print("评估: 优秀平衡 ✓")
        elif max_contrib < 0.5:  # 最大贡献小于50%
            print("评估: 良好平衡")
        elif max_contrib < 0.8:  # 最大贡献小于80%
            print("评估: 一般平衡")
        else:
            print("评估: 不平衡 ✗")

if __name__ == "__main__":
    test_balance_simulation()
