#!/usr/bin/env python3
"""
测试修复后的BZ缩放策略
"""

import numpy as np

def test_scaling_logic():
    """测试修复后的缩放逻辑"""
    print("=== 测试修复后的BZ缩放策略 ===\n")
    
    # 模拟各维度平均误差（真实数据的比例）
    dim_means = np.array([577.26, 191.47, 5.55, 22.38, 89.99])  # 从日志中提取
    print(f"各维度平均误差: {dim_means}")
    
    # 计算倒数权重（balanced策略）
    inverse_weights = 1.0 / (dim_means + 1e-6)
    scales = inverse_weights / inverse_weights.sum() * 5.0
    print(f"倒数权重（归一化）: {scales}")
    
    # 模拟一个样本的原始误差
    raw_errors = np.array([1216.55, 187.27, 5.88, 22.44, 89.79])  # 样本195的误差
    print(f"\n样本原始误差: {raw_errors}")
    
    # 修复前的错误计算（倒数权重）
    old_weighted = raw_errors * (1.0 / scales)
    print(f"修复前加权误差 (错误方法): {old_weighted}")
    old_ratios = old_weighted / old_weighted.sum() * 100
    print(f"修复前各维度占比: [{old_ratios[0]:.1f}%, {old_ratios[1]:.1f}%, {old_ratios[2]:.1f}%, {old_ratios[3]:.1f}%, {old_ratios[4]:.1f}%]")
    
    # 修复后的正确计算（直接用权重）
    new_weighted = raw_errors * scales
    print(f"\n修复后加权误差 (正确方法): {new_weighted}")
    new_ratios = new_weighted / new_weighted.sum() * 100
    print(f"修复后各维度占比: [{new_ratios[0]:.1f}%, {new_ratios[1]:.1f}%, {new_ratios[2]:.1f}%, {new_ratios[3]:.1f}%, {new_ratios[4]:.1f}%]")
    
    print(f"\n=== 对比结果 ===")
    print(f"修复前总误差: {old_weighted.sum():.2f}")
    print(f"修复后总误差: {new_weighted.sum():.2f}")
    
    # 各维度占比分析
    print(f"\n维度占比对比:")
    for i in range(5):
        print(f"维度 {i}: 修复前 {old_ratios[i]:.1f}% -> 修复后 {new_ratios[i]:.1f}%")
    
    # 判断平衡性
    new_std = np.std(new_ratios)
    old_std = np.std(old_ratios)
    
    print(f"\n平衡性分析:")
    print(f"修复前标准差: {old_std:.2f}%")
    print(f"修复后标准差: {new_std:.2f}%")
    print(f"改善效果: {((old_std - new_std) / old_std * 100):.1f}%")
    
    if new_std < 10:
        print("✅ 平衡性优秀 (标准差 < 10%)")
    elif new_std < 20:
        print("✅ 平衡性良好 (标准差 < 20%)")
    else:
        print("❌ 仍需改进")

if __name__ == "__main__":
    test_scaling_logic()
