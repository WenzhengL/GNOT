#!/usr/bin/env python3
"""
分析当前BZ策略中各维度误差的分布和缩放需求

快速分析脚本，帮助理解为什么需要维度缩放
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

# 添加当前目录到path
sys.path.append('/home/v-wenliao/gnot/GNOT')


def analyze_dimension_errors():
    """分析各维度误差的典型范围"""
    print("=== 分析维度误差分布 ===")
    
    # 加载数据
    data_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"
    unlabeled_path = os.path.join(data_dir, 'al_unlabeled.pkl')
    
    if not os.path.exists(unlabeled_path):
        print(f"数据文件不存在: {unlabeled_path}")
        return
    
    with open(unlabeled_path, 'rb') as f:
        unlabeled_data = pickle.load(f)
    
    print(f"加载了 {len(unlabeled_data)} 个未标注样本")
    
    # 采样分析
    sample_size = min(20, len(unlabeled_data))
    sample_indices = np.random.choice(len(unlabeled_data), sample_size, replace=False)
    
    dimension_names = ['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']
    all_stats = []
    
    print(f"\n分析前 {sample_size} 个样本的各维度数据特征:")
    print(f"{'样本':<6} {'维度':<12} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} {'范围':<12}")
    print("-" * 90)
    
    for idx in sample_indices:
        try:
            sample = unlabeled_data[idx]
            Y_true = np.array(sample[1])  # 真实值 [N, 5]
            
            if len(Y_true) == 0 or Y_true.shape[1] < 5:
                continue
            
            for dim in range(5):
                values = Y_true[:, dim]
                
                stats = {
                    'sample_idx': idx,
                    'dimension': dimension_names[dim],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values),
                    'mean_abs': np.mean(np.abs(values))
                }
                
                all_stats.append(stats)
                
                print(f"{idx:<6} {dimension_names[dim]:<12} {stats['mean']:<12.4f} "
                      f"{stats['std']:<12.4f} {stats['min']:<12.4f} "
                      f"{stats['max']:<12.4f} {stats['range']:<12.4f}")
        
        except Exception as e:
            print(f"样本 {idx} 处理失败: {e}")
            continue
    
    # 汇总统计
    if all_stats:
        df = pd.DataFrame(all_stats)
        
        print(f"\n=== 各维度汇总统计 ===")
        print(f"{'维度':<15} {'均值范围':<20} {'标准差范围':<20} {'数值范围':<20} {'建议缩放因子':<15}")
        print("-" * 100)
        
        suggested_scales = []
        
        for dim_name in dimension_names:
            dim_data = df[df['dimension'] == dim_name]
            
            if len(dim_data) > 0:
                mean_range = f"[{dim_data['mean'].min():.2f}, {dim_data['mean'].max():.2f}]"
                std_range = f"[{dim_data['std'].min():.2f}, {dim_data['std'].max():.2f}]"
                value_range = f"[{dim_data['min'].min():.2f}, {dim_data['max'].max():.2f}]"
                
                # 计算建议的缩放因子（基于典型值的大小）
                typical_scale = dim_data['std'].mean() + dim_data['mean_abs'].mean()
                
                suggested_scales.append(typical_scale)
                
                print(f"{dim_name:<15} {mean_range:<20} {std_range:<20} {value_range:<20} {typical_scale:<15.4f}")
        
        # 标准化缩放因子
        if suggested_scales:
            max_scale = max(suggested_scales)
            normalized_scales = [s / max_scale for s in suggested_scales]
            
            print(f"\n=== 建议的维度权重 ===")
            print("基于数据分布的建议权重（1.0为最大权重）:")
            for i, (dim_name, weight) in enumerate(zip(dimension_names, normalized_scales)):
                print(f"  {dim_name}: {weight:.4f}")
            
            print(f"\n手动权重建议（用于BZ策略）:")
            print(f"manual_weights = {[round(w, 3) for w in normalized_scales]}")
            
            # 特别针对pressure维度的建议
            pressure_weight = normalized_scales[0]
            if pressure_weight > 0.5:
                reduced_pressure = 0.1
                adjusted_weights = normalized_scales.copy()
                adjusted_weights[0] = reduced_pressure
                
                print(f"\n考虑到pressure维度可能主导误差，建议降低其权重:")
                print(f"调整后权重 = {[round(w, 3) for w in adjusted_weights]}")
                
                print(f"\n对应的代码:")
                print(f"# 原始权重（基于数据分布）")
                print(f"original_weights = {[round(w, 3) for w in normalized_scales]}")
                print(f"# 调整后权重（降低pressure影响）")
                print(f"adjusted_weights = {[round(w, 3) for w in adjusted_weights]}")


def simulate_error_scaling():
    """模拟误差缩放的效果"""
    print(f"\n=== 模拟误差缩放效果 ===")
    
    # 模拟典型的各维度误差（基于日志观察）
    typical_errors = np.array([
        281.529999,  # pressure (维度0)
        1.273253,    # wall-shear (维度1)
        0.484378,    # x-wall-shear (维度2)
        0.204999,    # y-wall-shear (维度3)
        0.544602     # z-wall-shear (维度4)
    ])
    
    dimension_names = ['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']
    
    print("原始误差:")
    for i, (name, error) in enumerate(zip(dimension_names, typical_errors)):
        print(f"  {name}: {error:.6f}")
    
    print(f"原始总误差: {np.sum(typical_errors):.6f}")
    print(f"pressure占比: {typical_errors[0]/np.sum(typical_errors)*100:.1f}%")
    
    # 方法1: 基于数值大小的缩放
    scales_by_magnitude = typical_errors / np.max(typical_errors)
    scaled_errors_1 = typical_errors / (typical_errors + 1e-6)  # 归一化缩放
    
    print(f"\n方法1 - 归一化缩放:")
    for i, (name, orig, scaled) in enumerate(zip(dimension_names, typical_errors, scaled_errors_1)):
        print(f"  {name}: {orig:.6f} -> {scaled:.6f}")
    print(f"归一化总误差: {np.sum(scaled_errors_1):.6f}")
    print(f"pressure占比: {scaled_errors_1[0]/np.sum(scaled_errors_1)*100:.1f}%")
    
    # 方法2: 手动权重
    manual_weights = np.array([0.1, 1.0, 1.0, 1.0, 1.0])  # 降低pressure权重
    scaled_errors_2 = typical_errors * manual_weights
    
    print(f"\n方法2 - 手动权重 {manual_weights}:")
    for i, (name, orig, scaled) in enumerate(zip(dimension_names, typical_errors, scaled_errors_2)):
        print(f"  {name}: {orig:.6f} -> {scaled:.6f}")
    print(f"手动权重总误差: {np.sum(scaled_errors_2):.6f}")
    print(f"pressure占比: {scaled_errors_2[0]/np.sum(scaled_errors_2)*100:.1f}%")
    
    # 方法3: 平方根缩放
    sqrt_scaled_errors = np.sqrt(typical_errors)
    
    print(f"\n方法3 - 平方根缩放:")
    for i, (name, orig, scaled) in enumerate(zip(dimension_names, typical_errors, sqrt_scaled_errors)):
        print(f"  {name}: {orig:.6f} -> {scaled:.6f}")
    print(f"平方根总误差: {np.sum(sqrt_scaled_errors):.6f}")
    print(f"pressure占比: {sqrt_scaled_errors[0]/np.sum(sqrt_scaled_errors)*100:.1f}%")


def main():
    """主函数"""
    print("BZ策略维度误差分析")
    print("=" * 50)
    
    # 分析实际数据的维度分布
    analyze_dimension_errors()
    
    # 模拟缩放效果
    simulate_error_scaling()
    
    print(f"\n=== 总结和建议 ===")
    print("1. 从日志可以看出，pressure维度的误差通常比其他维度大1-2个数量级")
    print("2. 这导致BZ策略几乎完全由pressure误差主导")
    print("3. 建议使用以下缩放策略之一:")
    print("   - 自适应缩放：根据数据分布自动计算缩放因子")
    print("   - 手动权重：manual_weights = [0.1, 1.0, 1.0, 1.0, 1.0]")
    print("   - 平方根缩放：对误差取平方根降低大值的影响")
    print("4. 在主动学习中使用 'bz_scaled' 或 'bz_manual' 策略替代原始 'bz' 策略")


if __name__ == "__main__":
    main()
