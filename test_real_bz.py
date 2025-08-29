#!/usr/bin/env python3
"""
测试更新后的实际BZ策略
"""

import sys
import os
sys.path.append('/home/v-wenliao/gnot/GNOT')

from bz_strategy_scale_fix import calculate_dimension_scales
import torch
import numpy as np

def test_real_bz_strategy():
    """测试真实的BZ策略实现"""
    print("=== 测试真实BZ策略实现 ===\n")
    
    # 创建模拟的unlabeled_data，格式与实际数据相同
    # 使用实际日志中的误差分布
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟MultipleTensors格式的数据
    class MockData:
        def __init__(self):
            # 模拟5个样本，每个样本有5个维度的值
            self.data = [
                torch.tensor([[693.24, 363.79, 10.65, 55.08, 71.06]], device=device),
                torch.tensor([[1315.96, 364.69, 11.13, 56.31, 72.77]], device=device),
                torch.tensor([[1265.22, 368.99, 10.47, 55.13, 68.76]], device=device),
                torch.tensor([[1284.40, 362.96, 11.43, 56.55, 73.44]], device=device),
                torch.tensor([[1205.76, 363.54, 11.10, 55.45, 74.08]], device=device),
            ]
    
    mock_unlabeled = [MockData() for _ in range(5)]
    
    # 测试计算缩放因子
    print("测试 calculate_dimension_scales 函数:")
    try:
        scales = calculate_dimension_scales(mock_unlabeled, sample_size=5, scaling_strategy='balanced')
        print(f"返回的缩放因子: {scales}")
        print(f"缩放因子类型: {type(scales)}")
        
        # 分析缩放效果
        if len(scales) == 5:
            print(f"\n缩放因子分析:")
            for i, scale in enumerate(scales):
                print(f"  维度 {i}: 权重 = {scale:.6f}")
            
            # 验证平衡效果
            print(f"\n平衡效果验证:")
            print(f"  最大权重: {np.max(scales):.6f}")
            print(f"  最小权重: {np.min(scales):.6f}") 
            print(f"  权重比例: {np.max(scales)/np.min(scales):.2f}")
            print(f"  权重标准差: {np.std(scales):.6f}")
            
            if np.max(scales)/np.min(scales) < 10:  # 权重差异小于10倍
                print("  ✓ 权重比例合理")
            else:
                print("  ✗ 权重差异过大")
                
        else:
            print(f"错误：返回缩放因子数量不正确，期望5个，实际{len(scales)}个")
            
    except Exception as e:
        print(f"测试失败：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_real_bz_strategy()
