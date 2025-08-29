#!/usr/bin/env python3
"""
简化版单样本误差测试 - 直接用模型预测
"""

import os
import sys
import pickle
import numpy as np
import torch

sys.path.append('/home/v-wenliao/gnot/GNOT')

def test_direct_sample_prediction():
    """直接测试单样本预测，不依赖数据加载器"""
    
    print("="*60)
    print("直接单样本预测测试")
    print("="*60)
    
    try:
        # 1. 加载已有的训练好的模型
        print("1. 尝试直接使用BZ策略中的模型...")
        
        # 检查是否有现成的model_tuple
        from alpa import get_al_args
        args = get_al_args()
        args.dataset = 'al_bz'
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 2. 加载未标注数据
        data_file = '/home/v-wenliao/gnot/GNOT/data/al_bz/al_unlabeled.pkl'
        if not os.path.exists(data_file):
            data_file = '/home/v-wenliao/gnot/GNOT/data/al_unlabeled.pkl'
        
        with open(data_file, 'rb') as f:
            unlabeled_data = pickle.load(f)
        
        print(f"加载了 {len(unlabeled_data)} 个未标注样本")
        
        # 3. 分析单个样本的原始数据
        print("\n3. 分析前5个样本的原始数据差异...")
        for i in range(min(5, len(unlabeled_data))):
            sample = unlabeled_data[i]
            
            coords = np.array(sample[0])  # 坐标
            values = np.array(sample[1])  # 真实值
            
            # 计算一些统计特征
            coord_mean = np.mean(coords, axis=0)
            coord_std = np.std(coords, axis=0)
            
            value_mean = np.mean(values, axis=0) if len(values.shape) > 1 else np.mean(values)
            value_std = np.std(values, axis=0) if len(values.shape) > 1 else np.std(values)
            
            print(f"样本 {i}:")
            print(f"  坐标形状: {coords.shape}")
            print(f"  值形状: {values.shape}")
            print(f"  坐标均值: {coord_mean}")
            print(f"  坐标标准差: {coord_std}")
            print(f"  值均值: {value_mean}")
            print(f"  值标准差: {value_std}")
            print(f"  点数: {len(coords)}")
            print()
        
        # 4. 尝试创建简单的误差度量
        print("4. 基于原始数据创建简单误差度量...")
        sample_complexity_scores = []
        
        for i in range(min(10, len(unlabeled_data))):
            sample = unlabeled_data[i]
            coords = np.array(sample[0])
            values = np.array(sample[1])
            
            # 几何复杂度
            coord_variance = np.var(coords)
            coord_range = np.max(coords) - np.min(coords)
            
            # 物理量复杂度
            if len(values.shape) > 1:
                value_variance = np.var(values)
                value_range = np.max(values) - np.min(values)
            else:
                value_variance = np.var(values)
                value_range = np.max(values) - np.min(values)
            
            # 组合复杂度评分
            complexity_score = (
                coord_variance * 10 +
                coord_range * 5 +
                value_variance * 3 +
                value_range * 1 +
                len(coords) * 0.001
            )
            
            sample_complexity_scores.append(complexity_score)
            
            print(f"样本 {i}: 复杂度评分 = {complexity_score:.6f}")
            print(f"  几何方差: {coord_variance:.6f}")
            print(f"  几何范围: {coord_range:.6f}")
            print(f"  值方差: {value_variance:.6f}")
            print(f"  值范围: {value_range:.6f}")
        
        print(f"\n复杂度评分统计:")
        print(f"  最小值: {min(sample_complexity_scores):.6f}")
        print(f"  最大值: {max(sample_complexity_scores):.6f}")
        print(f"  平均值: {np.mean(sample_complexity_scores):.6f}")
        print(f"  标准差: {np.std(sample_complexity_scores):.6f}")
        
        if np.std(sample_complexity_scores) > 1e-6:
            print("✅ 不同样本有不同的复杂度评分 - 可以用于主动学习选择!")
            
            # 选择复杂度最高的样本
            top_indices = np.argsort(sample_complexity_scores)[-3:]
            print(f"\n复杂度最高的3个样本索引: {top_indices}")
            print(f"对应的复杂度评分: {[sample_complexity_scores[i] for i in top_indices]}")
        else:
            print("❌ 所有样本的复杂度评分相同")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":
    test_direct_sample_prediction()
