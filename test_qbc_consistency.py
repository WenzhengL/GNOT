#!/usr/bin/env python3
"""
测试QBC策略的评估一致性
验证修复后的evaluate_model函数是否能产生稳定的误差矩阵
"""

import torch
import numpy as np
import pickle
import os
import sys
sys.path.append('/home/v-wenliao/gnot/GNOT')

from alqbc import train_model, evaluate_model, set_deterministic_training

def test_evaluation_consistency():
    """测试评估函数的一致性"""
    print("=== 测试QBC评估一致性 ===")
    
    # 设置确定性环境
    set_deterministic_training(42)
    
    # 加载测试数据
    data_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc"
    labeled_path = os.path.join(data_dir, 'al_labeled.pkl')
    test_path = os.path.join(data_dir, 'al_test.pkl')
    
    if not (os.path.exists(labeled_path) and os.path.exists(test_path)):
        print("❌ 测试数据不存在，跳过测试")
        return
    
    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # 使用少量数据进行快速测试
    labeled_data = labeled_data[:20]  # 只用20个样本训练
    test_data = test_data[:10]        # 只用10个样本测试
    
    print(f"使用数据: 训练={len(labeled_data)}, 测试={len(test_data)}")
    
    # 训练模型
    print("\n1. 训练模型...")
    model_tuple = train_model(labeled_data)
    model, metric_func, device = model_tuple
    
    # 多次评估，验证一致性
    print("\n2. 测试评估一致性...")
    metrics_list = []
    
    for i in range(5):
        print(f"  评估轮次 {i+1}...")
        
        # 重新设置种子确保数据加载一致
        set_deterministic_training(42)
        
        metric = evaluate_model(model_tuple, test_data)
        metrics_list.append(metric)
        print(f"    误差矩阵: {metric}")
    
    # 分析一致性
    print("\n3. 一致性分析:")
    metrics_array = np.array(metrics_list)
    
    print(f"平均误差: {np.mean(metrics_array, axis=0)}")
    print(f"标准差:   {np.std(metrics_array, axis=0)}")
    print(f"最大差异: {np.max(metrics_array, axis=0) - np.min(metrics_array, axis=0)}")
    
    # 判断是否一致
    max_std = np.max(np.std(metrics_array, axis=0))
    if max_std < 1e-6:
        print("✅ 评估结果完全一致，修复成功！")
    elif max_std < 1e-4:
        print("✅ 评估结果基本一致，修复有效")
    else:
        print(f"❌ 评估结果不一致，标准差={max_std:.8f}")
    
    # 4. 测试不同模型状态的影响
    print("\n4. 测试模型状态影响:")
    
    # 强制设置为train模式
    model.train()
    metric_train = evaluate_model(model_tuple, test_data)
    print(f"Train模式后评估: {metric_train}")
    
    # 强制设置为eval模式  
    model.eval()
    metric_eval = evaluate_model(model_tuple, test_data)
    print(f"Eval模式后评估:  {metric_eval}")
    
    # 比较差异
    diff = np.abs(metric_train - metric_eval)
    max_diff = np.max(diff)
    print(f"两种模式差异: {diff}")
    print(f"最大差异: {max_diff:.8f}")
    
    if max_diff < 1e-6:
        print("✅ 不同模式下评估结果一致")
    else:
        print(f"⚠️  不同模式下评估结果有差异，说明dropout仍在影响评估")

if __name__ == "__main__":
    test_evaluation_consistency()
