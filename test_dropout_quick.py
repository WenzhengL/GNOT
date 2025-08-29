#!/usr/bin/env python3
"""
快速测试QBC策略的评估一致性
直接测试evaluate_model函数的dropout影响
"""

import torch
import numpy as np
import pickle
import os
import sys
import time
sys.path.append('/home/v-wenliao/gnot/GNOT')

from alqbc import get_al_args, set_deterministic_training
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from train import validate_epoch

def quick_dropout_test():
    """快速测试dropout对评估的影响"""
    print("=== 快速Dropout影响测试 ===")
    
    # 设置确定性环境
    set_deterministic_training(42)
    
    # 获取参数和模型
    args = get_al_args()
    args.dataset = 'al_qbc'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"使用设备: {device}")
    
    # 先加载数据集以获得配置
    _, test_dataset = get_dataset(args)
    
    # 创建模型
    model = get_model(args).to(device)
    print("模型创建完成")
    
    # 强制启用dropout
    def force_enable_dropout(model, dropout_p=0.15):
        dropout_count = 0
        for name, module in model.named_modules():
            if hasattr(module, 'dropout') or 'drop' in name.lower():
                if hasattr(module, 'p'):
                    module.p = dropout_p
                    dropout_count += 1
                    print(f"  设置 {name} dropout_p = {dropout_p}")
        return dropout_count
    
    dropout_count = force_enable_dropout(model, 0.15)
    print(f"处理了 {dropout_count} 个dropout层")
    
    # 加载测试数据
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    data_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc"
    test_path = os.path.join(data_dir, 'al_test.pkl')
    
    if not os.path.exists(test_path):
        print(f"❌ 测试数据不存在: {test_path}")
        return
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # 只取前5个样本进行快速测试
    test_data = test_data[:5]
    print(f"使用 {len(test_data)} 个测试样本")
    
    # 准备测试数据文件
    temp_test_file = os.path.join(temp_data_dir, 'al_test.pkl')
    with open(temp_test_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    # 获取数据集和loader (已经加载过了，直接使用)
    test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    metric_func = get_loss_func('rel2', args, regularizer=False, normalizer=test_dataset.y_normalizer)
    
    print("\n=== 测试不同模型模式的影响 ===")
    
    results = {}
    
    # 1. 测试train模式 (dropout激活)
    print("1. 测试train模式 (dropout激活)...")
    train_results = []
    for i in range(3):
        model.train()  # 激活dropout
        with torch.no_grad():  # 不计算梯度但保持dropout
            val_result = validate_epoch(model, metric_func, test_loader, device)
            metric = val_result["metric"]
            train_results.append(metric)
            print(f"  轮次{i+1}: {metric}")
    
    results['train'] = np.array(train_results)
    
    # 2. 测试eval模式 (dropout关闭)
    print("\n2. 测试eval模式 (dropout关闭)...")
    eval_results = []
    for i in range(3):
        model.eval()  # 关闭dropout
        with torch.no_grad():
            val_result = validate_epoch(model, metric_func, test_loader, device)
            metric = val_result["metric"]
            eval_results.append(metric)
            print(f"  轮次{i+1}: {metric}")
    
    results['eval'] = np.array(eval_results)
    
    # 3. 分析结果
    print("\n=== 结果分析 ===")
    
    for mode, data in results.items():
        print(f"\n{mode.upper()}模式:")
        print(f"  平均误差: {np.mean(data, axis=0)}")
        print(f"  标准差:   {np.std(data, axis=0)}")
        print(f"  最大标准差: {np.max(np.std(data, axis=0)):.8f}")
    
    # 比较两种模式
    train_std = np.max(np.std(results['train'], axis=0))
    eval_std = np.max(np.std(results['eval'], axis=0))
    
    print(f"\n=== 关键对比 ===")
    print(f"Train模式最大标准差: {train_std:.8f}")
    print(f"Eval模式最大标准差:  {eval_std:.8f}")
    
    if train_std > 1e-6:
        print("❌ Train模式下结果不一致，dropout正在影响评估")
    else:
        print("✅ Train模式下结果一致")
    
    if eval_std < 1e-8:
        print("✅ Eval模式下结果完全一致，dropout已关闭")
    else:
        print("⚠️  Eval模式下结果略有差异")
    
    # 模式间差异
    mean_train = np.mean(results['train'], axis=0)
    mean_eval = np.mean(results['eval'], axis=0)
    mode_diff = np.abs(mean_train - mean_eval)
    max_mode_diff = np.max(mode_diff)
    
    print(f"\n两种模式间平均差异: {mode_diff}")
    print(f"最大模式差异: {max_mode_diff:.8f}")
    
    if max_mode_diff > 1e-6:
        print("❌ 两种模式结果差异明显，dropout影响显著")
        print("   建议: 在evaluate_model中明确使用model.eval()")
    else:
        print("✅ 两种模式结果一致，dropout控制正常")
    
    # 清理
    try:
        os.remove(temp_test_file)
    except:
        pass

if __name__ == "__main__":
    quick_dropout_test()
