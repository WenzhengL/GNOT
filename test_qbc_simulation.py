#!/usr/bin/env python3
"""
模拟真实QBC查询情况的dropout测试
"""

import torch
import numpy as np
import pickle
import os
import sys
import time
import random
sys.path.append('/home/v-wenliao/gnot/GNOT')

from alqbc import get_al_args, set_deterministic_training
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from train import validate_epoch

def simulate_qbc_dropout():
    """模拟QBC查询中的dropout行为"""
    print("=== 模拟QBC Dropout行为测试 ===")
    
    # 设置确定性环境
    set_deterministic_training(42)
    
    # 获取参数和模型
    args = get_al_args()
    args.dataset = 'al_qbc'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据集和创建模型
    _, test_dataset = get_dataset(args)
    model = get_model(args).to(device)
    metric_func = get_loss_func('rel2', args, regularizer=False, normalizer=test_dataset.y_normalizer)
    
    # 准备测试数据
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    data_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc"
    test_path = os.path.join(data_dir, 'al_test.pkl')
    
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    # 只用一个样本进行测试
    single_sample = [test_data[0]]
    
    temp_test_file = os.path.join(temp_data_dir, 'al_test.pkl')
    with open(temp_test_file, 'wb') as f:
        pickle.dump(single_sample, f)
    
    _, test_dataset = get_dataset(args)
    test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    
    print(f"使用设备: {device}")
    print(f"测试样本数: {len(single_sample)}")
    
    # 强制启用dropout
    for name, module in model.named_modules():
        if hasattr(module, 'p') and 'drop' in name.lower():
            module.p = 0.5  # 使用更高的dropout概率便于观察差异
            print(f"设置 {name} dropout_p = 0.5")
    
    print("\n=== 测试1: 使用no_grad() (模拟当前evaluate_model) ===")
    model.train()  # 激活dropout
    results_no_grad = []
    
    for i in range(5):
        with torch.no_grad():
            for g, u_p, g_u in test_loader:
                g = g.to(device)
                u_p = u_p.to(device) 
                g_u = g_u.to(device)
                
                pred = model(g, u_p, g_u).cpu().numpy()
                results_no_grad.append(pred.copy())
                break
        print(f"  no_grad轮次{i+1}: 平均值={np.mean(pred):.6f}")
    
    print("\n=== 测试2: 不使用no_grad() (模拟QBC查询) ===")
    model.train()  # 激活dropout
    results_grad = []
    
    for i in range(5):
        # 不使用no_grad，但也不计算梯度
        for g, u_p, g_u in test_loader:
            g = g.to(device)
            u_p = u_p.to(device)
            g_u = g_u.to(device)
            
            pred = model(g, u_p, g_u).detach().cpu().numpy()
            results_grad.append(pred.copy())
            break
        print(f"  有grad轮次{i+1}: 平均值={np.mean(pred):.6f}")
    
    print("\n=== 测试3: eval模式 ===")
    model.eval()  # 关闭dropout
    results_eval = []
    
    for i in range(5):
        with torch.no_grad():
            for g, u_p, g_u in test_loader:
                g = g.to(device)
                u_p = u_p.to(device)
                g_u = g_u.to(device)
                
                pred = model(g, u_p, g_u).cpu().numpy()
                results_eval.append(pred.copy())
                break
        print(f"  eval轮次{i+1}: 平均值={np.mean(pred):.6f}")
    
    # 分析结果
    print("\n=== 方差分析 ===")
    
    for name, results in [("no_grad", results_no_grad), ("有grad", results_grad), ("eval", results_eval)]:
        if results:
            results_array = np.array(results)
            variance = np.var(results_array)
            std = np.std(results_array)
            print(f"{name}模式:")
            print(f"  形状: {results_array.shape}")
            print(f"  总体方差: {variance:.10f}")
            print(f"  总体标准差: {std:.10f}")
            print(f"  样本间最大差异: {np.max(results_array) - np.min(results_array):.10f}")
    
    # 最重要的测试：模拟实际QBC行为
    print("\n=== 测试4: 完全模拟QBC查询行为 ===")
    
    def simulate_mc_sampling(mc_times=10):
        """完全模拟QBC中的Monte Carlo采样"""
        model.train()  # 确保dropout激活
        
        preds = []
        for mc_iter in range(mc_times):
            # 每次都重新设置不同的随机状态
            torch.manual_seed(42 + mc_iter)  # 不同的seed
            
            with torch.no_grad():
                for g, u_p, g_u in test_loader:
                    g = g.to(device)
                    u_p = u_p.to(device)
                    g_u = g_u.to(device)
                    
                    pred = model(g, u_p, g_u).cpu().numpy()
                    preds.append(pred.copy())
                    break
        
        return np.array(preds)
    
    mc_results = simulate_mc_sampling(10)
    mc_variance = np.var(mc_results, axis=0)
    mean_mc_var = np.mean(mc_variance)
    
    print(f"MC采样结果:")
    print(f"  采样次数: {len(mc_results)}")
    print(f"  逐点方差平均值: {mean_mc_var:.10f}")
    print(f"  逐点方差最大值: {np.max(mc_variance):.10f}")
    print(f"  样本间最大差异: {np.max(mc_results) - np.min(mc_results):.10f}")
    
    if mean_mc_var > 1e-8:
        print("✅ MC采样产生了显著方差，dropout正常工作")
    else:
        print("❌ MC采样方差极小，dropout可能未正常工作")
    
    # 清理
    try:
        os.remove(temp_test_file)
    except:
        pass

if __name__ == "__main__":
    simulate_qbc_dropout()
