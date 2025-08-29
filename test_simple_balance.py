#!/usr/bin/env python3
"""
简化测试BZ策略平衡效果
"""

import torch
import numpy as np

def test_simple_balance():
    """直接测试误差平衡算法"""
    print("=== 测试BZ策略误差平衡算法 ===\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟5个维度的典型误差（基于实际日志数据）
    # 这是单个样本的5个维度误差
    sample_errors = torch.tensor([
        [693.24, 363.79, 10.65, 55.08, 71.06],  # 样本1
        [1315.96, 364.69, 11.13, 56.31, 72.77], # 样本2  
        [1265.22, 368.99, 10.47, 55.13, 68.76], # 样本3
        [1284.40, 362.96, 11.43, 56.55, 73.44], # 样本4
        [1205.76, 363.54, 11.10, 55.45, 74.08], # 样本5
    ], device=device)
    
    print("原始误差分布 (5个样本, 5个维度):")
    for i in range(5):
        mean_err = sample_errors[:, i].mean().item()
        print(f"  维度 {i}: 均值={mean_err:.2f}")
    
    print(f"\n原始误差占比:")
    total_errors = sample_errors.sum(dim=1)
    for i in range(5):
        contrib = (sample_errors[:, i].sum() / sample_errors.sum()).item()
        print(f"  维度 {i}: {contrib*100:.1f}%")
    
    # 测试不同的平衡策略
    print("\n=== 测试不同平衡策略 ===")
    
    # 1. 原始方法（无缩放）
    print("\n1. 原始方法（无缩放）:")
    original_total = sample_errors.sum(dim=1)
    print(f"样本综合误差: {original_total.cpu().numpy()}")
    
    # 2. 倒数权重法
    print("\n2. 倒数权重法:")
    dim_means = sample_errors.mean(dim=0)
    inverse_weights = 1.0 / dim_means
    inverse_weights = inverse_weights / inverse_weights.sum() * 5  # 归一化
    print(f"倒数权重: {inverse_weights.cpu().numpy()}")
    weighted_errors_inv = sample_errors * inverse_weights
    inv_total = weighted_errors_inv.sum(dim=1)
    print(f"样本综合误差: {inv_total.cpu().numpy()}")
    
    # 计算各维度贡献
    print("各维度贡献占比:")
    for i in range(5):
        contrib = (weighted_errors_inv[:, i].sum() / weighted_errors_inv.sum()).item()
        print(f"  维度 {i}: {contrib*100:.1f}%")
    
    # 3. 强平衡法（目标等权重）
    print("\n3. 强平衡法（目标等权重）:")
    # 让每个维度的平均贡献都是20%
    target_weights = 1.0 / dim_means
    target_weights = target_weights / target_weights.max()  # 最大权重为1
    print(f"目标权重: {target_weights.cpu().numpy()}")
    
    weighted_errors_bal = sample_errors * target_weights
    bal_total = weighted_errors_bal.sum(dim=1)
    print(f"样本综合误差: {bal_total.cpu().numpy()}")
    
    # 计算各维度贡献
    print("各维度贡献占比:")
    for i in range(5):
        contrib = (weighted_errors_bal[:, i].sum() / weighted_errors_bal.sum()).item()
        print(f"  维度 {i}: {contrib*100:.1f}%")
    
    # 4. 极强平衡法（归一化）
    print("\n4. 极强平衡法（归一化+权重）:")
    # 先归一化，再应用小权重
    normalized_errors = sample_errors / dim_means
    strong_weights = torch.tensor([0.01, 1.0, 1.0, 1.0, 1.0], device=device)  # 压制pressure
    weighted_errors_strong = normalized_errors * strong_weights
    strong_total = weighted_errors_strong.sum(dim=1)
    print(f"样本综合误差: {strong_total.cpu().numpy()}")
    
    # 计算各维度贡献
    print("各维度贡献占比:")
    for i in range(5):
        contrib = (weighted_errors_strong[:, i].sum() / weighted_errors_strong.sum()).item()
        print(f"  维度 {i}: {contrib*100:.1f}%")
    
    # 评估最佳方法
    print("\n=== 平衡效果评估 ===")
    methods = ['原始', '倒数权重', '强平衡', '极强平衡']
    all_contributions = []
    
    # 计算各方法的贡献分布
    for method_errors in [sample_errors, weighted_errors_inv, weighted_errors_bal, weighted_errors_strong]:
        contribs = []
        for i in range(5):
            contrib = (method_errors[:, i].sum() / method_errors.sum()).item()
            contribs.append(contrib)
        all_contributions.append(contribs)
    
    for i, (method, contribs) in enumerate(zip(methods, all_contributions)):
        max_contrib = max(contribs)
        balance_score = np.std(contribs)
        print(f"{method}: 最大占比={max_contrib*100:.1f}%, 平衡得分={balance_score:.4f}")
        if max_contrib < 0.3:
            print(f"  评估: 优秀平衡 ✓")
        elif max_contrib < 0.5:
            print(f"  评估: 良好平衡")
        elif max_contrib < 0.8:
            print(f"  评估: 一般平衡")
        else:
            print(f"  评估: 不平衡 ✗")

if __name__ == "__main__":
    test_simple_balance()
