#!/usr/bin/env python3
"""
测试QBC策略中dropout是否正常工作
"""

import torch
import torch.nn as nn
import numpy as np

def test_basic_dropout():
    """测试基础dropout功能"""
    print("=== 测试基础dropout功能 ===")
    
    # 创建一个简单的带dropout的模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Dropout(p=0.5),
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    # 测试输入
    x = torch.randn(1, 10)
    
    # 在训练模式下多次前向传播
    model.train()
    predictions = []
    
    with torch.no_grad():
        for i in range(10):
            pred = model(x)
            predictions.append(pred.numpy())
    
    predictions = np.array(predictions)
    variance = np.var(predictions, axis=0)
    mean_variance = np.mean(variance)
    
    print(f"训练模式下：")
    print(f"  10次预测的平均方差: {mean_variance:.8f}")
    print(f"  dropout工作状态: {'正常' if mean_variance > 1e-6 else '异常'}")
    
    # 在评估模式下测试
    model.eval()
    predictions_eval = []
    
    with torch.no_grad():
        for i in range(10):
            pred = model(x)
            predictions_eval.append(pred.numpy())
    
    predictions_eval = np.array(predictions_eval)
    variance_eval = np.var(predictions_eval, axis=0)
    mean_variance_eval = np.mean(variance_eval)
    
    print(f"评估模式下：")
    print(f"  10次预测的平均方差: {mean_variance_eval:.8f}")
    print(f"  dropout工作状态: {'正确关闭' if mean_variance_eval < 1e-8 else '异常'}")
    
    return mean_variance > 1e-6 and mean_variance_eval < 1e-8

def test_dropout_with_different_p():
    """测试不同dropout概率的效果"""
    print("\n=== 测试不同dropout概率 ===")
    
    dropout_probs = [0.0, 0.1, 0.3, 0.5]
    x = torch.randn(1, 20)
    
    for p in dropout_probs:
        model = nn.Sequential(
            nn.Linear(20, 50),
            nn.Dropout(p=p),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
        
        model.train()
        predictions = []
        
        with torch.no_grad():
            for i in range(20):
                pred = model(x)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)
        variance = np.var(predictions, axis=0)
        mean_variance = np.mean(variance)
        
        print(f"Dropout p={p}: 平均方差={mean_variance:.8f}")

def test_force_enable_dropout():
    """测试强制启用dropout"""
    print("\n=== 测试强制启用dropout ===")
    
    # 创建dropout概率为0的模型
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.Dropout(p=0.0),  # 初始为0
        nn.ReLU(),
        nn.Linear(20, 5)
    )
    
    x = torch.randn(1, 10)
    
    # 测试初始状态（p=0）
    model.train()
    predictions_before = []
    
    with torch.no_grad():
        for i in range(10):
            pred = model(x)
            predictions_before.append(pred.numpy())
    
    variance_before = np.var(np.array(predictions_before), axis=0)
    mean_variance_before = np.mean(variance_before)
    
    print(f"强制启用前 (p=0): 平均方差={mean_variance_before:.8f}")
    
    # 强制启用dropout
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.3
            print(f"已将dropout概率调整为: {module.p}")
    
    # 测试修改后的状态
    predictions_after = []
    
    with torch.no_grad():
        for i in range(10):
            pred = model(x)
            predictions_after.append(pred.numpy())
    
    variance_after = np.var(np.array(predictions_after), axis=0)
    mean_variance_after = np.mean(variance_after)
    
    print(f"强制启用后 (p=0.3): 平均方差={mean_variance_after:.8f}")
    print(f"改进效果: {mean_variance_after / max(mean_variance_before, 1e-10):.2f}倍")
    
    return mean_variance_after > mean_variance_before * 100

if __name__ == "__main__":
    print("QBC Dropout 测试开始\n")
    
    # 运行所有测试
    test1_passed = test_basic_dropout()
    test_dropout_with_different_p()
    test2_passed = test_force_enable_dropout()
    
    print("\n" + "="*50)
    print("测试总结:")
    print(f"基础dropout测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"强制启用测试: {'✅ 通过' if test2_passed else '❌ 失败'}")
    
    if test1_passed and test2_passed:
        print("🎉 所有测试通过！QBC策略应该能正常工作")
    else:
        print("⚠️  存在问题，需要进一步调试")
