#!/usr/bin/env python3
"""
测试可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import os

def test_visualization():
    """测试可视化功能"""
    print("=== 测试可视化功能 ===")
    
    # 创建模拟数据
    epochs = list(range(1, 51))  # 50个epoch
    train_losses = [1.0 * np.exp(-0.05 * i) + 0.1 + 0.02 * np.random.randn() for i in epochs]
    val_losses = [0.8 * np.exp(-0.04 * i) + 0.15 + 0.03 * np.random.randn() for i in epochs]
    
    # 测试训练loss可视化
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.title('Training Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    loss_diff = np.diff(train_losses)
    plt.plot(epochs[1:], loss_diff, 'g-', label='Loss Change per Epoch', linewidth=2)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    plt.title('Loss Change Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Difference')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_training_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 训练可视化测试完成：test_training_visualization.png")
    
    # 测试主动学习进度可视化
    al_data = {
        'pressure': [0.25, 0.22, 0.20, 0.18, 0.16],
        'wall-shear': [0.30, 0.28, 0.25, 0.23, 0.21],
        'x-wall-shear': [0.35, 0.32, 0.29, 0.26, 0.24],
        'y-wall-shear': [0.33, 0.30, 0.28, 0.25, 0.23],
        'z-wall-shear': [0.38, 0.35, 0.32, 0.29, 0.27],
        'train_num': [100, 200, 300, 400, 500],
        'round': [1, 2, 3, 4, 5]
    }
    
    df = pd.DataFrame(al_data)
    
    plt.figure(figsize=(15, 10))
    
    metrics = ['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']
    metric_labels = ['Pressure', 'Wall Shear', 'X-Wall Shear', 'Y-Wall Shear', 'Z-Wall Shear']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        plt.subplot(2, 3, i+1)
        plt.plot(df['train_num'], df[metric], 'o-', linewidth=2, markersize=6, label=label)
        plt.title(f'{label} Error vs Training Samples')
        plt.xlabel('Training Samples')
        plt.ylabel('Error')
        plt.grid(True, alpha=0.3)
        plt.legend()
    
    plt.subplot(2, 3, 6)
    avg_error = df[metrics].mean(axis=1)
    plt.plot(df['train_num'], avg_error, 'ro-', linewidth=3, markersize=8, label='Average Error')
    plt.title('Average Error vs Training Samples')
    plt.xlabel('Training Samples')
    plt.ylabel('Average Error')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 趋势线
    z = np.polyfit(df['train_num'], avg_error, 1)
    p = np.poly1d(z)
    plt.plot(df['train_num'], p(df['train_num']), "r--", alpha=0.8, label=f'Trend (slope={z[0]:.6f})')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('test_active_learning_visualization.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 主动学习可视化测试完成：test_active_learning_visualization.png")
    
    # 保存CSV数据
    df.to_csv('test_metrics.csv', index=False)
    print("✅ 测试数据已保存：test_metrics.csv")

if __name__ == "__main__":
    print("可视化功能测试开始\n")
    
    try:
        test_visualization()
        print("\n🎉 所有可视化测试通过！")
        print("\n生成的文件:")
        print("  📊 test_training_visualization.png - 训练过程可视化")
        print("  📈 test_active_learning_visualization.png - 主动学习进度可视化")
        print("  📋 test_metrics.csv - 测试数据")
        
    except Exception as e:
        print(f"\n❌ 可视化测试失败: {e}")
        import traceback
        traceback.print_exc()
