#!/usr/bin/env python3
"""
快速测试PA预测误差计算

这个脚本用于测试PA策略的误差计算逻辑，不运行完整的主动学习流程
"""

import os
import sys
import numpy as np
import pickle

# 添加当前目录到路径
sys.path.append('/home/v-wenliao/gnot/GNOT')

def test_error_calculation():
    """测试误差计算逻辑"""
    
    print("="*60)
    print("测试PA预测误差计算逻辑")
    print("="*60)
    
    # 创建模拟数据
    print("1. 创建模拟数据...")
    
    # 模拟一个样本的坐标和真实值
    n_points = 100
    coords = np.random.rand(n_points, 3) * 10  # 坐标
    true_values = np.random.rand(n_points, 5) * 50 + 10  # 5个输出字段的真实值
    
    # 模拟预测值（添加一些误差）
    pred_values = true_values + np.random.normal(0, 1, true_values.shape)
    
    print(f"坐标形状: {coords.shape}")
    print(f"真实值形状: {true_values.shape}")
    print(f"预测值形状: {pred_values.shape}")
    print(f"真实值范围: [{true_values.min():.3f}, {true_values.max():.3f}]")
    print(f"预测值范围: [{pred_values.min():.3f}, {pred_values.max():.3f}]")
    
    # 测试误差计算
    print("\n2. 测试误差计算...")
    
    error_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    prediction_errors_per_field = []
    
    for field_idx in range(5):
        field_pred = pred_values[:, field_idx]
        field_true = true_values[:, field_idx]
        
        # 计算平均绝对误差 (MAE)
        mae = np.mean(np.abs(field_pred - field_true))
        
        # 计算相对误差 (避免除零)
        true_scale = np.mean(np.abs(field_true)) + 1e-8
        relative_error = mae / true_scale
        
        prediction_errors_per_field.append(relative_error)
        
        print(f"字段 {field_idx}:")
        print(f"  MAE: {mae:.6f}")
        print(f"  true_scale: {true_scale:.6f}")
        print(f"  relative_error: {relative_error:.6f}")
    
    # 计算总误差
    prediction_errors_array = np.array(prediction_errors_per_field)
    total_error = np.sum(prediction_errors_array * error_weights)
    
    print(f"\n各字段误差: {prediction_errors_per_field}")
    print(f"误差权重: {error_weights}")
    print(f"加权总误差: {total_error:.6f}")
    
    # 测试不同情况
    print("\n3. 测试边界情况...")
    
    # 情况1: 预测完全正确
    print("\n情况1: 预测完全正确")
    perfect_pred = true_values.copy()
    test_case_error(perfect_pred, true_values, error_weights, "完全正确")
    
    # 情况2: 预测完全错误 
    print("\n情况2: 预测完全错误")
    wrong_pred = true_values * 2  # 双倍的错误
    test_case_error(wrong_pred, true_values, error_weights, "完全错误")
    
    # 情况3: 小误差
    print("\n情况3: 小误差")
    small_error_pred = true_values + 0.01
    test_case_error(small_error_pred, true_values, error_weights, "小误差")


def test_case_error(pred_values, true_values, error_weights, case_name):
    """测试特定情况的误差计算"""
    
    prediction_errors_per_field = []
    
    for field_idx in range(5):
        field_pred = pred_values[:, field_idx]
        field_true = true_values[:, field_idx]
        
        mae = np.mean(np.abs(field_pred - field_true))
        true_scale = np.mean(np.abs(field_true)) + 1e-8
        relative_error = mae / true_scale
        
        prediction_errors_per_field.append(relative_error)
    
    prediction_errors_array = np.array(prediction_errors_per_field)
    total_error = np.sum(prediction_errors_array * error_weights)
    
    print(f"{case_name}:")
    print(f"  各字段误差: {[f'{e:.6f}' for e in prediction_errors_per_field]}")
    print(f"  总误差: {total_error:.6f}")
    print(f"  误差是否>0: {total_error > 0}")
    print(f"  误差是否>=0: {total_error >= 0}")


def test_data_loading():
    """测试数据加载"""
    
    print("\n4. 测试真实数据加载...")
    
    # 检查数据文件
    data_files = [
        '/home/v-wenliao/gnot/GNOT/data/al_pa/al_unlabeled.pkl',
        '/home/v-wenliao/gnot/GNOT/data/al_unlabeled.pkl'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"\n发现数据文件: {data_file}")
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                print(f"数据长度: {len(data)}")
                
                if len(data) > 0:
                    sample = data[0]
                    print(f"样本类型: {type(sample)}")
                    print(f"样本长度: {len(sample)}")
                    
                    if len(sample) >= 2:
                        coords = np.array(sample[0])
                        values = np.array(sample[1])
                        
                        print(f"坐标形状: {coords.shape}")
                        print(f"值形状: {values.shape}")
                        print(f"坐标范围: [{coords.min():.3f}, {coords.max():.3f}]")
                        print(f"值范围: [{values.min():.3f}, {values.max():.3f}]")
                        
                        # 检查是否有非零值
                        non_zero_values = values[values != 0]
                        print(f"非零值数量: {len(non_zero_values)}")
                        if len(non_zero_values) > 0:
                            print(f"非零值范围: [{non_zero_values.min():.3f}, {non_zero_values.max():.3f}]")
                
            except Exception as e:
                print(f"数据加载失败: {e}")
        else:
            print(f"数据文件不存在: {data_file}")


if __name__ == "__main__":
    print("开始测试PA预测误差计算...")
    
    test_error_calculation()
    test_data_loading()
    
    print("\n测试完成!")
    
    print("""
总结:
1. 如果所有误差都是0，可能是因为:
   - 预测值与真实值完全相同（模型过拟合）
   - 数据预处理问题
   - 误差计算逻辑问题

2. 建议修改策略:
   - 即使误差为0，也可以随机选择样本
   - 或者使用其他几何特征作为辅助
   - 检查数据是否正确加载

3. 调试步骤:
   - 查看前几个样本的详细误差计算过程
   - 确认预测值和真实值都正确加载
   - 检查数据预处理是否有问题
""")
