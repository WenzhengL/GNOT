#!/usr/bin/env python3
"""
测试基于预测误差的PA策略

用法:
python test_pa_prediction_error.py

这个脚本展示了如何使用新的基于预测误差的PA策略。
"""

import os
import sys
import numpy as np
import pickle

# 添加当前目录到路径
sys.path.append('/home/v-wenliao/gnot/GNOT')

from alpa import (
    prepare_active_learning_data, 
    train_model, 
    evaluate_model,
    pa_query_average_error,
    pa_query_prediction_error,
    pa_query_physics_based,
    active_learning_loop
)

def test_pa_strategies():
    """测试不同的PA策略"""
    
    print("="*60)
    print("测试基于预测误差的PA策略")
    print("="*60)
    
    # 配置参数
    data_dir = '/home/v-wenliao/gnot/GNOT/data/result'
    output_dir = '/home/v-wenliao/gnot/GNOT/data/al_pa_test'
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        return
    
    # 准备主动学习数据
    print("\n1. 准备主动学习数据...")
    try:
        prepare_active_learning_data(
            data_dir=data_dir,
            output_dir=output_dir,
            init_labeled_num=5,    # 初始标注样本数
            test_ratio=0.1,        # 测试集比例
            random_seed=42,
            overwrite=True         # 重新生成数据
        )
        print("✓ 数据准备完成")
    except Exception as e:
        print(f"✗ 数据准备失败: {e}")
        return
    
    # 运行主动学习流程测试
    print("\n2. 测试PA策略（基于预测误差）...")
    try:
        # 测试基于平均预测误差的PA策略（默认版本）
        active_learning_loop(
            dataset_name='al_pa_test',
            output_dir=output_dir,
            strategy='pa',           # 使用基于预测误差的PA策略
            rounds=2,               # 只运行2轮进行测试
            select_num=2,           # 每轮选择2个样本
            checkpoint_freq=1       # 每轮保存检查点
        )
        print("✓ PA预测误差策略测试完成")
    except Exception as e:
        print(f"✗ PA预测误差策略测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
    
    print("\n3. 测试结果分析...")
    try:
        # 检查输出文件
        result_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith(('.csv', '.json', '.pkl')):
                    result_files.append(os.path.join(root, file))
        
        print(f"生成的结果文件:")
        for file in result_files[:10]:  # 只显示前10个
            print(f"  - {file}")
        
        if len(result_files) > 10:
            print(f"  ... 还有 {len(result_files) - 10} 个文件")
            
    except Exception as e:
        print(f"结果分析失败: {e}")

def demonstrate_pa_usage():
    """演示PA策略的不同用法"""
    
    print("\n" + "="*60)
    print("PA策略使用说明")
    print("="*60)
    
    print("""
修改后的PA策略提供三种使用方式:

1. 默认版本 - 基于五个值的平均误差 (strategy='pa'):
   这是用户要求的默认版本，计算pressure, wall-shear, x-wall-shear, 
   y-wall-shear, z-wall-shear五个值的平均预测误差。

2. 自定义权重版本 (strategy='pa_custom'):
   可以为每个输出字段设置不同的权重，例如：
   - 如果更关心pressure的准确性，可以设置pressure_weight=2.0
   - 如果更关心wall-shear相关字段，可以增加这些字段的权重

3. 物理一致性版本 (strategy='pa_physics'):
   保留了原始的基于物理方程（Navier-Stokes）一致性的版本，用于对比研究。

使用示例:

# 基于平均预测误差的PA策略（推荐）
active_learning_loop(
    dataset_name='al_pa',
    output_dir='./output',
    strategy='pa',          # 使用默认的预测误差策略
    rounds=10,
    select_num=5
)

# 自定义权重的PA策略
active_learning_loop(
    dataset_name='al_pa', 
    output_dir='./output',
    strategy='pa_custom',   # 使用自定义权重策略
    rounds=10,
    select_num=5
)

主要特点:
- 基于真实的预测误差进行样本选择
- 优先选择模型预测误差最大的未标注样本
- 这些样本对提高模型性能最有帮助
- 支持灵活的权重配置

误差计算方法:
1. 对每个未标注样本进行预测
2. 计算预测值与真实值的平均绝对误差(MAE)
3. 使用相对误差进行归一化
4. 按设置的权重合并五个字段的误差
5. 选择总误差最大的样本

这种方法比原始的物理一致性方法更直接，计算效率更高，
并且直接针对模型的预测能力进行优化。
""")

if __name__ == "__main__":
    print("开始测试基于预测误差的PA策略...")
    
    # 显示使用说明
    demonstrate_pa_usage()
    
    # 运行测试
    test_pa_strategies()
    
    print(f"\n测试完成!")
    print(f"如果测试成功，你现在可以使用新的PA策略:")
    print(f"  - strategy='pa' : 基于平均预测误差（默认推荐）")
    print(f"  - strategy='pa_custom' : 基于自定义权重的预测误差")
    print(f"  - strategy='pa_physics' : 基于物理一致性（原版本）")
