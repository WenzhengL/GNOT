#!/usr/bin/env python3
"""
测试BZ策略

BZ策略直接使用GNOT的validate_epoch函数来评估未标注样本，
选择模型表现最差（metric最大）的样本
"""

import os
import sys
sys.path.append('/home/v-wenliao/gnot/GNOT')

def test_bz_strategy():
    """测试BZ策略"""
    
    print("="*60)
    print("测试BZ策略（基于GNOT模型测试框架）")
    print("="*60)
    
    # 配置参数
    data_dir = '/home/v-wenliao/gnot/GNOT/data/result'
    output_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz_test'
    
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查数据目录是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据目录 {data_dir} 不存在")
        return
    
    try:
        from alpa import (
            prepare_active_learning_data, 
            active_learning_loop
        )
        
        # 准备主动学习数据
        print("\n1. 准备主动学习数据...")
        prepare_active_learning_data(
            data_dir=data_dir,
            output_dir=output_dir,
            init_labeled_num=5,    # 初始标注样本数
            test_ratio=0.1,        # 测试集比例
            random_seed=42,
            overwrite=True         # 重新生成数据
        )
        print("✓ 数据准备完成")
        
        # 运行BZ策略测试
        print("\n2. 测试BZ策略...")
        active_learning_loop(
            dataset_name='al_bz_test',
            output_dir=output_dir,
            strategy='bz',           # 使用BZ策略
            rounds=2,               # 只运行2轮进行测试
            select_num=2,           # 每轮选择2个样本
            checkpoint_freq=1       # 每轮保存检查点
        )
        print("✓ BZ策略测试完成")
        
    except Exception as e:
        print(f"✗ BZ策略测试失败: {e}")
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


def show_bz_strategy_info():
    """显示BZ策略信息"""
    
    print("\n" + "="*60)
    print("BZ策略说明")
    print("="*60)
    
    print("""
BZ策略特点:

1. 直接使用GNOT框架:
   - 使用train.py中的validate_epoch函数
   - 完全兼容现有的GNOT模型和数据处理流程
   - 无需额外的误差计算或物理方程

2. 选择策略:
   - 为每个未标注样本计算metric值
   - 选择metric最大的样本（模型表现最差的样本）
   - 这些样本是模型最需要学习的

3. 优势:
   - 简洁高效，直接复用现有代码
   - 基于模型的实际表现进行选择
   - 自动适应不同的loss函数和metric

4. 使用方法:
   
   from alpa import active_learning_loop
   
   # 使用BZ策略
   active_learning_loop(
       dataset_name='your_dataset',
       output_dir='./output',
       strategy='bz',        # 使用BZ策略
       rounds=10,
       select_num=5
   )

5. 工作原理:
   - 对每个未标注样本单独创建数据加载器
   - 使用训练好的模型进行评估
   - 通过validate_epoch函数计算metric
   - 选择metric值最大的样本加入训练集

6. 与其他策略对比:
   - PA策略: 基于预测误差（需要真实标签）
   - QBC策略: 基于预测不确定性（需要多次预测）
   - BZ策略: 基于模型评估metric（直接使用GNOT框架）
   - 几何策略: 基于数据几何特征（不需要模型）

BZ策略是对GNOT框架最自然、最直接的主动学习扩展。
""")


if __name__ == "__main__":
    print("开始测试BZ策略...")
    
    # 显示策略信息
    show_bz_strategy_info()
    
    # 运行测试
    test_bz_strategy()
    
    print(f"\n测试完成!")
    print(f"如果测试成功，你现在可以使用BZ策略:")
    print(f"  strategy='bz' : 基于GNOT模型测试框架的策略")
    print(f"  这是最适合GNOT模型的主动学习策略")
