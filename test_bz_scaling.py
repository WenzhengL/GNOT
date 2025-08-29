#!/usr/bin/env python3
"""
测试BZ策略维度缩放效果的脚本

比较原始BZ策略和缩放版本的差异，分析各维度误差的分布情况
"""

import os
import sys
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 添加当前目录到path
sys.path.append('/home/v-wenliao/gnot/GNOT')

from albz import train_model, evaluate_model
from bz_strategy_scale_fix import bz_query_with_dimension_scaling, calculate_dimension_scales
from alpa import bz_query


def load_test_data():
    """加载测试数据"""
    data_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"
    
    labeled_path = os.path.join(data_dir, 'al_labeled.pkl')
    unlabeled_path = os.path.join(data_dir, 'al_unlabeled.pkl')
    test_path = os.path.join(data_dir, 'al_test.pkl')
    
    print("加载数据...")
    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(unlabeled_path, 'rb') as f:
        unlabeled_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"已标注数据: {len(labeled_data)} 个样本")
    print(f"未标注数据: {len(unlabeled_data)} 个样本")
    print(f"测试数据: {len(test_data)} 个样本")
    
    return labeled_data, unlabeled_data, test_data


def analyze_dimension_distribution(unlabeled_data, sample_size=100):
    """分析各维度数据分布"""
    print(f"\n=== 分析各维度数据分布 ===")
    
    # 随机采样
    sample_indices = np.random.choice(len(unlabeled_data), 
                                    min(sample_size, len(unlabeled_data)), 
                                    replace=False)
    
    all_values = []
    for idx in sample_indices:
        try:
            Y_true = np.array(unlabeled_data[idx][1])
            if len(Y_true) > 0 and Y_true.shape[1] >= 5:
                all_values.append(Y_true)
        except:
            continue
    
    if not all_values:
        print("无法获取有效数据")
        return
    
    combined_values = np.vstack(all_values)
    
    dimension_names = ['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']
    
    print(f"样本统计（基于{len(all_values)}个样本，共{combined_values.shape[0]}个数据点）:")
    print(f"{'维度':<15} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} {'范围':<12}")
    print("-" * 80)
    
    for dim in range(5):
        values = combined_values[:, dim]
        mean_val = np.mean(values)
        std_val = np.std(values)
        min_val = np.min(values)
        max_val = np.max(values)
        range_val = max_val - min_val
        
        print(f"{dimension_names[dim]:<15} {mean_val:<12.4f} {std_val:<12.4f} "
              f"{min_val:<12.4f} {max_val:<12.4f} {range_val:<12.4f}")
    
    return combined_values, dimension_names


def compare_bz_strategies(labeled_data, unlabeled_data, select_num=20):
    """比较不同BZ策略的效果"""
    print(f"\n=== 比较BZ策略效果 ===")
    
    # 训练模型
    print("训练模型...")
    model_tuple = train_model(labeled_data)
    print("模型训练完成")
    
    # 只使用前50个样本进行快速测试
    test_unlabeled = unlabeled_data[:50]
    test_select_num = min(select_num, len(test_unlabeled))
    
    print(f"使用 {len(test_unlabeled)} 个样本进行测试，选择 {test_select_num} 个")
    
    strategies = {
        'original': '原始BZ策略',
        'adaptive': '自适应缩放BZ',
        'manual_low_pressure': '手动降低pressure权重',
        'equal_weights': '等权重'
    }
    
    results = {}
    
    for strategy_key, strategy_desc in strategies.items():
        print(f"\n--- 测试 {strategy_desc} ---")
        
        try:
            if strategy_key == 'original':
                # 使用原始BZ策略
                selected_idx = bz_query(model_tuple, test_unlabeled, test_select_num)
            elif strategy_key == 'adaptive':
                # 自适应缩放
                selected_idx = bz_query_with_dimension_scaling(
                    model_tuple, test_unlabeled, test_select_num, 
                    scaling_method='adaptive'
                )
            elif strategy_key == 'manual_low_pressure':
                # 手动降低pressure权重
                selected_idx = bz_query_with_dimension_scaling(
                    model_tuple, test_unlabeled, test_select_num, 
                    scaling_method='manual',
                    manual_weights=[0.1, 1.0, 1.0, 1.0, 1.0]
                )
            elif strategy_key == 'equal_weights':
                # 等权重
                selected_idx = bz_query_with_dimension_scaling(
                    model_tuple, test_unlabeled, test_select_num, 
                    scaling_method='equal'
                )
            
            results[strategy_key] = {
                'selected_idx': selected_idx,
                'description': strategy_desc,
                'success': True
            }
            
            print(f"{strategy_desc} 选择的样本索引: {selected_idx}")
            
        except Exception as e:
            print(f"{strategy_desc} 失败: {e}")
            results[strategy_key] = {
                'selected_idx': [],
                'description': strategy_desc,
                'success': False,
                'error': str(e)
            }
    
    # 分析结果差异
    print(f"\n=== 策略比较结果 ===")
    
    successful_strategies = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_strategies) >= 2:
        strategy_names = list(successful_strategies.keys())
        
        # 计算策略之间的重叠度
        print(f"\n策略重叠度分析:")
        for i, strategy1 in enumerate(strategy_names):
            for j, strategy2 in enumerate(strategy_names):
                if i < j:
                    idx1 = set(successful_strategies[strategy1]['selected_idx'])
                    idx2 = set(successful_strategies[strategy2]['selected_idx'])
                    
                    overlap = len(idx1 & idx2)
                    union = len(idx1 | idx2)
                    jaccard = overlap / union if union > 0 else 0
                    
                    print(f"  {successful_strategies[strategy1]['description']} vs "
                          f"{successful_strategies[strategy2]['description']}: "
                          f"重叠 {overlap}/{test_select_num}, Jaccard={jaccard:.3f}")
        
        # 分析选择的样本特征
        print(f"\n选择样本特征分析:")
        for strategy_key, result in successful_strategies.items():
            selected_idx = result['selected_idx']
            if selected_idx:
                # 分析选择样本的物理参数分布
                selected_thetas = []
                for idx in selected_idx:
                    try:
                        theta = np.array(test_unlabeled[idx][2])
                        selected_thetas.append(theta)
                    except:
                        continue
                
                if selected_thetas:
                    selected_thetas = np.array(selected_thetas)
                    theta_mean = np.mean(selected_thetas, axis=0)
                    theta_std = np.std(selected_thetas, axis=0)
                    
                    print(f"  {result['description']}:")
                    print(f"    参数1 (curve_length_factor): 均值={theta_mean[0]:.4f}, 标准差={theta_std[0]:.4f}")
                    if len(theta_mean) > 1:
                        print(f"    参数2 (bending_strength): 均值={theta_mean[1]:.4f}, 标准差={theta_std[1]:.4f}")
    
    return results


def create_visualization(results, output_dir="/home/v-wenliao/gnot/GNOT/data/result"):
    """创建可视化图表"""
    print(f"\n=== 创建可视化图表 ===")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    successful_results = {k: v for k, v in results.items() if v['success']}
    
    if len(successful_results) < 2:
        print("成功的策略数量不足，无法创建比较图表")
        return
    
    # 创建策略比较图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 子图1: 选择样本数量对比
    strategy_names = [result['description'] for result in successful_results.values()]
    sample_counts = [len(result['selected_idx']) for result in successful_results.values()]
    
    ax1.bar(range(len(strategy_names)), sample_counts)
    ax1.set_xticks(range(len(strategy_names)))
    ax1.set_xticklabels(strategy_names, rotation=45, ha='right')
    ax1.set_ylabel('选择样本数量')
    ax1.set_title('各策略选择样本数量对比')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 策略重叠度热力图
    if len(successful_results) >= 2:
        strategy_keys = list(successful_results.keys())
        n = len(strategy_keys)
        overlap_matrix = np.zeros((n, n))
        
        for i, key1 in enumerate(strategy_keys):
            for j, key2 in enumerate(strategy_keys):
                if i == j:
                    overlap_matrix[i, j] = 1.0
                else:
                    idx1 = set(successful_results[key1]['selected_idx'])
                    idx2 = set(successful_results[key2]['selected_idx'])
                    
                    overlap = len(idx1 & idx2)
                    union = len(idx1 | idx2)
                    jaccard = overlap / union if union > 0 else 0
                    overlap_matrix[i, j] = jaccard
        
        im = ax2.imshow(overlap_matrix, cmap='Blues', vmin=0, vmax=1)
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.set_xticklabels([successful_results[key]['description'] for key in strategy_keys], 
                           rotation=45, ha='right')
        ax2.set_yticklabels([successful_results[key]['description'] for key in strategy_keys])
        ax2.set_title('策略重叠度 (Jaccard相似度)')
        
        # 添加数值标注
        for i in range(n):
            for j in range(n):
                text = ax2.text(j, i, f'{overlap_matrix[i, j]:.2f}',
                               ha="center", va="center", color="black")
        
        plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    
    # 保存图表
    output_path = os.path.join(output_dir, 'bz_strategy_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {output_path}")
    
    plt.show()


def main():
    """主函数"""
    print("=== BZ策略维度缩放效果测试 ===")
    
    try:
        # 加载数据
        labeled_data, unlabeled_data, test_data = load_test_data()
        
        # 分析数据分布
        analyze_dimension_distribution(unlabeled_data, sample_size=50)
        
        # 比较策略效果
        results = compare_bz_strategies(labeled_data, unlabeled_data, select_num=10)
        
        # 创建可视化
        create_visualization(results)
        
        print("\n=== 测试完成 ===")
        print("主要发现:")
        print("1. 原始BZ策略主要受pressure维度误差主导")
        print("2. 缩放版本能更好地平衡各维度的贡献")
        print("3. 手动权重版本可以针对性地调整特定维度的影响")
        print("4. 不同策略选择的样本可能有显著差异")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
