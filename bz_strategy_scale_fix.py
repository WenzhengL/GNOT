#!/usr/bin/env python3
"""
BZ策略误差缩放修复版本

解决维度0（pressure）误差过大导致策略偏向的问题，通过对各维度误差进行适当的缩放来平衡各维度的贡献。
"""

import os
import pickle
import numpy as np
import torch
from tqdm import tqdm
import json
import time
import shutil
import dgl
from torch.nn.utils.rnn import pad_sequence

# 从原始模块导入必要的函数和类
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader, MultipleTensors
from args import get_args as get_original_args
from train import train, validate_epoch
from utils import get_seed


def get_al_args():
    """获取主动学习的参数配置"""
    args = get_original_args()
    args.dataset = 'al_bz'
    args.data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz'
    args.batch_size = 1
    args.train_num = None
    args.train_data = 'train_data.pkl'
    args.val_data = 'val_data.pkl'
    args.test_data = 'test_data.pkl'
    return args


def calculate_dimension_scales(unlabeled_data, sample_size=50, scaling_strategy='balanced'):
    """
    计算各维度误差的缩放因子
    
    Args:
        unlabeled_data: 未标注数据列表
        sample_size: 采样样本数量
        scaling_strategy: 缩放策略
            - 'balanced': 真正平衡各维度贡献
            - 'sqrt': 平方根缩放
            - 'log': 对数缩放
        
    Returns:
        scales: 各维度的缩放因子 [dim0_scale, dim1_scale, ...]
    """
    print(f"计算维度缩放因子（策略: {scaling_strategy}）...")
    
    # 随机采样一些数据来估算各维度的范围
    sample_indices = np.random.choice(len(unlabeled_data), 
                                    min(sample_size, len(unlabeled_data)), 
                                    replace=False)
    
    all_values = []
    for idx in sample_indices:
        try:
            Y_true = np.array(unlabeled_data[idx][1])  # 真实值
            if len(Y_true) > 0 and Y_true.shape[1] >= 5:
                all_values.append(Y_true)
        except:
            continue
    
    if not all_values:
        print("警告: 无法采样数据，使用默认缩放因子")
        return np.array([0.1, 1.0, 1.0, 1.0, 1.0])  # 默认降低pressure权重
    
    # 合并所有采样数据
    combined_values = np.vstack(all_values)
    
    # 计算各维度的统计信息
    dimension_stats = []
    for dim in range(5):
        values = combined_values[:, dim]
        dim_mean = np.mean(np.abs(values))
        dim_std = np.std(values)
        dim_range = np.max(values) - np.min(values)
        dim_var = np.var(values)
        
        # 使用方差作为主要的缩放基准
        scale_factor = max(dim_std, dim_range * 0.1, 1e-6)
        
        dimension_stats.append({
            'dim': dim,
            'mean_abs': dim_mean,
            'std': dim_std,
            'range': dim_range,
            'var': dim_var,
            'scale_factor': scale_factor
        })
        
        print(f"  维度 {dim}: 均值绝对值={dim_mean:.6f}, 标准差={dim_std:.6f}, "
              f"范围={dim_range:.6f}, 方差={dim_var:.6f}, 缩放因子={scale_factor:.6f}")
    
    scales = np.array([stat['scale_factor'] for stat in dimension_stats])
    
    # 应用不同的缩放策略
    if scaling_strategy == 'balanced':
        # 真正平衡策略：倒数权重法（已通过测试验证）
        # 计算各维度的平均误差量级
        dim_means = np.array([stat['mean_abs'] for stat in dimension_stats])
        print(f"各维度平均误差: {dim_means}")
        
        # 使用倒数权重：误差大的维度权重小，误差小的维度权重大
        inverse_weights = 1.0 / (dim_means + 1e-6)
        
        # 归一化到总权重为5，这样平均每个维度权重为1
        scales = inverse_weights / inverse_weights.sum() * 5.0
        
        print(f"倒数权重（原始）: {inverse_weights}")
        print(f"平衡策略权重: {scales}")
        
    elif scaling_strategy == 'sqrt':
        # 平方根缩放
        scales = np.sqrt(scales)
        scales = scales / np.max(scales)
        print(f"平方根缩放权重: {scales}")
        
    elif scaling_strategy == 'log':
        # 对数缩放
        scales = np.log(scales + 1.0)
        scales = scales / np.max(scales)
        print(f"对数缩放权重: {scales}")
        
    else:
        # 标准标准化
        scales = scales / np.max(scales)
        print(f"标准化权重: {scales}")
    
    # 确保权重都是正数且在合理范围内
    scales = np.clip(scales, 0.01, 10.0)
    
    print(f"最终缩放因子: {scales}")
    
    return scales


def bz_query_with_dimension_scaling(model_tuple, unlabeled_data, select_num, 
                                  scaling_method='adaptive', 
                                  manual_weights=None):
    """
    带维度误差缩放的BZ策略
    
    Args:
        model_tuple: (model, metric_func, device)
        unlabeled_data: 未标注数据列表
        select_num: 需要选择的样本数量
        scaling_method: 缩放方法
            - 'adaptive': 自适应计算缩放因子
            - 'manual': 使用手动设置的权重
            - 'equal': 各维度等权重
        manual_weights: 手动权重 [w0, w1, w2, w3, w4]
    
    Returns:
        selected_idx: 选择的样本索引列表
    """
    model, metric_func, device = model_tuple
    print(f"=== BZ策略（带维度缩放）开始 ===")
    print(f"设备: {device}, 样本数: {len(unlabeled_data)}, 缩放方法: {scaling_method}")
    
    # 计算维度缩放因子
    if scaling_method == 'adaptive':
        dimension_scales = calculate_dimension_scales(unlabeled_data, scaling_strategy='balanced')
    elif scaling_method == 'manual' and manual_weights is not None:
        dimension_scales = np.array(manual_weights)
        print(f"使用手动权重: {dimension_scales}")
    elif scaling_method == 'equal':
        dimension_scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        print(f"使用等权重: {dimension_scales}")
    else:
        print(f"未知缩放方法 {scaling_method}，使用等权重")
        dimension_scales = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    model.eval()
    sample_errors = []
    failed_samples = 0
    dimension_errors_record = []  # 记录各维度误差用于分析
    
    print("开始计算各样本的预测误差...")
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="BZ误差计算")):
        sample_error = 0.0
        dim_errors = [0.0] * 5
        
        try:
            # === 数据预处理 ===
            coords = np.array(sample[0])  # 坐标 [N, 3]
            values = np.array(sample[1])  # 真实值 [N, 5]
            theta = np.array(sample[2])   # 物理参数 [2]
            branch_data = sample[3]       # 分支数据
            
            # 数据验证
            if len(coords) == 0 or len(values) == 0:
                sample_errors.append(999.0)  # 失败样本使用高误差
                dimension_errors_record.append(dim_errors)
                failed_samples += 1
                continue
            
            num_points = len(coords)
            max_points = 50000  # 限制点数避免内存问题
            
            if num_points > max_points:
                indices = np.random.choice(num_points, max_points, replace=False)
                coords_limited = coords[indices]
                values_limited = values[indices]
                num_points = max_points
            else:
                coords_limited = coords
                values_limited = values
            
            # 转换为张量
            coords_tensor = torch.tensor(coords_limited, dtype=torch.float32).to(device)
            values_tensor = torch.tensor(values_limited, dtype=torch.float32).to(device)
            theta_tensor = torch.tensor(theta, dtype=torch.float32).to(device)
            
            # === 创建图结构 ===
            # 简化的k-近邻图
            k = min(6, num_points - 1)
            edges_src = []
            edges_dst = []
            
            # 链式连接
            for j in range(num_points - 1):
                edges_src.append(j)
                edges_dst.append(j + 1)
            
            # 额外连接
            for j in range(0, num_points, 10):
                for k_offset in range(1, min(k, num_points - j)):
                    if j + k_offset < num_points:
                        edges_src.append(j)
                        edges_dst.append(j + k_offset)
            
            if len(edges_src) == 0:
                edges_src = [0]
                edges_dst = [0 if num_points == 1 else 1]
            
            g = dgl.graph((edges_src, edges_dst), num_nodes=num_points)
            g = g.to(device)
            g.ndata['x'] = coords_tensor
            g.ndata['y'] = values_tensor
            
            # === 准备模型输入 ===
            u_p = theta_tensor.unsqueeze(0)  # [1, 2]
            
            # 处理分支数据
            if isinstance(branch_data, tuple) and len(branch_data) > 0:
                branch_array = branch_data[0][:num_points]
                if isinstance(branch_array, np.ndarray):
                    branch_tensor = torch.tensor(branch_array, dtype=torch.float32).to(device)
                else:
                    branch_tensor = torch.tensor(branch_array, dtype=torch.float32).to(device)
                
                if len(branch_tensor.shape) == 1:
                    branch_tensor = branch_tensor.unsqueeze(-1)
                
                padded = pad_sequence([branch_tensor]).permute(1, 0, 2)
                g_u = MultipleTensors([padded])
            else:
                zero_tensor = torch.zeros((num_points, 1), dtype=torch.float32).to(device)
                padded = pad_sequence([zero_tensor]).permute(1, 0, 2)
                g_u = MultipleTensors([padded])
            
            # === 模型预测 ===
            pred = model(g, u_p, g_u)
            target = values_tensor
            
            # 确保形状匹配
            if pred.shape != target.shape:
                min_rows = min(pred.shape[0], target.shape[0])
                min_cols = min(pred.shape[1] if len(pred.shape) > 1 else 1, 
                              target.shape[1] if len(target.shape) > 1 else 1)
                pred = pred[:min_rows, :min_cols]
                target = target[:min_rows, :min_cols]
            
            # === 计算各维度误差并应用缩放 ===
            raw_errors = []
            
            # 先计算所有维度的原始误差
            for dim in range(min(5, target.shape[1])):
                dim_mse = torch.mean((pred[:, dim] - target[:, dim]) ** 2).item()
                raw_errors.append(dim_mse)
                dim_errors[dim] = dim_mse  # 记录原始误差
            
            # 应用权重进行平衡
            weighted_errors = []
            for dim in range(len(raw_errors)):
                # 核心修复：直接用权重相乘来平衡各维度误差
                # 权重小的维度（原误差大）影响降低，权重大的维度（原误差小）影响保持
                if dimension_scales[dim] > 0:
                    # 直接使用权重：小权重降低大误差的影响，大权重保持小误差的影响
                    weighted_error = raw_errors[dim] * dimension_scales[dim]
                else:
                    weighted_error = raw_errors[dim]
                
                weighted_errors.append(weighted_error)
                
                if i < 5:  # 前5个样本显示详细信息
                    print(f"样本 {i} 维度 {dim}: 原始误差={raw_errors[dim]:.6f}, "
                          f"权重={dimension_scales[dim]:.6f}, "
                          f"加权误差={weighted_error:.6f}")
            
            # 综合误差评分（使用加权后的误差）
            sample_error = np.sum(weighted_errors)
            
            # 记录信息
            sample_errors.append(sample_error)
            dimension_errors_record.append(dim_errors)
            
            if i < 5:
                print(f"样本 {i}: 综合缩放误差 = {sample_error:.6f}")
                print(f"  数据点数: {num_points}")
                print(f"  预测形状: {pred.shape}, 真实值形状: {target.shape}")
                print(f"  坐标范围: [{coords_tensor.min().item():.4f}, {coords_tensor.max().item():.4f}]")
                print(f"  真实值范围: [{target.min().item():.4f}, {target.max().item():.4f}]")
                print(f"  预测值范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"  物理参数 theta: [{theta_tensor[0].item():.4f}, {theta_tensor[1].item():.4f}]")
                
        except Exception as e:
            if i < 3:
                print(f"样本 {i} 预测失败: {e}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
            
            sample_errors.append(999.0)  # 失败样本使用高误差
            dimension_errors_record.append(dim_errors)
            failed_samples += 1
    
    # === 结果分析 ===
    valid_errors = [e for e in sample_errors if e != 999.0]
    
    print(f"\n=== BZ策略（缩放版）统计 ===")
    print(f"成功预测样本数: {len(valid_errors)}")
    print(f"失败样本数: {failed_samples}")
    
    if len(valid_errors) > 0:
        print(f"有效误差最小值: {min(valid_errors):.6f}")
        print(f"有效误差最大值: {max(valid_errors):.6f}")
        print(f"有效误差平均值: {np.mean(valid_errors):.6f}")
        print(f"有效误差标准差: {np.std(valid_errors):.6f}")
        
        # 分析各维度误差分布
        if dimension_errors_record:
            dim_arrays = np.array(dimension_errors_record)
            print(f"\n各维度原始误差统计:")
            for dim in range(5):
                dim_values = dim_arrays[:, dim]
                valid_dim_values = dim_values[dim_values > 0]
                if len(valid_dim_values) > 0:
                    print(f"  维度 {dim}: 均值={np.mean(valid_dim_values):.6f}, "
                          f"标准差={np.std(valid_dim_values):.6f}, "
                          f"范围=[{np.min(valid_dim_values):.6f}, {np.max(valid_dim_values):.6f}]")
    else:
        print("所有样本预测都失败了")
    
    # === 选择样本 ===
    selected_idx = np.argsort(sample_errors)[-select_num:]
    selected_errors = [sample_errors[i] for i in selected_idx]
    
    print(f"\n选中样本的加权误差范围: {min(selected_errors):.6f} - {max(selected_errors):.6f}")
    
    # 显示前几个选中样本的详细信息
    print(f"\n=== 选中样本详细信息 ===")
    sorted_selection = sorted(zip(selected_idx, selected_errors), key=lambda x: x[1], reverse=True)
    for rank, (idx, error) in enumerate(sorted_selection[:10]):  # 显示前10个
        print(f"排名 {rank+1}: 样本索引 {idx}, 加权误差 {error:.6f}")
        if idx < len(dimension_errors_record):
            original_errors = dimension_errors_record[idx]
            print(f"    原始各维度误差: {[f'{e:.6f}' for e in original_errors]}")
    
    print(f"BZ策略（平衡版）完成，选择了 {len(selected_idx)} 个样本")
    
    return selected_idx.tolist()


def compare_scaling_methods(model_tuple, unlabeled_data, select_num=10):
    """
    比较不同缩放方法的效果
    """
    print("=== 比较不同缩放方法 ===")
    
    methods = [
        ('equal', '等权重', None),
        ('adaptive', '自适应缩放', None),
        ('manual_pressure_down', '手动降低pressure权重', [0.1, 1.0, 1.0, 1.0, 1.0]),
        ('manual_balanced', '手动平衡权重', [0.2, 1.0, 1.0, 1.0, 1.0])
    ]
    
    results = {}
    
    for method_name, method_desc, manual_weights in methods:
        print(f"\n--- 测试方法: {method_desc} ---")
        
        try:
            if method_name.startswith('manual'):
                selected_idx = bz_query_with_dimension_scaling(
                    model_tuple, unlabeled_data[:50], select_num,  # 只用50个样本测试
                    scaling_method='manual', 
                    manual_weights=manual_weights
                )
            else:
                selected_idx = bz_query_with_dimension_scaling(
                    model_tuple, unlabeled_data[:50], select_num,
                    scaling_method=method_name
                )
            
            results[method_name] = {
                'selected_idx': selected_idx,
                'description': method_desc
            }
            
        except Exception as e:
            print(f"方法 {method_desc} 失败: {e}")
            results[method_name] = {
                'selected_idx': [],
                'description': method_desc,
                'error': str(e)
            }
    
    # 分析结果差异
    print(f"\n=== 方法比较结果 ===")
    for method_name, result in results.items():
        if 'error' not in result:
            print(f"{result['description']}: 选择样本 {result['selected_idx']}")
        else:
            print(f"{result['description']}: 失败 - {result['error']}")
    
    return results


if __name__ == "__main__":
    print("BZ策略维度缩放修复测试")
    
    # 这里可以添加测试代码
    print("请在主动学习脚本中调用 bz_query_with_dimension_scaling 函数")
