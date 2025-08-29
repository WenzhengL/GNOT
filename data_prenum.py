# -*- coding: utf-8 -*-
"""
主动学习数据预处理程序
用于将原始样本数据处理成主动学习所需的格式

作者: Assistant
日期: 2025-01-XX
"""

import os
import pickle
import random
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
from pathlib import Path


def set_random_seed(seed=42):
    """设置随机种子确保结果可重现"""
    random.seed(seed)
    np.random.seed(seed)
    print(f"已设置随机种子: {seed}")


def process_sample(folder):
    """
    处理单个样本文件夹
    
    参数:
        folder: 样本文件夹路径
        
    返回:
        样本数据 [X, Y, Theta, (empty_branch,)] 或 None（如果处理失败）
    """
    csv_path = os.path.join(folder, 'test2.csv')
    param_path = os.path.join(folder, 'parameter_vel.json')
    
    # 检查必要文件是否存在
    if not (os.path.exists(csv_path) and os.path.exists(param_path)):
        print(f"警告: {folder} 缺少必要文件 (test2.csv 或 parameter_vel.json)")
        return None
    
    try:
        # 读取CSV数据
        df = pd.read_csv(csv_path, skipinitialspace=True)
        
        # 提取空间坐标 (X)
        required_coord_cols = ['x-coordinate', 'y-coordinate', 'z-coordinate']
        missing_coord_cols = [col for col in required_coord_cols if col not in df.columns]
        if missing_coord_cols:
            print(f"错误: {folder} 缺少坐标列: {missing_coord_cols}")
            return None
        X = df[required_coord_cols].values
        
        # 提取物理量 (Y)
        required_physics_cols = ['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']
        missing_physics_cols = [col for col in required_physics_cols if col not in df.columns]
        if missing_physics_cols:
            print(f"错误: {folder} 缺少物理量列: {missing_physics_cols}")
            return None
        Y = df[required_physics_cols].values
        
        # 读取参数文件
        with open(param_path, 'r') as f:
            params_data = json.load(f)
        
        # 处理参数数据（可能是列表格式）
        if isinstance(params_data, list) and len(params_data) > 0:
            params = params_data[0]
        elif isinstance(params_data, dict):
            params = params_data
        else:
            print(f"错误: {folder} 参数文件格式不正确")
            return None
        
        # 扩展的关键参数 (Theta) - 包含所有重要的物理和几何参数
        theta_keys = [
            # 原有几何参数
            'curve_length_factor',      # 曲线长度因子
            'bending_strength',         # 弯曲强度
            
            # 新增物理参数
            'velocity',                 # 流体速度
            'diameter',                 # 特征直径
            'Renold',                   # 雷诺数
            'mu',                       # 动力粘度
            'rho',                      # 密度
            'scale_factor',             # 尺度因子
            'boulayer'                  # 边界层厚度
        ]
        
        Theta = []
        missing_params = []
        
        for k in theta_keys:
            if k in params and isinstance(params[k], (float, int)):
                Theta.append(float(params[k]))
            else:
                # 记录缺失的参数
                missing_params.append(k)
                
                # 根据参数类型设置合理的默认值
                if k == 'curve_length_factor':
                    Theta.append(1.0)  # 默认无弯曲
                elif k == 'bending_strength':
                    Theta.append(0.0)  # 默认无弯曲强度
                elif k == 'velocity':
                    Theta.append(0.1)  # 默认低速流动
                elif k == 'diameter':
                    Theta.append(0.01)  # 默认1cm特征尺寸
                elif k == 'Renold':
                    Theta.append(100.0)  # 默认层流雷诺数
                elif k == 'mu':
                    Theta.append(0.001)  # 默认水的粘度（约1cP）
                elif k == 'rho':
                    Theta.append(1000.0)  # 默认水的密度
                elif k == 'scale_factor':
                    Theta.append(1.0)  # 默认无缩放
                elif k == 'boulayer':
                    Theta.append(0.01)  # 默认边界层厚度
                else:
                    Theta.append(0.0)  # 其他未知参数默认为0
        
        # 如果有缺失参数，给出警告
        if missing_params and len(missing_params) < len(theta_keys):  # 只有部分缺失才警告
            print(f"警告: {folder} 缺少参数 {missing_params[:3]}{'...' if len(missing_params) > 3 else ''}，已使用默认值")
        
        Theta = np.array(Theta)
        
        # 创建空分支（根据GNOT要求）
        empty_branch = np.zeros((X.shape[0], 1))
        
        # 验证数据维度
        if X.shape[0] != Y.shape[0]:
            print(f"错误: {folder} 坐标和物理量数据行数不匹配: {X.shape[0]} vs {Y.shape[0]}")
            return None
        
        # 简化输出信息（只对前几个样本显示详细信息）
        sample_name = os.path.basename(folder)
        print(f"成功处理 {sample_name}: {X.shape[0]} 个数据点, {len(Theta)} 个参数")
        
        # 详细显示参数信息（仅对前几个样本）
        if 'sample-000' in sample_name or sample_name in ['sample-0001', 'sample-0002']:
            print(f"  参数详情:")
            for i, (key, value) in enumerate(zip(theta_keys, Theta)):
                print(f"    {i:2d}. {key:20s}: {value:.6f}")
        
        return [X, Y, Theta, (empty_branch,)]
        
    except Exception as e:
        print(f"处理 {folder} 时出错: {e}")
        return None


def split_for_active_learning_fixed(all_samples, labeled_num=100, unlabeled_num=500, test_num=100, random_seed=42):
    """
    将样本分割为固定数量的标注集、未标注池和测试集
    
    参数:
        all_samples: 所有样本路径列表
        labeled_num: 标注样本数量
        unlabeled_num: 未标注样本数量
        test_num: 测试样本数量
        random_seed: 随机种子
        
    返回:
        (labeled_samples, unlabeled_samples, test_samples)
    """
    print(f"开始固定数量数据分割...")
    print(f"  - 总样本数: {len(all_samples)}")
    print(f"  - 需要标注数: {labeled_num}")
    print(f"  - 需要未标注数: {unlabeled_num}")
    print(f"  - 需要测试数: {test_num}")
    print(f"  - 总需求: {labeled_num + unlabeled_num + test_num}")
    print(f"  - 随机种子: {random_seed}")
    
    # 检查样本数量是否足够
    total_needed = labeled_num + unlabeled_num + test_num
    if len(all_samples) < total_needed:
        raise ValueError(f"样本数量不足！需要 {total_needed} 个样本，但只有 {len(all_samples)} 个可用样本")
    
    # 设置随机种子并打乱数据
    random.seed(random_seed)
    all_samples_copy = all_samples.copy()
    random.shuffle(all_samples_copy)
    
    # 按固定数量分割数据
    test_samples = all_samples_copy[:test_num]
    labeled_samples = all_samples_copy[test_num:test_num + labeled_num]
    unlabeled_samples = all_samples_copy[test_num + labeled_num:test_num + labeled_num + unlabeled_num]
    
    print(f"数据分割完成:")
    print(f"  - 测试集: {len(test_samples)} 个样本")
    print(f"  - 初始标注集: {len(labeled_samples)} 个样本")
    print(f"  - 未标注池: {len(unlabeled_samples)} 个样本")
    print(f"  - 剩余未使用: {len(all_samples) - total_needed} 个样本")
    
    return labeled_samples, unlabeled_samples, test_samples


def build_active_learning_dataset_fixed(data_dir, labeled_num=100, unlabeled_num=500, test_num=100, random_seed=42, sample_pattern='sample-*'):
    """
    构建固定数量的主动学习数据集
    
    参数:
        data_dir: 原始数据目录
        labeled_num: 标注样本数量
        unlabeled_num: 未标注样本数量
        test_num: 测试样本数量
        random_seed: 随机种子
        sample_pattern: 样本文件夹匹配模式
        
    返回:
        (labeled_data, unlabeled_data, test_data)
    """
    print(f"=== 构建固定数量主动学习数据集 ===")
    print(f"数据目录: {data_dir}")
    print(f"目标配置: labeled={labeled_num}, unlabeled={unlabeled_num}, test={test_num}")
    
    # 扫描样本文件夹
    if not os.path.exists(data_dir):
        raise ValueError(f"数据目录不存在: {data_dir}")
    
    # 查找所有符合模式的样本文件夹
    all_folders = [d for d in os.listdir(data_dir) if d.startswith('sample-')]
    all_samples = [os.path.join(data_dir, d) for d in all_folders
                  if os.path.isdir(os.path.join(data_dir, d))]
    
    print(f"发现 {len(all_samples)} 个样本文件夹")
    
    if len(all_samples) == 0:
        raise ValueError(f"在 {data_dir} 中未找到样本文件夹（应以 'sample-' 开头）")
    
    # 分割样本
    labeled_samples, unlabeled_samples, test_samples = split_for_active_learning_fixed(
        all_samples, labeled_num, unlabeled_num, test_num, random_seed
    )
    
    # 处理各部分数据
    print("\n=== 处理数据 ===")
    
    print("处理初始标注数据...")
    labeled_data = []
    failed_labeled = 0
    for sample in tqdm(labeled_samples, desc="初始标注"):
        processed = process_sample(sample)
        if processed is not None:
            labeled_data.append(processed)
        else:
            failed_labeled += 1
    
    print("处理未标注池数据...")
    unlabeled_data = []
    failed_unlabeled = 0
    for sample in tqdm(unlabeled_samples, desc="未标注池"):
        processed = process_sample(sample)
        if processed is not None:
            unlabeled_data.append(processed)
        else:
            failed_unlabeled += 1
    
    print("处理测试集数据...")
    test_data = []
    failed_test = 0
    for sample in tqdm(test_samples, desc="测试集"):
        processed = process_sample(sample)
        if processed is not None:
            test_data.append(processed)
        else:
            failed_test += 1
    
    # 检查是否需要补充数据
    print(f"\n=== 数据处理结果 ===")
    print(f"成功处理的样本:")
    print(f"  - 初始标注: {len(labeled_data)}/{len(labeled_samples)} (失败: {failed_labeled})")
    print(f"  - 未标注池: {len(unlabeled_data)}/{len(unlabeled_samples)} (失败: {failed_unlabeled})")
    print(f"  - 测试集: {len(test_data)}/{len(test_samples)} (失败: {failed_test})")
    
    # 如果处理失败导致数量不足，从剩余样本中补充
    total_needed = labeled_num + unlabeled_num + test_num
    used_samples = set(labeled_samples + unlabeled_samples + test_samples)
    remaining_samples = [s for s in all_samples if s not in used_samples]
    
    if len(labeled_data) < labeled_num and remaining_samples:
        shortage = labeled_num - len(labeled_data)
        print(f"\n补充标注数据，缺少 {shortage} 个...")
        for sample in tqdm(remaining_samples[:shortage * 2], desc="补充标注"):  # 多取一些防止失败
            if len(labeled_data) >= labeled_num:
                break
            processed = process_sample(sample)
            if processed is not None:
                labeled_data.append(processed)
                remaining_samples.remove(sample)
    
    if len(unlabeled_data) < unlabeled_num and remaining_samples:
        shortage = unlabeled_num - len(unlabeled_data)
        print(f"\n补充未标注数据，缺少 {shortage} 个...")
        for sample in tqdm(remaining_samples[:shortage * 2], desc="补充未标注"):
            if len(unlabeled_data) >= unlabeled_num:
                break
            processed = process_sample(sample)
            if processed is not None:
                unlabeled_data.append(processed)
                remaining_samples.remove(sample)
    
    if len(test_data) < test_num and remaining_samples:
        shortage = test_num - len(test_data)
        print(f"\n补充测试数据，缺少 {shortage} 个...")
        for sample in tqdm(remaining_samples[:shortage * 2], desc="补充测试"):
            if len(test_data) >= test_num:
                break
            processed = process_sample(sample)
            if processed is not None:
                test_data.append(processed)
                remaining_samples.remove(sample)
    
    # 最终统计
    print(f"\n=== 最终数据统计 ===")
    print(f"  - 标注数据: {len(labeled_data)} / {labeled_num} (目标)")
    print(f"  - 未标注数据: {len(unlabeled_data)} / {unlabeled_num} (目标)")
    print(f"  - 测试数据: {len(test_data)} / {test_num} (目标)")
    
    # 检查是否达到目标
    if len(labeled_data) < labeled_num:
        print(f"警告: 标注数据不足，目标{labeled_num}个，实际{len(labeled_data)}个")
    if len(unlabeled_data) < unlabeled_num:
        print(f"警告: 未标注数据不足，目标{unlabeled_num}个，实际{len(unlabeled_data)}个")
    if len(test_data) < test_num:
        print(f"警告: 测试数据不足，目标{test_num}个，实际{len(test_data)}个")
    
    return labeled_data, unlabeled_data, test_data


def save_active_learning_data(labeled_data, unlabeled_data, test_data, output_dir):
    """
    保存主动学习数据集
    
    参数:
        labeled_data: 标注数据
        unlabeled_data: 未标注数据
        test_data: 测试数据
        output_dir: 输出目录
    """
    print(f"\n=== 保存数据集 ===")
    os.makedirs(output_dir, exist_ok=True)
    
    # 定义文件路径
    labeled_path = os.path.join(output_dir, 'al_labeled.pkl')
    unlabeled_path = os.path.join(output_dir, 'al_unlabeled.pkl')
    test_path = os.path.join(output_dir, 'al_test.pkl')
    
    # 保存数据
    with open(labeled_path, 'wb') as f:
        pickle.dump(labeled_data, f)
    print(f"已保存初始标注数据: {labeled_path} ({len(labeled_data)} 个样本)")
    
    with open(unlabeled_path, 'wb') as f:
        pickle.dump(unlabeled_data, f)
    print(f"已保存未标注池数据: {unlabeled_path} ({len(unlabeled_data)} 个样本)")
    
    with open(test_path, 'wb') as f:
        pickle.dump(test_data, f)
    print(f"已保存测试集数据: {test_path} ({len(test_data)} 个样本)")
    
    # 更新数据集信息，包含扩展的参数说明
    info = {
        'total_samples': len(labeled_data) + len(unlabeled_data) + len(test_data),
        'labeled_samples': len(labeled_data),
        'unlabeled_samples': len(unlabeled_data),
        'test_samples': len(test_data),
        'target_config': {
            'labeled_target': 100,
            'unlabeled_target': 500,
            'test_target': 100
        },
        'created_time': pd.Timestamp.now().isoformat(),
        'data_structure': {
            'sample_format': '[X, Y, Theta, (empty_branch,)]',
            'X': 'spatial coordinates (N, 3) [x-coordinate, y-coordinate, z-coordinate]',
            'Y': 'physics quantities (N, 5) [pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear]',
            'Theta': 'parameters (9,) [curve_length_factor, bending_strength, velocity, diameter, Renold, mu, rho, scale_factor, boulayer]',
            'empty_branch': 'zeros (N, 1)',
            'parameter_details': {
                'curve_length_factor': 'geometric parameter: curve length scaling factor',
                'bending_strength': 'geometric parameter: bending intensity',
                'velocity': 'flow parameter: fluid velocity (m/s)',
                'diameter': 'geometric parameter: characteristic diameter (m)',
                'Renold': 'flow parameter: Reynolds number (dimensionless)',
                'mu': 'fluid parameter: dynamic viscosity (Pa·s)',
                'rho': 'fluid parameter: density (kg/m³)',
                'scale_factor': 'simulation parameter: geometric scaling factor',
                'boulayer': 'flow parameter: boundary layer thickness (m)'
            },
            'default_values': {
                'curve_length_factor': 1.0,
                'bending_strength': 0.0,
                'velocity': 0.1,
                'diameter': 0.01,
                'Renold': 100.0,
                'mu': 0.001,
                'rho': 1000.0,
                'scale_factor': 1.0,
                'boulayer': 0.01
            }
        }
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"已保存数据集信息: {info_path}")

# ...existing code...


def prepare_active_learning_data_fixed(data_dir, output_dir, labeled_num=100, unlabeled_num=500, test_num=100, random_seed=42, overwrite=False):
    """
    完整的固定数量主动学习数据准备流程
    
    参数:
        data_dir: 原始数据目录
        output_dir: 输出目录
        labeled_num: 标注样本数量
        unlabeled_num: 未标注样本数量
        test_num: 测试样本数量
        random_seed: 随机种子
        overwrite: 是否覆盖已存在的文件
    """
    print("=" * 60)
    print("主动学习数据预处理程序 - 固定数量模式")
    print("=" * 60)
    print(f"目标配置: labeled={labeled_num}, unlabeled={unlabeled_num}, test={test_num}")
    
    # 检查输出文件是否已存在
    labeled_path = os.path.join(output_dir, 'al_labeled.pkl')
    unlabeled_path = os.path.join(output_dir, 'al_unlabeled.pkl')
    test_path = os.path.join(output_dir, 'al_test.pkl')
    
    if not overwrite and all(os.path.exists(p) for p in [labeled_path, unlabeled_path, test_path]):
        print(f"数据集已存在于 {output_dir}")
        print("如需重新生成，请使用 --overwrite 参数")
        return
    
    # 设置随机种子
    set_random_seed(random_seed)
    
    # 构建数据集
    labeled_data, unlabeled_data, test_data = build_active_learning_dataset_fixed(
        data_dir, labeled_num, unlabeled_num, test_num, random_seed
    )
    
    # 验证数据集质量
    if len(labeled_data) < labeled_num * 0.9:  # 允许10%的容差
        print(f"警告: 标注数据严重不足 ({len(labeled_data)}/{labeled_num})")
    if len(unlabeled_data) < unlabeled_num * 0.9:
        print(f"警告: 未标注数据严重不足 ({len(unlabeled_data)}/{unlabeled_num})")
    if len(test_data) < test_num * 0.9:
        print(f"警告: 测试数据严重不足 ({len(test_data)}/{test_num})")
    
    # 保存数据集
    save_active_learning_data(labeled_data, unlabeled_data, test_data, output_dir)
    
    print(f"\n=== 数据预处理完成 ===")
    print(f"输出目录: {output_dir}")
    print(f"可用于主动学习的文件:")
    print(f"  - al_labeled.pkl: 初始训练数据 ({len(labeled_data)} 个样本)")
    print(f"  - al_unlabeled.pkl: 未标注数据池 ({len(unlabeled_data)} 个样本)")
    print(f"  - al_test.pkl: 测试数据 ({len(test_data)} 个样本)")


def validate_dataset(dataset_dir):
    """
    验证数据集完整性和格式
    
    参数:
        dataset_dir: 数据集目录
    """
    print("=== 验证数据集 ===")
    
    files_to_check = ['al_labeled.pkl', 'al_unlabeled.pkl', 'al_test.pkl']
    
    for filename in files_to_check:
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.exists(filepath):
            print(f"❌ 缺少文件: {filename}")
            continue
        
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ {filename}: {len(data)} 个样本")
            
            if len(data) > 0:
                sample = data[0]
                if len(sample) >= 4:
                    X, Y, Theta, branch = sample[:4]
                    print(f"   - 坐标维度: {np.array(X).shape}")
                    print(f"   - 物理量维度: {np.array(Y).shape}")
                    print(f"   - 参数维度: {np.array(Theta).shape}")
                    
                    # 显示参数详情（扩展版）
                    theta_keys = [
                        'curve_length_factor', 'bending_strength', 'velocity', 'diameter', 
                        'Renold', 'mu', 'rho', 'scale_factor', 'boulayer'
                    ]
                    
                    if len(Theta) == len(theta_keys):
                        print(f"   - 参数详情 (扩展版 9 参数):")
                        for i, (key, value) in enumerate(zip(theta_keys, Theta)):
                            print(f"     {i:2d}. {key:20s}: {value:.6f}")
                    elif len(Theta) == 2:
                        print(f"   - 参数详情 (原版 2 参数):")
                        original_keys = ['curve_length_factor', 'bending_strength']
                        for i, (key, value) in enumerate(zip(original_keys, Theta)):
                            print(f"     {i:2d}. {key:20s}: {value:.6f}")
                    else:
                        print(f"   - 警告: 参数数量异常，期望2或9个，实际{len(Theta)}个")
                        print(f"   - 参数值: {Theta}")
                else:
                    print(f"   - 警告: 样本格式不正确")
                    
        except Exception as e:
            print(f"❌ {filename}: 读取失败 - {e}")

# ...existing code...


def main():
    """主函数，处理命令行参数"""
    parser = argparse.ArgumentParser(description='主动学习数据预处理程序 - 扩展参数版本')
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='原始数据目录路径')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='输出目录路径')
    parser.add_argument('--labeled_num', type=int, default=100,
                       help='标注样本数量 (默认: 100)')
    parser.add_argument('--unlabeled_num', type=int, default=500,
                       help='未标注样本数量 (默认: 500)')
    parser.add_argument('--test_num', type=int, default=100,
                       help='测试样本数量 (默认: 100)')
    parser.add_argument('--random_seed', type=int, default=42,
                       help='随机种子 (默认: 42)')
    parser.add_argument('--overwrite', action='store_true',
                       help='覆盖已存在的文件')
    parser.add_argument('--validate', action='store_true',
                       help='仅验证现有数据集')
    parser.add_argument('--show_params', action='store_true',
                       help='显示详细的参数信息')
    
    args = parser.parse_args()
    
    if args.validate:
        validate_dataset(args.output_dir)
    else:
        print("=" * 60)
        print("主动学习数据预处理程序 - 扩展参数版本 (9个参数)")
        print("=" * 60)
        print("扩展参数列表:")
        print("  0. curve_length_factor  - 曲线长度因子")
        print("  1. bending_strength     - 弯曲强度") 
        print("  2. velocity             - 流体速度 (m/s)")
        print("  3. diameter             - 特征直径 (m)")
        print("  4. Renold               - 雷诺数")
        print("  5. mu                   - 动力粘度 (Pa·s)")
        print("  6. rho                  - 密度 (kg/m³)")
        print("  7. scale_factor         - 尺度因子")
        print("  8. boulayer             - 边界层厚度 (m)")
        print("=" * 60)
        
        prepare_active_learning_data_fixed(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            labeled_num=args.labeled_num,
            unlabeled_num=args.unlabeled_num,
            test_num=args.test_num,
            random_seed=args.random_seed,
            overwrite=args.overwrite
        )

if __name__ == "__main__":
    main()


'''
python data_prenum.py \
    --data_dir /home/v-wenliao/gnot/GNOT/data/result \
    --output_dir /home/v-wenliao/gnot/GNOT/data/new/200-2500 \
    --labeled_num 200 \
    --unlabeled_num 3398 \
    --test_num 100 \
    --overwrite

python data_prenum.py \
    --output_dir /home/v-wenliao/gnot/GNOT/data/al_bz \
    --validate
'''