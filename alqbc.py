import os
import pickle
import random
import csv
import json
import numpy as np
import pandas as pd
import torch
import time
import sys
import shutil
import subprocess
from tqdm import tqdm
from io import StringIO

# === 1. 导入你的GNOT训练和评估相关函数 ===
from train import train, validate_epoch
from args import get_args as get_original_args
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from utils import get_seed


# === 添加确定性训练设置函数 ===
def set_deterministic_training(seed=42):
    """设置确定性训练，确保结果可重现"""
    # Python随机种子
    random.seed(seed)
    
    # NumPy随机种子
    np.random.seed(seed)
    
    # PyTorch随机种子
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 确保CUDA操作的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量确保完全确定性
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    
    print(f"已设置确定性训练，种子: {seed}")


# === 0. 数据预处理函数 ===
def process_sample(folder):
    csv_path = os.path.join(folder, 'test2.csv')
    param_path = os.path.join(folder, 'parameter_vel.json')
    if not (os.path.exists(csv_path) and os.path.exists(param_path)):
        return None
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        X = df[['x-coordinate', 'y-coordinate', 'z-coordinate']].values
        Y = df[['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear']].values
        with open(param_path, 'r') as f:
            params = json.load(f)[0]
        theta_keys = ['curve_length_factor', 'bending_strength']
        Theta = []
        for k in theta_keys:
            if k in params and isinstance(params[k], (float, int)):
                Theta.append(params[k])
        Theta = np.array(Theta)
        empty_branch = np.zeros((X.shape[0], 1))
        return [X, Y, Theta, (empty_branch,)]
    except Exception as e:
        print(f"处理 {folder} 时出错: {e}")
        return None

def split_for_active_learning(all_samples, init_labeled_num=10, test_ratio=0.1, random_seed=42):
    random.seed(random_seed)
    random.shuffle(all_samples)
    test_num = int(len(all_samples) * test_ratio)
    test_samples = all_samples[:test_num]
    pool_samples = all_samples[test_num:]
    labeled_samples = pool_samples[:init_labeled_num]
    unlabeled_samples = pool_samples[init_labeled_num:]
    return labeled_samples, unlabeled_samples, test_samples

def build_active_learning_dataset(data_dir, init_labeled_num=10, test_ratio=0.1, random_seed=42):
    all_samples = [os.path.join(data_dir, d) for d in os.listdir(data_dir)
                  if d.startswith('sample-') and os.path.isdir(os.path.join(data_dir, d))]
    labeled, unlabeled, test = split_for_active_learning(
        all_samples, init_labeled_num, test_ratio, random_seed
    )
    print(f"初始已标注: {len(labeled)}，未标注池: {len(unlabeled)}，测试集: {len(test)}")
    labeled_data = [process_sample(f) for f in tqdm(labeled, desc="处理初始已标注")]
    unlabeled_data = [process_sample(f) for f in tqdm(unlabeled, desc="处理未标注池")]
    test_data = [process_sample(f) for f in tqdm(test, desc="处理测试集")]
    labeled_data = [x for x in labeled_data if x is not None]
    unlabeled_data = [x for x in unlabeled_data if x is not None]
    test_data = [x for x in test_data if x is not None]
    return labeled_data, unlabeled_data, test_data

def prepare_active_learning_data(data_dir, output_dir, init_labeled_num=10, test_ratio=0.1, random_seed=42, overwrite=False):
    os.makedirs(output_dir, exist_ok=True)
    labeled_path = os.path.join(output_dir, 'al_labeled.pkl')
    unlabeled_path = os.path.join(output_dir, 'al_unlabeled.pkl')
    test_path = os.path.join(output_dir, 'al_test.pkl')
    # 如果文件已存在则跳过，除非overwrite为True
    if overwrite or not (os.path.exists(labeled_path) and os.path.exists(unlabeled_path) and os.path.exists(test_path)):
        labeled_data, unlabeled_data, test_data = build_active_learning_dataset(
            data_dir, init_labeled_num, test_ratio, random_seed
        )
        with open(labeled_path, 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(unlabeled_path, 'wb') as f:
            pickle.dump(unlabeled_data, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        print(f"主动学习数据集已保存，初始已标注: {len(labeled_data)}，未标注池: {len(unlabeled_data)}，测试集: {len(test_data)}")
    else:
        print("主动学习数据集已存在，跳过数据划分。")


# === 1. 获取参数 ===
def get_al_args():
    class Args:
        # 数据与运行相关
        dataset = 'al_qbc'
        component = 'all'
        seed = 2023
        space_dim = 3
        gpu = 2
        use_tb = 0
        comment = ""
        train_num = 'all'
        test_num = 'all'
        sort_data = 0
        normalize_x = 'unit'
        use_normalizer = 'unit'
        # 训练相关 - 保守的内存设置
        epochs = 2
        optimizer = 'AdamW'
        lr = 1e-3
        weight_decay = 5e-6
        grad_clip = 1000.0
        batch_size = 4  # 更保守的batch size
        val_batch_size = 2
        no_cuda = False
        lr_method = 'cycle'
        lr_step_size = 50
        warmup_epochs = 50
        loss_name = 'rel2'
        # 模型相关
        model_name = 'GNOT'
        n_hidden = 64
        n_layers = 3
        act = 'gelu'
        n_head = 1
        ffn_dropout = 0.15
        attn_dropout = 0.15
        mlp_layers = 3
        attn_type = 'linear'
        hfourier_dim = 0
        n_experts = 1
        branch_sizes = [2]
        n_inner = 4
    return Args()

# === 2. 训练模型 (修改为使用独立的中间数据目录) ===
def train_model(labeled_data):
    # 在训练前重新设置随机种子确保一致性
    set_deterministic_training(42)
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 使用时间戳创建唯一的临时文件，避免冲突
    import time
    timestamp = int(time.time() * 1000000)  # 微秒级时间戳
    temp_train_file = os.path.join(temp_data_dir, f'temp_train_{timestamp}.pkl')
    
    # 保存到临时位置
    with open(temp_train_file, 'wb') as f:
        pickle.dump(labeled_data, f)
    
    # 定义标准训练文件路径（在独立目录中）
    standard_train_path = os.path.join(temp_data_dir, 'al_labeled.pkl')
    backup_train_path = os.path.join(temp_data_dir, 'al_labeled_backup.pkl')
    
    # 备份现有文件（如果存在）
    backup_made = False
    if os.path.exists(standard_train_path):
        import shutil
        shutil.copy2(standard_train_path, backup_train_path)
        backup_made = True
    
    # 复制临时文件到标准位置
    import shutil
    shutil.copy2(temp_train_file, standard_train_path)
    
    try:
        args = get_al_args()
        args.dataset = 'al_qbc'  # 确保使用正确的数据集名称
        args.no_cuda = False  # 使用GPU
        
        num_samples = len(labeled_data)
        print(f"训练样本数: {num_samples}")
        print(f"中间数据保存位置: {temp_data_dir}")
        
        # GPU设备选择
        if torch.cuda.is_available() and not args.no_cuda:
            device = torch.device('cuda:0')  # 使用第一个可见的GPU
            # 清理GPU缓存
            torch.cuda.empty_cache()
            print(f"使用设备: {device}")
            print(f"GPU内存状态: {torch.cuda.get_device_properties(0).total_memory // 1024**2}MB 总计")
            print(f"GPU可用内存: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) // 1024**2}MB")
        else:
            device = torch.device('cpu')
            print(f"使用设备: {device}")
        
        get_seed(args.seed)
        train_dataset, _ = get_dataset(args)
        train_loader = MIODataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
        model = get_model(args).to(device)
        
        # === 验证并强制设置dropout ===
        print("=== 验证模型dropout配置 ===")
        dropout_count = 0
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                print(f"发现dropout层: {name}, 当前p={module.p}")
                if module.p < 0.1:  # 如果dropout概率太小，强制设置
                    module.p = 0.15
                    print(f"  -> 已调整为: p={module.p}")
                dropout_count += 1
        print(f"总计处理 {dropout_count} 个dropout层")
        print("=" * 40)
        
        loss_func = get_loss_func(args.loss_name, args, regularizer=True, normalizer=train_dataset.y_normalizer)
        metric_func = get_loss_func('rel2', args, regularizer=False, normalizer=train_dataset.y_normalizer)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = None
        
        try:
            train(model, loss_func, metric_func, train_loader, train_loader, optimizer, scheduler, epochs=args.epochs, device=device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"GPU内存不足: {e}")
            print("尝试进一步减小batch size...")
            # 清理内存
            del model, optimizer, train_loader
            torch.cuda.empty_cache()
            
            # 减半batch size重试
            args.batch_size = max(1, args.batch_size // 2)
            args.val_batch_size = max(1, args.val_batch_size // 2)
            print(f"重新尝试，使用batch_size: {args.batch_size}")
            
            # 重新创建
            train_loader = MIODataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
            model = get_model(args).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            
            train(model, loss_func, metric_func, train_loader, train_loader, optimizer, scheduler, epochs=args.epochs, device=device)
        
        return model, metric_func, device
        
    finally:
        # 清理临时文件
        try:
            os.remove(temp_train_file)
            if backup_made:
                shutil.move(backup_train_path, standard_train_path)  # 恢复原文件
            elif os.path.exists(standard_train_path):
                os.remove(standard_train_path)  # 清理临时文件
        except Exception as cleanup_error:
            print(f"清理临时文件失败: {cleanup_error}")

# === 3. 评估模型 (修改为使用独立的中间数据目录) ===
def evaluate_model(model_tuple, test_data):
    model, metric_func, device = model_tuple
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 使用时间戳创建唯一的临时文件
    import time
    timestamp = int(time.time() * 1000000)
    temp_test_file = os.path.join(temp_data_dir, f'temp_test_{timestamp}.pkl')
    
    # 保存到临时位置
    with open(temp_test_file, 'wb') as f:
        pickle.dump(test_data, f)
    
    # 定义标准测试文件路径（在独立目录中）
    standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
    backup_test_path = os.path.join(temp_data_dir, 'al_test_backup.pkl')
    
    # 备份现有文件（如果存在）
    backup_made = False
    if os.path.exists(standard_test_path):
        import shutil
        shutil.copy2(standard_test_path, backup_test_path)
        backup_made = True
    
    # 复制临时文件到标准位置
    import shutil
    shutil.copy2(temp_test_file, standard_test_path)
    
    try:
        args = get_al_args()
        args.dataset = 'al_qbc'  # 确保使用正确的数据集名称
        _, test_dataset = get_dataset(args)
        test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        
        # === 关键修复：确保评估时使用eval模式 ===
        model.eval()  # 明确设置为评估模式，关闭dropout
        val_result = validate_epoch(model, metric_func, test_loader, device)
        metric = val_result["metric"]
        return metric
        
    finally:
        # 清理临时文件
        try:
            os.remove(temp_test_file)
            if backup_made:
                shutil.move(backup_test_path, standard_test_path)  # 恢复原文件
            elif os.path.exists(standard_test_path):
                os.remove(standard_test_path)  # 清理临时文件
        except Exception as cleanup_error:
            print(f"清理临时测试文件失败: {cleanup_error}")


def downsample_pointcloud(points, method='uniform', target_points=1000):
    """
    点云降采样函数
    
    参数:
    - points: 原始点云 (N, 3)
    - method: 降采样方法 ('uniform', 'random', 'farthest')
    - target_points: 目标点数
    
    返回:
    - 降采样后的点云
    """
    points = np.array(points)
    if len(points) <= target_points:
        return points
    
    if method == 'uniform':
        # 均匀采样：等间隔选择点
        indices = np.linspace(0, len(points) - 1, target_points, dtype=int)
        return points[indices]
    
    elif method == 'random':
        # 随机采样
        indices = np.random.choice(len(points), target_points, replace=False)
        return points[indices]
    
    elif method == 'farthest':
        # 最远点采样：保持几何分布特性
        sampled_indices = [0]  # 从第一个点开始
        remaining_indices = list(range(1, len(points)))
        
        for _ in range(target_points - 1):
            if not remaining_indices:
                break
            
            # 计算每个剩余点到已采样点的最小距离
            max_min_dist = -1
            farthest_idx = remaining_indices[0]
            
            for idx in remaining_indices:
                min_dist = min([np.linalg.norm(points[idx] - points[sampled_idx]) 
                               for sampled_idx in sampled_indices])
                if min_dist > max_min_dist:
                    max_min_dist = min_dist
                    farthest_idx = idx
            
            sampled_indices.append(farthest_idx)
            remaining_indices.remove(farthest_idx)
        
        return points[sampled_indices]
    
    else:
        raise ValueError(f"未知的降采样方法: {method}")

def chamfer_distance(P1, P2, downsample=True, target_points=1000, sample_method='uniform'):
    """
    优化的Chamfer距离计算，支持点云降采样
    
    参数:
    - P1, P2: 输入点云
    - downsample: 是否进行降采样
    - target_points: 降采样目标点数
    - sample_method: 降采样方法
    """
    P1, P2 = np.array(P1), np.array(P2)
    
    if downsample:
        if len(P1) > target_points:
            P1 = downsample_pointcloud(P1, method=sample_method, target_points=target_points)
        if len(P2) > target_points:
            P2 = downsample_pointcloud(P2, method=sample_method, target_points=target_points)
    
    dist1 = np.mean([np.min(np.linalg.norm(p1 - P2, axis=1)) for p1 in P1])
    dist2 = np.mean([np.min(np.linalg.norm(p2 - P1, axis=1)) for p2 in P2])
    return dist1 + dist2

def geometry_variance_query_optimized(labeled_data, unlabeled_data, select_num, 
                                    downsample=True, target_points=1000, sample_method='uniform'):
    """
    优化的几何差异性查询，支持点云降采样
    
    参数:
    - downsample: 是否启用点云降采样
    - target_points: 降采样目标点数（推荐500-2000）
    - sample_method: 降采样方法 ('uniform', 'random', 'farthest')
    """
    print(f"开始几何差异性查询 - 降采样: {downsample}, 目标点数: {target_points}, 方法: {sample_method}")
    
    all_data = labeled_data + unlabeled_data
    all_points = [sample[0] for sample in all_data]
    N = len(all_points)
    
    # 调试信息
    print(f"DEBUG: 已标注样本数: {len(labeled_data)}")
    print(f"DEBUG: 未标注样本数: {len(unlabeled_data)}")
    print(f"DEBUG: 总样本数N: {N}")
    print(f"DEBUG: 需要计算的距离对数: {N*(N-1)//2}")
    
    # 显示原始点云信息
    original_sizes = [len(points) for points in all_points]
    print(f"原始点云大小范围: {min(original_sizes)} - {max(original_sizes)}, 平均: {np.mean(original_sizes):.0f}")
    
    # 如果样本数量过大，强制使用更快的策略
    if N > 300:
        print(f"WARNING: 样本数量 {N} 过大，强制切换到快速几何特征方法")
        return geometry_variance_query_fast(labeled_data, unlabeled_data, select_num)
    
    # 构建距离矩阵
    M = np.zeros((N, N))
    
    # 添加进度条
    progress_bar = tqdm(total=N*(N-1)//2, desc="计算距离矩阵")
    
    for i in range(N):
        for j in range(i+1, N):
            d = chamfer_distance(all_points[i], all_points[j], 
                               downsample=downsample, 
                               target_points=target_points, 
                               sample_method=sample_method)
            M[i, j] = M[j, i] = d
            progress_bar.update(1)
    
    progress_bar.close()
    
    # 最远点采样选择
    labeled_idx = set(range(len(labeled_data)))
    unlabeled_idx = list(range(len(labeled_data), N))
    selected = list(labeled_idx)
    
    print(f"开始选择 {select_num} 个最具差异性的样本...")
    for _ in range(select_num):
        if not unlabeled_idx:
            break
        min_dists = [min(M[i, selected]) if selected else 0 for i in unlabeled_idx]
        next_idx = unlabeled_idx[np.argmax(min_dists)]
        selected.append(next_idx)
        unlabeled_idx.remove(next_idx)
    
    result_indices = [i - len(labeled_data) for i in selected[len(labeled_data):]]
    print(f"几何差异性查询完成，选择了 {len(result_indices)} 个样本")
    return result_indices

def geometry_variance_query(labeled_data, unlabeled_data, select_num):
    """
    兼容性包装函数，使用优化的降采样版本
    """
    # 自动选择降采样参数
    all_data = labeled_data + unlabeled_data
    all_points = [sample[0] for sample in all_data]
    max_points = max(len(points) for points in all_points)
    N = len(all_points)
    
    print(f"样本统计: 已标注={len(labeled_data)}, 未标注={len(unlabeled_data)}, 总计={N}")
    print(f"点云大小: 最大={max_points}, 平均={np.mean([len(p) for p in all_points]):.0f}")
    
    # 如果样本数过多，直接使用快速方法
    if N > 250:
        print(f"样本数量 {N} 过大，切换到超快速几何特征方法")
        return geometry_variance_query_fast(labeled_data, unlabeled_data, select_num)
    
    # 根据点云大小选择策略
    if max_points > 5000:
        # 大点云：激进降采样
        target_points = 500
        sample_method = 'farthest'  # 保持几何特征
        print(f"检测到大点云（最大{max_points}点），使用激进降采样策略: {target_points}点")
    elif max_points > 2000:
        # 中等点云：适度降采样
        target_points = 1000
        sample_method = 'uniform'
        print(f"检测到中等点云（最大{max_points}点），使用适度降采样策略: {target_points}点")
    else:
        # 小点云：轻度降采样或不降采样
        target_points = 1500
        sample_method = 'uniform'
        print(f"检测到小点云（最大{max_points}点），使用轻度降采样策略: {target_points}点")
    
    # 预估计算时间
    estimated_pairs = N * (N - 1) // 2
    estimated_time_seconds = estimated_pairs * 0.1  # 每对大约0.1秒
    estimated_minutes = estimated_time_seconds / 60
    
    print(f"预估计算: {estimated_pairs} 个距离对, 大约需要 {estimated_minutes:.1f} 分钟")
    
    if estimated_minutes > 30:  # 超过30分钟就用快速方法
        print("预估时间过长，强制使用快速几何特征方法")
        return geometry_variance_query_fast(labeled_data, unlabeled_data, select_num)
    
    return geometry_variance_query_optimized(
        labeled_data, unlabeled_data, select_num,
        downsample=True, target_points=target_points, sample_method=sample_method
    )

def geometry_features_distance(points1, points2):
    """
    基于几何特征的快速距离计算（作为Chamfer距离的快速替代）
    """
    def extract_features(points):
        points = np.array(points)
        # 重心
        centroid = np.mean(points, axis=0)
        # 包围盒
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        bbox_size = bbox_max - bbox_min
        # 分布特征
        variance = np.var(points, axis=0)
        std = np.std(points, axis=0)
        # 距离特征
        distances = np.linalg.norm(points - centroid, axis=1)
        mean_dist = np.mean(distances)
        max_dist = np.max(distances)
        
        return np.concatenate([centroid, bbox_size, variance, std, [mean_dist, max_dist]])
    
    feat1 = extract_features(points1)
    feat2 = extract_features(points2)
    return np.linalg.norm(feat1 - feat2)

def geometry_variance_query_fast(labeled_data, unlabeled_data, select_num):
    """
    基于几何特征的快速几何差异性查询（超快版本）
    """
    print(f"使用快速几何特征方法进行几何差异性查询...")
    
    all_data = labeled_data + unlabeled_data
    all_points = [sample[0] for sample in all_data]
    N = len(all_points)
    
    # 构建距离矩阵（使用几何特征）
    M = np.zeros((N, N))
    
    progress_bar = tqdm(total=N*(N-1)//2, desc="计算几何特征距离")
    for i in range(N):
        for j in range(i+1, N):
            d = geometry_features_distance(all_points[i], all_points[j])
            M[i, j] = M[j, i] = d
            progress_bar.update(1)
    progress_bar.close()
    
    # 最远点采样
    labeled_idx = set(range(len(labeled_data)))
    unlabeled_idx = list(range(len(labeled_data), N))
    selected = list(labeled_idx)
    
    for _ in range(select_num):
        if not unlabeled_idx:
            break
        min_dists = [min(M[i, selected]) if selected else 0 for i in unlabeled_idx]
        next_idx = unlabeled_idx[np.argmax(min_dists)]
        selected.append(next_idx)
        unlabeled_idx.remove(next_idx)
    
    return [i - len(labeled_data) for i in selected[len(labeled_data):]]


def force_enable_dropout(model, dropout_p=0.15):
    """
    强制启用模型中的所有dropout层，确保QBC策略有效
    """
    dropout_layers = []
    
    def apply_dropout(module):
        for name, child in module.named_children():
            if isinstance(child, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                if child.p < 0.01:  # 如果dropout概率太小
                    child.p = dropout_p
                    print(f"  强制启用 {name}: p={child.p}")
                dropout_layers.append((name, child))
            apply_dropout(child)
    
    apply_dropout(model)
    return dropout_layers

def verify_dropout_working(model, test_input, device, num_tests=5):
    """
    验证dropout是否真正起作用（产生不同的输出）
    """
    model.train()  # 确保在训练模式
    predictions = []
    
    with torch.no_grad():
        for i in range(num_tests):
            # 多次前向传播，应该得到不同结果
            if isinstance(test_input, tuple):
                g, u_p, g_u = test_input
                pred = model(g.to(device), u_p.to(device), g_u.to(device)).cpu().numpy()
            else:
                pred = model(test_input.to(device)).cpu().numpy()
            predictions.append(pred)
    
    # 计算预测间的差异
    predictions = np.array(predictions)
    variance = np.var(predictions, axis=0)
    mean_variance = np.mean(variance)
    
    print(f"Dropout验证结果:")
    print(f"  预测数量: {len(predictions)}")
    print(f"  平均方差: {mean_variance:.8f}")
    print(f"  最大方差: {np.max(variance):.8f}")
    print(f"  Dropout工作状态: {'正常' if mean_variance > 1e-6 else '异常'}")
    
    return mean_variance > 1e-6


def diagnose_dropout(model, test_input, device, num_runs=2):
    """深入诊断为什么MC Dropout方差为0。

    输出可能原因:
      1) 没有Dropout层 / p=0
      2) Dropout层处于eval模式
      3) Dropout输入本身全部为0或常数 -> 掉不出差异
      4) 只有1个样本且后续汇聚逻辑把随机性抹平
    """
    print("\n[Dropout 深度诊断] 开始")
    # 收集所有 dropout 层
    dropout_layers = []
    for name, m in model.named_modules():
        if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            dropout_layers.append((name, m))
    if not dropout_layers:
        print("  ✗ 未找到任何Dropout层 -> 需要在模型结构中显式加入")
        return
    print(f"  找到 {len(dropout_layers)} 个Dropout层: {[n for n,_ in dropout_layers]}")

    # 预/后 hook 捕获输入输出
    layer_stats = {}
    def make_pre(name):
        def _pre(module, inp):
            x = inp[0]
            layer_stats.setdefault(name, {})['input'] = x.detach().float().clone()
        return _pre
    def make_post(name):
        def _post(module, inp, out):
            layer_stats.setdefault(name, {})['output'] = out.detach().float().clone()
        return _post

    hooks = []
    for name, layer in dropout_layers:
        hooks.append(layer.register_forward_pre_hook(make_pre(name)))
        hooks.append(layer.register_forward_hook(make_post(name)))

    # 运行两次前向
    model.train()
    preds = []
    with torch.no_grad():
        for r in range(num_runs):
            # 清空上一轮捕获
            for k in layer_stats:
                layer_stats[k].pop('input', None)
                layer_stats[k].pop('output', None)
            if isinstance(test_input, tuple):
                g, u_p, g_u = test_input
                pred = model(g.to(device), u_p.to(device), g_u.to(device))
            else:
                pred = model(test_input.to(device))
            preds.append(pred.detach().cpu())
            # 记录每层输入输出范数
            for k, v in layer_stats.items():
                if 'runs' not in v:
                    v['runs'] = []
                inp_norm = float(v['input'].abs().mean()) if 'input' in v else float('nan')
                out_norm = float(v['output'].abs().mean()) if 'output' in v else float('nan')
                v['runs'].append((inp_norm, out_norm))

    # 解除hook
    for h in hooks:
        h.remove()

    # 比较两次预测差异
    if len(preds) == 2:
        diff = (preds[0] - preds[1]).abs().mean().item()
        print(f"  两次整体预测平均绝对差: {diff:.8e}")
    else:
        print("  预测次数不足2次，无法比较")

    # 分析每层
    for name, _ in dropout_layers:
        stat = layer_stats.get(name, {})
        runs = stat.get('runs', [])
        if not runs:
            print(f"  层 {name}: 未捕获到运行数据")
            continue
        inp_means = [r[0] for r in runs]
        out_means = [r[1] for r in runs]
        var_out = np.var(out_means)
        comment = []
        layer = dict(dropout_layers)[name]
        if layer.p == 0.0:
            comment.append('p=0')
        if all(abs(x) < 1e-8 for x in inp_means):
            comment.append('输入≈0 (上游输出为常数)')
        if var_out < 1e-12:
            comment.append('多次输出均值无波动')
        mode_flag = 'train' if layer.training else 'eval'
        comment.append(f'mode={mode_flag}')
        print(f"  层 {name}: p={layer.p}, 输入均值范围[{min(inp_means):.2e},{max(inp_means):.2e}], 输出均值范围[{min(out_means):.2e},{max(out_means):.2e}], 输出均值方差={var_out:.2e} -> {'; '.join(comment)}")

    print("[Dropout 深度诊断] 结束\n")

def qbc_query_fixed(model_tuple, unlabeled_data, select_num, mc_times=10):
    """
    强化版QBC查询 - 确保dropout必须工作
    """
    model, metric_func, device = model_tuple
    
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    print(f"开始QBC查询，MC采样次数: {mc_times}")
    print("=" * 50)
    
    # === 第1步：强制启用dropout ===
    print("第1步：强制启用dropout层")
    dropout_layers = force_enable_dropout(model, dropout_p=0.15)
    print(f"处理了 {len(dropout_layers)} 个dropout层")
    
    # === 第2步：验证dropout工作状态 ===
    print("\n第2步：验证dropout工作状态")
    
    # 获取一个测试样本来验证dropout
    if len(unlabeled_data) > 0:
        test_sample = unlabeled_data[0]
        
        # 创建测试数据
        timestamp = int(time.time() * 1000000)
        temp_test_file = os.path.join(temp_data_dir, f'dropout_test_{timestamp}.pkl')
        
        with open(temp_test_file, 'wb') as f:
            pickle.dump([test_sample], f)
        
        standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
        shutil.copy2(temp_test_file, standard_test_path)
        
        args = get_al_args()
        args.dataset = 'al_qbc'
        _, test_dataset = get_dataset(args)
        test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
        
        # 获取测试输入
        for g, u_p, g_u in test_loader:
            test_input = (g, u_p, g_u)
            break
        
        # 验证dropout
        dropout_working = verify_dropout_working(model, test_input, device, num_tests=8)
        
        # 清理测试文件
        try:
            os.remove(temp_test_file)
            os.remove(standard_test_path)
        except:
            pass
        
        if not dropout_working:
            print("❌ 警告：Dropout未正常工作！尝试增强dropout...")
            # 进一步增强dropout
            force_enable_dropout(model, dropout_p=0.25)
            dropout_working = verify_dropout_working(model, test_input, device, num_tests=8)
            
            if not dropout_working:
                print("❌ 严重警告：Dropout仍然无效！QBC策略可能退化")
                # 深度诊断
                diagnose_dropout(model, test_input, device)
        else:
            print("✅ Dropout工作正常，可以进行QBC查询")
    
    print("=" * 50)
    
    # === 第3步：执行QBC查询 ===
    variances = []
    failed_samples = 0
    successful_samples = 0
    
    # 获取模型的原始状态
    original_state = model.training
    
    print(f"\n第3步：对 {len(unlabeled_data)} 个样本执行QBC查询")
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="QBC查询")):
        try:
            # 创建临时数据文件
            timestamp = int(time.time() * 1000000) + i + random.randint(1, 1000)
            temp_file = os.path.join(temp_data_dir, f'qbc_temp_{i}_{timestamp}.pkl')
            
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            shutil.copy2(temp_file, standard_test_path)
            
            args = get_al_args()
            args.dataset = 'al_qbc'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            preds = []
            
            # === 强制使用训练模式进行MC采样 ===
            model.train()  # 确保dropout激活
            
            # Monte Carlo采样
            for mc_iter in range(mc_times):
                with torch.no_grad():
                    for g, u_p, g_u in tmp_loader:
                        g = g.to(device)
                        u_p = u_p.to(device)
                        g_u = g_u.to(device)
                        
                        # 确保每次预测都经过dropout
                        pred = model(g, u_p, g_u).cpu().numpy()
                        preds.append(pred.copy())
                        break
            
            # 计算方差
            if len(preds) >= 2:
                preds_array = np.stack(preds, axis=0)
                
                # 多种方差计算方法
                pointwise_var = np.var(preds_array, axis=0)
                mean_var = np.mean(pointwise_var)
                max_var = np.max(pointwise_var)
                total_var = np.var(preds_array)
                std_of_means = np.std(np.mean(preds_array, axis=(1,2)))
                
                # 综合方差指标
                variance = mean_var + 0.2 * max_var + 0.05 * total_var + 1.0 * std_of_means
                variances.append(max(variance, 1e-12))
                
                if variance > 1e-8:
                    successful_samples += 1
                
                if i < 5:  # 详细调试信息
                    print(f"\nQBC样本 {i}: 预测数={len(preds)}")
                    print(f"  预测范围: {np.min(preds_array):.6f} - {np.max(preds_array):.6f}")
                    print(f"  平均方差={mean_var:.8f}, 最大方差={max_var:.8f}")
                    print(f"  均值标准差={std_of_means:.8f}, 综合方差={variance:.8f}")
                    
            else:
                variances.append(1e-12)
                failed_samples += 1
                
            # 清理临时文件
            try:
                os.remove(temp_file)
                if os.path.exists(standard_test_path):
                    os.remove(standard_test_path)
            except:
                pass
                
        except Exception as e:
            if i < 5:
                print(f"\nQBC样本 {i} 失败: {e}")
            variances.append(1e-12)
            failed_samples += 1
    
    # 恢复模型原始状态
    model.train(original_state)
    
    # === 第4步：结果分析 ===
    print(f"\n第4步：QBC查询结果分析")
    print("=" * 50)
    print(f"总样本数: {len(unlabeled_data)}")
    print(f"失败样本: {failed_samples}")
    print(f"成功样本: {successful_samples}")
    print(f"成功率: {successful_samples / len(unlabeled_data) * 100:.1f}%")
    
    valid_variances = [v for v in variances if v > 1e-10]
    if valid_variances:
        print(f"有效方差数量: {len(valid_variances)}")
        print(f"方差范围: {min(valid_variances):.8f} - {max(valid_variances):.8f}")
        print(f"平均方差: {np.mean(valid_variances):.8f}")
        print(f"方差中位数: {np.median(valid_variances):.8f}")
    
    # 选择高不确定性样本
    if successful_samples < select_num * 0.1:  # 如果成功样本太少
        print("❌ QBC严重失败：有效样本太少，使用随机采样")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    # 选择方差最高的样本
    selected_idx = np.argsort(variances)[-select_num:]
    selected_variances = [variances[i] for i in selected_idx]
    
    print(f"✅ QBC选择完成")
    print(f"选中样本方差统计:")
    print(f"  最小: {min(selected_variances):.8f}")
    print(f"  最大: {max(selected_variances):.8f}")
    print(f"  平均: {np.mean(selected_variances):.8f}")
    print("=" * 50)
    
    return selected_idx.tolist()
    
    # 结果统计
    print(f"\nQBC查询统计:")
    print(f"  总样本: {len(unlabeled_data)}")
    print(f"  失败样本: {failed_samples}")
    print(f"  成功率: {(len(unlabeled_data) - failed_samples) / len(unlabeled_data) * 100:.1f}%")
    
    valid_variances = [v for v in variances if v > 0]
    if valid_variances:
        print(f"  有效方差数量: {len(valid_variances)}")
        print(f"  方差范围: {min(valid_variances):.8f} - {max(valid_variances):.8f}")
        print(f"  平均方差: {np.mean(valid_variances):.8f}")
    
    # 选择高不确定性样本
    if all(v == 0.0 for v in variances):
        print("QBC失败：所有方差为0，使用随机采样")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    selected_idx = np.argsort(variances)[-select_num:]
    selected_variances = [variances[i] for i in selected_idx]
    
    print(f"QBC选择完成，选中样本方差: {selected_variances}")
    return selected_idx.tolist()

def pa_query(model_tuple, unlabeled_data, select_num, rho=1.0, mu=1.0, lam=1e-4):
    """
    Physics-Aware查询 - 选择违反物理定律最严重的样本
    修改为使用独立的中间数据目录
    """
    model, metric_func, device = model_tuple
    print(f"PA查询开始，设备: {device}, 样本数: {len(unlabeled_data)}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_pa/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"中间数据目录: {temp_data_dir}")
    
    model.eval()
    scores = []
    
    for i, sample in enumerate(unlabeled_data):
        if i % 50 == 0:
            print(f"PA进度: {i}/{len(unlabeled_data)}")
        
        # 使用时间戳创建唯一的临时文件
        import time
        timestamp = int(time.time() * 1000000)
        temp_file = os.path.join(temp_data_dir, f'pa_temp_{i}_{timestamp}.pkl')
        
        try:
            # 为每个样本构建临时数据集
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            # 复制到标准位置
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            args = get_al_args()
            args.dataset = 'al_pa'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            with torch.no_grad():
                for batch_idx, (g, u_p, g_u) in enumerate(tmp_loader):
                    # ===== 完整的DGL图设备转移 =====
                    # 方法1：整体转移图到设备
                    try:
                        g = g.to(device)
                    except:
                        # 方法2：手动转移所有图数据
                        if hasattr(g, 'ndata'):
                            for key in g.ndata:
                                if torch.is_tensor(g.ndata[key]):
                                    g.ndata[key] = g.ndata[key].to(device)
                        if hasattr(g, 'edata'):
                            for key in g.edata:
                                if torch.is_tensor(g.edata[key]):
                                    g.edata[key] = g.edata[key].to(device)
                    
                    # 转移其他输入
                    u_p = u_p.to(device)
                    g_u = g_u.to(device)
                    
                    # 确保模型在正确设备上
                    model = model.to(device)
                    
                    # 模型预测
                    Y_pred = model(g, u_p, g_u)
                    Y_pred = Y_pred.cpu().numpy()
                    break
                break
            
            # 计算物理一致性评分
            try:
                L_cont = np.mean(np.abs(divergence(Y_pred, sample[0])))
                gradY = np.gradient(Y_pred, axis=0)
                momentum = rho * np.sum(Y_pred * gradY, axis=1) - mu * laplacian(Y_pred, sample[0])
                L_mom = np.mean(np.linalg.norm(momentum, axis=-1))
                L_ns = L_cont + lam * L_mom
            except Exception as e:
                print(f"物理评分计算失败: {e}")
                L_ns = 0.0
                
            scores.append(L_ns)
            
        except Exception as e:
            print(f"PA样本 {i} 处理失败: {e}")
            scores.append(0.0)
        finally:
            # 清理临时文件
            try:
                os.remove(temp_file)
                if os.path.exists(standard_test_path):
                    os.remove(standard_test_path)
            except:
                pass
    
    # 选择物理不一致性最高的样本
    if len(scores) == 0:
        print("警告：PA查询失败，使用随机采样")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    selected_idx = np.argsort(scores)[-select_num:]
    print(f"PA查询完成，选择了 {len(selected_idx)} 个物理不一致样本")
    return selected_idx.tolist()

# 同时修复divergence和laplacian函数，添加错误处理
def divergence(Y, X):
    """计算散度，添加错误处理"""
    try:
        # 尝试导入scipy
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        div = []
        for i, x in enumerate(X):
            idx = tree.query_ball_point(x, r=1e-2)
            if len(idx) < 2:
                div.append(0)
                continue
            grad = (Y[idx] - Y[i]).mean(axis=0) / (X[idx] - x + 1e-8).mean(axis=0)
            div.append(np.sum(grad))
        return np.array(div)
    except ImportError:
        print("scipy不可用，使用简化的散度计算")
        # 简化版本：使用有限差分
        div = []
        for i in range(len(Y)):
            if i == 0 or i == len(Y) - 1:
                div.append(0)
            else:
                grad_approx = (Y[i+1] - Y[i-1]) / 2
                div.append(np.sum(grad_approx))
        return np.array(div)
    except Exception as e:
        print(f"散度计算失败: {e}")
        return np.zeros(len(Y))

def laplacian(Y, X):
    """计算拉普拉斯算子，添加错误处理"""
    try:
        # 尝试导入scipy
        from scipy.spatial import cKDTree
        tree = cKDTree(X)
        lap = []
        for i, x in enumerate(X):
            idx = tree.query_ball_point(x, r=1e-2)
            if len(idx) < 2:
                lap.append(0)
                continue
            lap.append((Y[idx] - Y[i]).mean(axis=0).sum())
        return np.array(lap)
    except ImportError:
        print("scipy不可用，使用简化的拉普拉斯计算")
        # 简化版本：使用二阶有限差分
        lap = []
        for i in range(len(Y)):
            if i <= 1 or i >= len(Y) - 2:
                lap.append(0)
            else:
                laplacian_approx = Y[i+1] - 2*Y[i] + Y[i-1]
                lap.append(np.sum(laplacian_approx))
        return np.array(lap)
    except Exception as e:
        print(f"拉普拉斯计算失败: {e}")
        return np.zeros(len(Y))


def random_active_learning_with_logging(
    rounds=5, select_num=5, seed=42, 
    output_dir='./al_rounds', data_update_dir=None, strategy='random'
):
    """
    带详细日志的主动学习主循环
    """
    # ... 前面的代码保持不变 ...
    
    # 从数据更新目录读取数据
    if not data_update_dir:
        raise ValueError("必须指定 data_update_dir 参数")
    
    labeled_path = os.path.join(data_update_dir, 'al_labeled.pkl')
    unlabeled_path = os.path.join(data_update_dir, 'al_unlabeled.pkl')
    test_path = os.path.join(data_update_dir, 'al_test.pkl')
    
    # 检查必要文件是否存在
    required_files = [labeled_path, unlabeled_path, test_path]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"错误：以下必要文件不存在:")
        for f in missing_files:
            print(f"  - {f}")
        return
    
    # 加载数据
    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(unlabeled_path, 'rb') as f:
        unlabeled_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    print(f"从 {data_update_dir} 加载数据成功:")
    print(f"  - 已标注训练集: {len(labeled_data)} 个样本")
    print(f"  - 未标注池: {len(unlabeled_data)} 个样本")
    print(f"  - 测试集: {len(test_data)} 个样本")
    
    # 确保中间数据目录存在
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # 准备CSV文件记录性能指标
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear', 'train_num', 'round'])

    for r in range(rounds):
        print(f"\n=== 主动学习第{r+1}轮 ===")
        
        # 创建本轮日志文件
        round_data_dir = os.path.join(output_dir, f'round_{r+1}')
        os.makedirs(round_data_dir, exist_ok=True)
        log_file = os.path.join(round_data_dir, 'round_log.txt')
        
        # 日志记录函数
        def log_and_print(message):
            print(message)
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(message + '\n')
        
        log_and_print(f"=== 主动学习第{r+1}轮开始 ===")
        log_and_print(f"时间: {pd.Timestamp.now()}")
        log_and_print(f"策略: {strategy}")
        log_and_print(f"当前数据状态:")
        log_and_print(f"  - 已标注训练集: {len(labeled_data)} 个样本")
        log_and_print(f"  - 未标注池: {len(unlabeled_data)} 个样本")
        log_and_print(f"  - 测试集: {len(test_data)} 个样本")
        
        # === 1. 更新数据文件到指定目录 ===
        with open(labeled_path, 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(unlabeled_path, 'wb') as f:
            pickle.dump(unlabeled_data, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        log_and_print(f"数据文件已更新到: {data_update_dir}")
        
        # === 2. 保存轮次结果 ===
        with open(os.path.join(round_data_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(os.path.join(round_data_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        
        # === 3. 训练和评估 ===
        log_and_print("开始模型训练...")
        train_start_time = pd.Timestamp.now()
        
        model_tuple = train_model(labeled_data)
        
        train_end_time = pd.Timestamp.now()
        train_duration = train_end_time - train_start_time
        log_and_print(f"训练完成，耗时: {train_duration}")
        
        log_and_print("开始模型评估...")
        eval_start_time = pd.Timestamp.now()
        
        metric = evaluate_model(model_tuple, test_data)
        
        eval_end_time = pd.Timestamp.now()
        eval_duration = eval_end_time - eval_start_time
        log_and_print(f"评估完成，耗时: {eval_duration}")
        log_and_print(f"模型性能指标: {metric}")

        # 记录性能指标到CSV
        train_num = len(labeled_data)
        if isinstance(metric, (np.ndarray, list)) and len(metric) == 5:
            row = list(metric) + [train_num, r+1]
        else:
            row = [metric]*5 + [train_num, r+1]
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        # === 4. 检查终止条件 ===
        if len(unlabeled_data) == 0:
            log_and_print("未标注池已空，主动学习提前结束。")
            break
            
        if len(unlabeled_data) < select_num:
            select_num = len(unlabeled_data)
            log_and_print(f"调整选择数量为: {select_num}")
            
        # === 5. 策略选择新样本 ===
        log_and_print(f"开始{strategy}策略选择...")
        log_and_print(f"从 {len(unlabeled_data)} 个候选中选择 {select_num} 个")
        
        strategy_start_time = pd.Timestamp.now()
        
        # 重定向策略输出到日志
        import sys
        from io import StringIO
        
        # 捕获策略输出
        old_stdout = sys.stdout
        strategy_output = StringIO()
        sys.stdout = strategy_output
        
        try:
            if strategy == 'random':
                newly_selected = random.sample(unlabeled_data, select_num)
                selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                
            elif strategy == 'gv':
                selected_idx = geometry_variance_query(labeled_data, unlabeled_data, select_num)
                newly_selected = [unlabeled_data[i] for i in selected_idx]
                
            elif strategy == 'gv_fast':
                selected_idx = geometry_variance_query_fast(labeled_data, unlabeled_data, select_num)
                newly_selected = [unlabeled_data[i] for i in selected_idx]
                
            elif strategy == 'qbc':
                selected_idx = qbc_query_fixed(model_tuple, unlabeled_data, select_num)
                newly_selected = [unlabeled_data[i] for i in selected_idx]
                
            elif strategy == 'pa':
                selected_idx = pa_query(model_tuple, unlabeled_data, select_num)
                newly_selected = [unlabeled_data[i] for i in selected_idx]
                
            else:
                log_and_print(f"未知策略: {strategy}, 使用随机采样")
                newly_selected = random.sample(unlabeled_data, select_num)
                selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                
        except Exception as e:
            log_and_print(f"{strategy}策略失败: {e}, 回退到随机采样")
            newly_selected = random.sample(unlabeled_data, select_num)
            selected_idx = [unlabeled_data.index(x) for x in newly_selected]
        
        finally:
            # 恢复标准输出
            sys.stdout = old_stdout
            
            # 记录策略输出到日志
            strategy_log_content = strategy_output.getvalue()
            if strategy_log_content.strip():
                log_and_print("=== 策略执行日志 ===")
                for line in strategy_log_content.strip().split('\n'):
                    log_and_print(line)
                log_and_print("=== 策略执行日志结束 ===")
        
        strategy_end_time = pd.Timestamp.now()
        strategy_duration = strategy_end_time - strategy_start_time
        log_and_print(f"策略选择完成，耗时: {strategy_duration}")
        log_and_print(f"选中样本索引: {selected_idx}")

        # === 6. 保存策略信息 ===
        strategy_info = {
            'strategy': strategy,
            'select_num': len(newly_selected),
            'selected_indices': selected_idx,
            'round': r + 1,
            'total_labeled': len(labeled_data),
            'total_unlabeled': len(unlabeled_data),
            'train_duration_seconds': train_duration.total_seconds(),
            'eval_duration_seconds': eval_duration.total_seconds(),
            'strategy_duration_seconds': strategy_duration.total_seconds(),
            'timestamp': pd.Timestamp.now().isoformat(),
            'performance_metric': metric.tolist() if hasattr(metric, 'tolist') else metric
        }
        
        with open(os.path.join(round_data_dir, 'strategy_info.json'), 'w') as f:
            json.dump(strategy_info, f, indent=2)

        # === 7. 更新数据集 ===
        labeled_data.extend(newly_selected)
        newly_selected_ids = set(id(x) for x in newly_selected)
        unlabeled_data = [x for x in unlabeled_data if id(x) not in newly_selected_ids]

        log_and_print(f"第{r+1}轮完成:")
        log_and_print(f"  - 本轮新增样本: {len(newly_selected)}")
        log_and_print(f"  - 累计已标注: {len(labeled_data)}")
        log_and_print(f"  - 剩余未标注: {len(unlabeled_data)}")
        log_and_print(f"  - 本轮总耗时: {strategy_end_time - train_start_time}")
        log_and_print("=" * 50)

    # === 8. 最终更新和总结 ===
    with open(labeled_path, 'wb') as f:
        pickle.dump(labeled_data, f)
    with open(unlabeled_path, 'wb') as f:
        pickle.dump(unlabeled_data, f)
    
    print(f"\n=== 主动学习流程结束 ===")
    print(f"结果保存位置:")
    print(f"  - 性能记录: {csv_path}")
    print(f"  - 轮次结果和日志: {output_dir}")


    
# === 5. GPU选择函数 ===
def select_best_gpu():
    """智能选择最佳GPU"""
    try:
        # 获取GPU信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True, check=True)
        gpu_info = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                gpu_id = int(parts[0])
                free_mem = int(parts[1])
                total_mem = int(parts[2])
                gpu_info.append((gpu_id, free_mem, total_mem, free_mem/total_mem))
        
        if not gpu_info:
            print("未检测到GPU，使用CPU")
            return None
        
        # 按可用内存排序，选择最佳GPU
        gpu_info.sort(key=lambda x: x[1], reverse=True)  # 按可用内存排序
        best_gpu = gpu_info[0]
        
        print("GPU状态:")
        for gpu_id, free_mem, total_mem, ratio in gpu_info:
            status = "✓ 选中" if gpu_id == best_gpu[0] else ""
            print(f"  GPU {gpu_id}: {free_mem}MB / {total_mem}MB 可用 ({ratio:.1%}) {status}")
        
        selected_gpu = best_gpu[0]
        print(f"\n自动选择GPU {selected_gpu} (可用内存: {best_gpu[1]}MB)")
        return selected_gpu
        
    except Exception as e:
        print(f"GPU检测失败: {e}")
        print("默认使用GPU 0")
        return 0

# === 修改主程序调用 ===
if __name__ == "__main__":
    import subprocess
    
    # === 设置完全确定性训练 ===
    print("=== 设置确定性训练环境 ===")
    set_deterministic_training(42)
    print("已启用完全确定性训练，相同初始数据将产生相同结果\n")
    
    # 智能选择GPU
    selected_gpu = select_best_gpu()
    if selected_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
        print(f"设置CUDA_VISIBLE_DEVICES={selected_gpu}")
    else:
        print("将使用CPU进行训练")
    
    # 策略选择
    strategy = 'qbc'  # 当前使用random策略
    # strategy = 'gv'        # 高精度几何方法
    # strategy = 'qbc'       # 查询委员会
    # strategy = 'pa'        # 物理感知
    
    print(f"=== 策略选择 ===")
    print(f"当前使用策略: {strategy}")
    
    # 定义路径
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc"  # 数据更新目录
    output_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc/al_rounds17"  # 结果保存目录
    temp_data_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc/data"  # 中间数据目录
    
    print(f"=== 目录配置 ===")
    print(f"数据更新目录: {data_update_dir}")
    print(f"  - 程序将从此目录读取初始数据，并每轮更新数据文件")
    print(f"  - 需要的文件: al_labeled.pkl, al_unlabeled.pkl, al_test.pkl")
    print(f"结果保存目录: {output_dir}")
    print(f"  - 每轮保存 train_data.pkl, test_data.pkl, strategy_info.json")
    print(f"中间数据目录: {temp_data_dir}")
    print(f"  - 临时训练和测试文件，不影响 /home/v-wenliao/gnot/GNOT/data/al_labeled.pkl")
    
    # 创建中间数据目录
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # 检查数据目录
    print(f"\n=== 数据文件检查 ===")
    required_files = [
        os.path.join(data_update_dir, 'al_labeled.pkl'),
        os.path.join(data_update_dir, 'al_unlabeled.pkl'),
        os.path.join(data_update_dir, 'al_test.pkl')
    ]
    
    print("检查必要文件:")
    all_exist = True
    for file_path in required_files:
        exists = os.path.exists(file_path)
        status = "✓ 存在" if exists else "✗ 缺失"
        print(f"  - {os.path.basename(file_path)}: {status}")
        if not exists:
            all_exist = False
    
    # 检查原始数据不会被影响

    # === 修改数据集名称为 al_qbc ===
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc"  # 数据更新目录
    output_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc/al_rounds17"  # 结果保存目录
    temp_data_dir = "/home/v-wenliao/gnot/GNOT/data/al_qbc/data"  # 中间数据目录
    original_file = "/home/v-wenliao/gnot/GNOT/data/al_labeled.pkl"
    if os.path.exists(original_file):
        print(f"✓ 原始文件 {original_file} 存在，不会被影响")
    else:
        print(f"ℹ 原始文件 {original_file} 不存在，这是正常的")
    
    if not all_exist:
        print(f"\n请确保在 {data_update_dir} 目录中放置所有必要的数据文件后再运行程序。")
    else:
        print(f"\n所有数据文件准备就绪，开始主动学习...")
        
        # 启动主动学习
        random_active_learning_with_logging(
            rounds=6,
            select_num=100,
            seed=42,
            output_dir=output_dir,  # 结果保存目录
            data_update_dir=data_update_dir,  # 数据更新目录
            strategy=strategy
        )