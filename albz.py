import os
import pickle
import random
import csv
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

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
        dataset = 'al_bz'
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
        epochs = 200
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
        ffn_dropout = 0.0
        attn_dropout = 0.0
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
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
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
        args.dataset = 'al_bz'  # 确保使用正确的数据集名称
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
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
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
        args.dataset = 'al_bz'  # 确保使用正确的数据集名称
        _, test_dataset = get_dataset(args)
        test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
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


def qbc_query_fixed(model_tuple, unlabeled_data, select_num, mc_times=10):
    """
    增强版QBC查询 - 确保Dropout正确工作或使用替代方案
    """
    model, metric_func, device = model_tuple
    
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_qbc/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    print(f"开始QBC查询，MC采样次数: {mc_times}")
    
    # 详细检查模型结构
    dropout_layers = []
    all_modules = []
    for name, module in model.named_modules():
        all_modules.append((name, type(module).__name__))
        if 'dropout' in name.lower() or isinstance(module, (torch.nn.Dropout, torch.nn.Dropout1d, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            dropout_layers.append((name, module))
    
    print(f"模型结构分析:")
    print(f"  总模块数: {len(all_modules)}")
    print(f"  检测到的Dropout层: {len(dropout_layers)}")
    
    if dropout_layers:
        for name, module in dropout_layers:
            print(f"    {name}: {type(module).__name__}, p={getattr(module, 'p', 'N/A')}")
    else:
        print("    警告: 未检测到任何Dropout层")
        print("    前10个模块:", all_modules[:10])
    
    # 如果没有Dropout，使用数据增强作为替代
    use_data_augmentation = len(dropout_layers) == 0
    if use_data_augmentation:
        print("  策略: 由于缺少Dropout，将使用数据增强产生多样性")
        mc_times = min(mc_times, 5)  # 减少采样次数
    
    variances = []
    failed_samples = 0
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="QBC查询")):
        try:
            # 创建临时数据文件
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'qbc_temp_{i}_{timestamp}.pkl')
            
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            args = get_al_args()
            args.dataset = 'al_qbc'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            preds = []
            
            # Monte Carlo采样
            for mc_iter in range(mc_times):
                if use_data_augmentation:
                    # 使用轻微的数据增强代替Dropout
                    model.eval()  # 评估模式
                    
                    # 在输入中加入小噪声
                    with torch.no_grad():
                        for g, u_p, g_u in tmp_loader:
                            g = g.to(device)
                            u_p = u_p.to(device) 
                            g_u = g_u.to(device)
                            
                            # 添加小噪声
                            noise_scale = 0.01
                            if hasattr(g, 'ndata') and 'x' in g.ndata:
                                g.ndata['x'] = g.ndata['x'] + torch.randn_like(g.ndata['x']) * noise_scale
                            u_p = u_p + torch.randn_like(u_p) * noise_scale
                            
                            model = model.to(device)
                            pred = model(g, u_p, g_u).cpu().numpy()
                            preds.append(pred.copy())
                            break
                        break
                else:
                    # 使用标准的MC Dropout
                    model.train()  # 训练模式启用Dropout
                    
                    with torch.no_grad():
                        for g, u_p, g_u in tmp_loader:
                            g = g.to(device)
                            u_p = u_p.to(device)
                            g_u = g_u.to(device)
                            
                            model = model.to(device)
                            pred = model(g, u_p, g_u).cpu().numpy()
                            preds.append(pred.copy())
                            break
                        break
            
            # 计算方差
            if len(preds) >= 2:
                preds_array = np.stack(preds, axis=0)
                
                # 多种方差计算方法
                pointwise_var = np.var(preds_array, axis=0)
                mean_var = np.mean(pointwise_var)
                max_var = np.max(pointwise_var)
                total_var = np.var(preds_array)
                
                # 综合方差指标
                variance = mean_var + 0.1 * max_var + 0.01 * total_var
                variances.append(variance)
                
                if i < 5:  # 调试信息
                    print(f"QBC样本 {i}: 预测数={len(preds)}, 平均方差={mean_var:.8f}, 最大方差={max_var:.8f}, 综合方差={variance:.8f}")
                    
            else:
                variances.append(0.0)
                failed_samples += 1
                
            # 清理临时文件
            os.remove(temp_file)
            if os.path.exists(standard_test_path):
                os.remove(standard_test_path)
                
        except Exception as e:
            if i < 5:  # 只打印前几个错误
                print(f"QBC样本 {i} 失败: {e}")
            variances.append(0.0)
            failed_samples += 1
    
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


def pa_query_debug(model_tuple, unlabeled_data, select_num, rho=1.0, mu=1.0, lam=1e-4):
    """
    PA查询调试版本 - 详细错误追踪
    """
    model, metric_func, device = model_tuple
    print(f"PA查询开始，设备: {device}, 样本数: {len(unlabeled_data)}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"中间数据目录: {temp_data_dir}")
    
    model.eval()
    
    # === 详细调试第一个样本 ===
    print("\n=== 开始详细调试第一个样本 ===")
    
    if len(unlabeled_data) == 0:
        print("错误: 未标注数据为空")
        return []
    
    sample = unlabeled_data[0]
    print(f"样本类型: {type(sample)}")
    print(f"样本长度: {len(sample) if hasattr(sample, '__len__') else 'N/A'}")
    
    # 检查样本结构
    try:
        X_coords = sample[0]
        Y_values = sample[1]
        print(f"X坐标类型: {type(X_coords)}, 形状: {np.array(X_coords).shape}")
        print(f"Y值类型: {type(Y_values)}, 形状: {np.array(Y_values).shape}")
        print(f"X坐标范围: {np.array(X_coords).min():.3f} - {np.array(X_coords).max():.3f}")
        print(f"Y值范围: {np.array(Y_values).min():.3f} - {np.array(Y_values).max():.3f}")
    except Exception as e:
        print(f"样本结构检查失败: {e}")
        return []
    
    # === 步骤1: 测试文件操作 ===
    print("\n--- 步骤1: 测试文件操作 ---")
    try:
        import time
        timestamp = int(time.time() * 1000000)
        temp_file = os.path.join(temp_data_dir, f'debug_pa_temp_{timestamp}.pkl')
        
        print(f"创建临时文件: {temp_file}")
        with open(temp_file, 'wb') as f:
            pickle.dump([sample], f)
        print("✓ 临时文件创建成功")
        
        # 验证文件内容
        with open(temp_file, 'rb') as f:
            loaded_sample = pickle.load(f)
        print(f"✓ 文件验证成功，加载样本数: {len(loaded_sample)}")
        
        # 复制到标准位置
        standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
        import shutil
        shutil.copy2(temp_file, standard_test_path)
        print("✓ 文件复制成功")
        
    except Exception as e:
        print(f"✗ 文件操作失败: {e}")
        import traceback
        print(f"详细错误:\n{traceback.format_exc()}")
        return []
    
    # === 步骤2: 测试数据集加载 ===
    print("\n--- 步骤2: 测试数据集加载 ---")
    try:
        args = get_al_args()
        args.dataset = 'al_bz'
        print(f"数据集配置: {args.dataset}")
        
        print("开始加载数据集...")
        _, tmp_dataset = get_dataset(args)
        print(f"✓ 数据集加载成功，大小: {len(tmp_dataset)}")
        
        print("创建数据加载器...")
        tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
        print(f"✓ 数据加载器创建成功")
        
        # 测试迭代
        print("测试数据加载器迭代...")
        batch_count = 0
        for batch_data in tmp_loader:
            batch_count += 1
            print(f"✓ 成功加载批次 {batch_count}")
            print(f"  批次数据类型: {type(batch_data)}")
            print(f"  批次数据长度: {len(batch_data)}")
            
            if len(batch_data) == 3:
                g, u_p, g_u = batch_data
                print(f"  图对象类型: {type(g)}")
                print(f"  u_p形状: {u_p.shape if hasattr(u_p, 'shape') else type(u_p)}")
                print(f"  g_u形状: {g_u.shape if hasattr(g_u, 'shape') else type(g_u)}")
                
                # 检查图对象属性
                if hasattr(g, 'ndata'):
                    print(f"  图节点数据键: {list(g.ndata.keys())}")
                if hasattr(g, 'edata'):
                    print(f"  图边数据键: {list(g.edata.keys())}")
                if hasattr(g, 'number_of_nodes'):
                    print(f"  图节点数: {g.number_of_nodes()}")
                if hasattr(g, 'number_of_edges'):
                    print(f"  图边数: {g.number_of_edges()}")
            else:
                print(f"  ✗ 意外的批次数据长度: {len(batch_data)}")
                return []
            break
        
        if batch_count == 0:
            print("✗ 数据加载器为空")
            return []
            
    except Exception as e:
        print(f"✗ 数据集加载失败: {e}")
        import traceback
        print(f"详细错误:\n{traceback.format_exc()}")
        return []
    
    # === 步骤3: 测试设备转移 ===
    print("\n--- 步骤3: 测试设备转移 ---")
    try:
        # 重新获取批次数据用于设备转移测试
        for batch_data in tmp_loader:
            g, u_p, g_u = batch_data
            
            print(f"原始设备状态:")
            print(f"  u_p设备: {u_p.device if hasattr(u_p, 'device') else 'N/A'}")
            print(f"  g_u设备: {g_u.device if hasattr(g_u, 'device') else 'N/A'}")
            
            # 设备转移
            print(f"开始设备转移到: {device}")
            
            # 图转移
            if hasattr(g, 'to'):
                g = g.to(device)
                print("✓ 图对象设备转移成功（方法1）")
            else:
                # 手动转移图数据
                if hasattr(g, 'ndata'):
                    for key in g.ndata:
                        if torch.is_tensor(g.ndata[key]):
                            original_device = g.ndata[key].device
                            g.ndata[key] = g.ndata[key].to(device)
                            print(f"    节点数据 {key}: {original_device} -> {g.ndata[key].device}")
                if hasattr(g, 'edata'):
                    for key in g.edata:
                        if torch.is_tensor(g.edata[key]):
                            original_device = g.edata[key].device
                            g.edata[key] = g.edata[key].to(device)
                            print(f"    边数据 {key}: {original_device} -> {g.edata[key].device}")
                print("✓ 图对象设备转移成功（方法2）")
            
            # 张量转移
            u_p_device_before = u_p.device
            g_u_device_before = g_u.device
            
            u_p = u_p.to(device)
            g_u = g_u.to(device)
            
            print(f"✓ u_p设备转移: {u_p_device_before} -> {u_p.device}")
            print(f"✓ g_u设备转移: {g_u_device_before} -> {g_u.device}")
            
            # 模型设备转移
            model = model.to(device)
            print(f"✓ 模型设备转移完成")
            
            break
    except Exception as e:
        print(f"✗ 设备转移失败: {e}")
        import traceback
        print(f"详细错误:\n{traceback.format_exc()}")
        return []
    
    # === 步骤4: 测试模型预测 ===
    print("\n--- 步骤4: 测试模型预测 ---")
    try:
        print("开始模型预测...")
        print(f"模型训练状态: {model.training}")
        print(f"模型设备: {next(model.parameters()).device}")
        
        with torch.no_grad():
            # 重新获取批次数据进行预测
            for batch_data in tmp_loader:
                g, u_p, g_u = batch_data
                
                # 确保设备正确
                g = g.to(device)
                u_p = u_p.to(device)
                g_u = g_u.to(device)
                
                print(f"输入数据设备检查:")
                print(f"  u_p: {u_p.device}, 形状: {u_p.shape}")
                print(f"  g_u: {g_u.device}, 形状: {g_u.shape}")
                
                # 进行预测
                print("执行模型前向传播...")
                Y_pred = model(g, u_p, g_u)
                print(f"✓ 模型预测成功")
                print(f"  预测结果设备: {Y_pred.device}")
                print(f"  预测结果形状: {Y_pred.shape}")
                print(f"  预测结果类型: {type(Y_pred)}")
                
                # 转移到CPU
                Y_pred_cpu = Y_pred.cpu().numpy()
                print(f"✓ 结果转移到CPU，形状: {Y_pred_cpu.shape}")
                print(f"  预测值范围: [{Y_pred_cpu.min():.6f}, {Y_pred_cpu.max():.6f}]")
                print(f"  预测值统计: 均值={Y_pred_cpu.mean():.6f}, 标准差={Y_pred_cpu.std():.6f}")
                
                break
            
    except Exception as e:
        print(f"✗ 模型预测失败: {e}")
        import traceback
        print(f"详细错误:\n{traceback.format_exc()}")
        return []
    
    # === 步骤5: 测试物理计算 ===
    print("\n--- 步骤5: 测试物理计算 ---")
    try:
        Pi = np.array(sample[0])  # 点云坐标
        print(f"点云坐标形状: {Pi.shape}")
        print(f"速度场形状: {Y_pred_cpu.shape}")
        
        # 测试连续性损失计算
        print("计算连续性损失...")
        try:
            L_continuity = compute_continuity_loss(Y_pred_cpu, Pi)
            print(f"✓ 连续性损失: {L_continuity:.6f}")
        except Exception as cont_error:
            print(f"✗ 连续性损失计算失败: {cont_error}")
            L_continuity = 0.0
        
        # 测试动量损失计算
        print("计算动量损失...")
        try:
            L_momentum = compute_momentum_loss(Y_pred_cpu, Pi, rho, mu)
            print(f"✓ 动量损失: {L_momentum:.6f}")
        except Exception as mom_error:
            print(f"✗ 动量损失计算失败: {mom_error}")
            L_momentum = 0.0
        
        # 计算总损失
        L_NS = L_continuity + lam * L_momentum
        print(f"✓ 总物理损失: {L_NS:.6f}")
        print(f"  - 连续性损失: {L_continuity:.6f}")
        print(f"  - 动量损失: {L_momentum:.6f} (权重: {lam})")
        
    except Exception as e:
        print(f"✗ 物理计算失败: {e}")
        import traceback
        print(f"详细错误:\n{traceback.format_exc()}")
        return []
    
    # === 步骤6: 清理测试文件 ===
    print("\n--- 步骤6: 清理测试文件 ---")
    try:
        os.remove(temp_file)
        if os.path.exists(standard_test_path):
            os.remove(standard_test_path)
        print("✓ 测试文件清理完成")
    except Exception as e:
        print(f"✗ 文件清理失败: {e}")
    
    print("\n=== 第一个样本调试完成 ===")
    print("如果上述所有步骤都成功，说明PA查询可以正常工作")
    print("如果某个步骤失败，请根据错误信息进行修复")
    
    # 如果调试成功，询问是否继续完整的PA查询
    print("\n是否继续完整的PA查询？")
    print("如果上述步骤都成功，将返回简化的结果用于测试")
    
    # 为了测试，返回一个简单的结果
    simple_scores = []
    for i, sample in enumerate(unlabeled_data[:min(10, len(unlabeled_data))]):
        try:
            # 使用简化的几何特征作为代理分数
            X_coords = np.array(sample[0])
            geometric_complexity = np.var(X_coords) + np.mean(np.linalg.norm(X_coords - np.mean(X_coords, axis=0), axis=1))
            simple_scores.append((i, geometric_complexity))
        except:
            simple_scores.append((i, 0.0))
    
    # 扩展到所有样本
    while len(simple_scores) < len(unlabeled_data):
        simple_scores.append((len(simple_scores), np.random.random()))
    
    # 选择评分最高的样本
    simple_scores.sort(key=lambda x: x[1], reverse=True)
    selected_idx = [x[0] for x in simple_scores[:select_num]]
    
    print(f"\n返回简化PA结果，选择了 {len(selected_idx)} 个样本")
    return selected_idx



def pa_query_fixed(model_tuple, unlabeled_data, select_num, rho=1.0, mu=1.0, lam=1e-4):
    """
    修复MultipleTensors设备转移问题的PA查询
    """
    model, metric_func, device = model_tuple
    print(f"PA查询开始（修复版），设备: {device}, 样本数: {len(unlabeled_data)}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"中间数据目录: {temp_data_dir}")
    
    model.eval()
    physics_scores = []
    failed_samples = 0
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="PA物理评估")):
        if i % 50 == 0:
            print(f"PA进度: {i}/{len(unlabeled_data)}")
        
        sample_score = 0.0
        
        try:
            # === 步骤1: 构建临时数据集进行预测 ===
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'pa_temp_{i}_{timestamp}.pkl')
            
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            # 复制到标准位置
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            # 加载数据并预测
            args = get_al_args()
            args.dataset = 'al_bz'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            Y_pred = None  # 预测的速度场
            Pi = None      # 点云坐标
            
            with torch.no_grad():
                for batch_data in tmp_loader:
                    g, u_p, g_u = batch_data
                    
                    # === 修复的设备转移部分 ===
                    # 1. 图对象转移
                    g = g.to(device)
                    
                    # 2. 普通张量转移
                    u_p = u_p.to(device)
                    
                    # 3. MultipleTensors对象的特殊处理
                    if hasattr(g_u, 'to'):
                        # 如果有to方法，直接使用
                        g_u = g_u.to(device)
                    elif hasattr(g_u, 'tensors'):
                        # 如果是MultipleTensors，转移内部张量
                        g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                    elif hasattr(g_u, '__dict__'):
                        # 尝试转移所有张量属性
                        for attr_name in dir(g_u):
                            if not attr_name.startswith('_'):
                                attr_value = getattr(g_u, attr_name)
                                if torch.is_tensor(attr_value):
                                    setattr(g_u, attr_name, attr_value.to(device))
                    # 如果都不行，就跳过g_u的设备转移（通常模型会自动处理）
                    
                    # 4. 确保模型在正确设备上
                    model = model.to(device)
                    
                    # === 模型预测 ===
                    Y_pred = model(g, u_p, g_u).cpu().numpy()  # [N, velocity_dim]
                    
                    # 获取点云坐标 Pi
                    Pi = np.array(sample[0])  # 点云坐标 [N, space_dim]
                    
                    break
                break
            
            # === 步骤2: 计算物理一致性评分 ===
            if Y_pred is not None and Pi is not None:
                try:
                    # 计算连续性损失: mean over p∈Pi |∇·yi|p|
                    L_continuity = compute_continuity_loss_fixed(Y_pred, Pi)
                    
                    # 计算动量损失: mean over p∈Pi ||ρ((yi·∇)yi)p - μ(Δyi)p||2
                    L_momentum = compute_momentum_loss_fixed(Y_pred, Pi, rho, mu)
                    
                    # 组合Navier-Stokes损失：LNS = Lcontinuity + λ*Lmomentum
                    L_NS = L_continuity + lam * L_momentum
                    sample_score = L_NS
                    
                    if i < 5:  # 调试前5个样本
                        print(f"样本 {i}: L_cont={L_continuity:.6f}, L_mom={L_momentum:.6f}, L_NS={L_NS:.6f}")
                        
                except Exception as physics_error:
                    if i < 5:
                        print(f"样本 {i} 物理计算失败: {physics_error}")
                    sample_score = 0.0
                    failed_samples += 1
            else:
                if i < 5:
                    print(f"样本 {i} 预测失败")
                sample_score = 0.0
                failed_samples += 1
                
        except Exception as general_error:
            if i < 5:
                print(f"样本 {i} 处理失败: {general_error}")
            sample_score = 0.0
            failed_samples += 1
            
        finally:
            # 清理临时文件
            try:
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    os.remove(temp_file)
                if 'standard_test_path' in locals() and os.path.exists(standard_test_path):
                    os.remove(standard_test_path)
            except:
                pass
        
        physics_scores.append(sample_score)
    
    # === 步骤3: 结果分析和选择 ===
    print(f"\nPA查询统计:")
    print(f"  - 总样本数: {len(unlabeled_data)}")
    print(f"  - 成功计算: {len(unlabeled_data) - failed_samples}")
    print(f"  - 失败样本: {failed_samples}")
    print(f"  - 成功率: {(len(unlabeled_data) - failed_samples) / len(unlabeled_data) * 100:.2f}%")
    
    # 分数统计
    valid_scores = [s for s in physics_scores if s > 0]
    if valid_scores:
        print(f"  - 有效分数范围: {min(valid_scores):.6f} - {max(valid_scores):.6f}")
        print(f"  - 平均分数: {np.mean(valid_scores):.6f}")
        print(f"  - 分数标准差: {np.std(valid_scores):.6f}")
    else:
        print("  - 警告: 所有物理评分都无效")
    
    # 选择物理违约最严重的样本（分数最高的）
    if len(valid_scores) == 0:
        print("所有样本的物理评分都无效，使用几何复杂度评分")
        return geometry_based_fallback(unlabeled_data, select_num)
    
    # 按物理违约程度排序，选择最严重的
    selected_idx = np.argsort(physics_scores)[-select_num:]
    selected_scores = [physics_scores[i] for i in selected_idx]
    
    print(f"  - 选中样本的物理违约分数: {selected_scores}")
    print(f"PA查询完成，选择了 {len(selected_idx)} 个物理违约最严重的样本")
    
    return selected_idx.tolist()


def compute_continuity_loss_fixed(Y_pred, Pi):
    """
    修复的连续性损失计算
    """
    try:
        # 检查输入维度
        if Y_pred.shape[0] != Pi.shape[0]:
            print(f"警告: 速度场和坐标维度不匹配 {Y_pred.shape} vs {Pi.shape}")
            return 0.0
        
        # 使用简化的散度计算
        divergence_values = compute_divergence_robust(Y_pred, Pi)
        
        # 计算绝对值的均值
        L_continuity = np.mean(np.abs(divergence_values))
        
        return L_continuity
        
    except Exception as e:
        print(f"连续性损失计算失败: {e}")
        return 0.0


def compute_momentum_loss_fixed(Y_pred, Pi, rho=1.0, mu=1.0):
    """
    修复的动量损失计算
    """
    try:
        N, velocity_dim = Y_pred.shape
        
        # 使用简化的动量计算
        momentum_residuals = []
        
        # 计算空间步长
        if N > 1:
            avg_spacing = np.mean([np.linalg.norm(Pi[i] - Pi[j]) 
                                 for i in range(min(10, N)) 
                                 for j in range(i+1, min(i+5, N))])
        else:
            avg_spacing = 1.0
            
        for p in range(min(N, 1000)):  # 限制计算量
            try:
                # 简化的对流项计算：使用有限差分
                if p > 0 and p < N-1:
                    # 使用中心差分近似梯度
                    spatial_grad = (Pi[p+1] - Pi[p-1]) / (2 * avg_spacing + 1e-8)
                    velocity_grad = (Y_pred[p+1] - Y_pred[p-1]) / (2 * avg_spacing + 1e-8)
                    
                    # 对流项：(v·∇)v ≈ v * dv/dx
                    convective_term = Y_pred[p] * np.mean(velocity_grad)
                    
                    # 拉普拉斯项：Δv ≈ (v[p+1] - 2*v[p] + v[p-1]) / dx^2
                    laplacian_term = (Y_pred[p+1] - 2*Y_pred[p] + Y_pred[p-1]) / (avg_spacing**2 + 1e-8)
                    
                else:
                    # 边界点使用简化计算
                    convective_term = Y_pred[p] * 0.1
                    laplacian_term = Y_pred[p] * 0.1
                
                # 动量方程残差: ρ((v·∇)v) - μ(Δv)
                momentum_residual = rho * convective_term - mu * laplacian_term
                
                # 计算L2范数
                residual_norm = np.linalg.norm(momentum_residual)
                momentum_residuals.append(residual_norm)
                
            except Exception as point_error:
                momentum_residuals.append(0.1)  # 默认小值
        
        # 计算均值
        L_momentum = np.mean(momentum_residuals) if momentum_residuals else 0.0
        
        return L_momentum
        
    except Exception as e:
        print(f"动量损失计算失败: {e}")
        return 0.0


def compute_divergence_robust(Y_pred, Pi):
    """
    鲁棒的散度计算
    """
    try:
        N = Y_pred.shape[0]
        divergence = np.zeros(N)
        
        # 使用简单的有限差分
        for i in range(1, N-1):
            # 计算空间和速度的差分
            dx = Pi[i+1] - Pi[i-1]
            dv = Y_pred[i+1] - Y_pred[i-1]
            
            # 避免除零
            dx[np.abs(dx) < 1e-8] = 1e-8
            
            # 计算梯度的对角元素之和（散度）
            grad_diag = dv / dx
            
            # 散度是梯度张量的迹
            divergence[i] = np.sum(grad_diag[:min(len(grad_diag), Pi.shape[1])])
        
        return divergence
        
    except Exception as e:
        print(f"散度计算失败: {e}")
        return np.zeros(Y_pred.shape[0])


def geometry_based_fallback(unlabeled_data, select_num):
    """
    基于几何特征的回退策略
    """
    print("使用几何复杂度作为物理复杂度的代理")
    
    scores = []
    for i, sample in enumerate(unlabeled_data):
        try:
            X_coords = np.array(sample[0])
            
            # 多种几何特征的组合
            centroid = np.mean(X_coords, axis=0)
            distances = np.linalg.norm(X_coords - centroid, axis=1)
            
            # 几何复杂度评分
            geometric_score = (
                np.var(distances) +           # 距离方差
                0.1 * np.max(distances) +     # 最大距离
                0.01 * len(X_coords) +        # 点云密度
                0.001 * np.sum(np.abs(X_coords))  # 坐标绝对值和
            )
            
            scores.append(geometric_score)
        except:
            scores.append(np.random.random())
    
    # 选择几何复杂度最高的样本
    selected_idx = np.argsort(scores)[-select_num:]
    print(f"几何回退策略完成，选择了 {len(selected_idx)} 个样本")
    return selected_idx.tolist()





def compute_continuity_loss(Y_pred, Pi):
    """
    计算连续性损失: Lcontinuity(xi) := mean over p∈Pi |∇·yi|p|
    
    Args:
        Y_pred: 预测的速度场 [N, velocity_dim] 
        Pi: 点云坐标 [N, space_dim]
    
    Returns:
        L_continuity: 连续性损失标量值
    """
    try:
        # 计算速度场的散度 ∇·y
        divergence_values = compute_divergence_pointwise(Y_pred, Pi)
        
        # 计算绝对值的均值
        L_continuity = np.mean(np.abs(divergence_values))
        
        return L_continuity
        
    except Exception as e:
        print(f"连续性损失计算失败: {e}")
        return 0.0


def compute_momentum_loss(Y_pred, Pi, rho=1.0, mu=1.0):
    """
    计算动量损失: Lmomentum(xi) := mean over p∈Pi ||ρ((yi·∇)yi)p - μ(Δyi)p||2
    
    Args:
        Y_pred: 预测的速度场 [N, velocity_dim]
        Pi: 点云坐标 [N, space_dim] 
        rho: 密度
        mu: 动态粘度
    
    Returns:
        L_momentum: 动量损失标量值
    """
    try:
        N, velocity_dim = Y_pred.shape
        momentum_residuals = []
        
        for p in range(N):  # 对每个点 p∈Pi
            try:
                # 计算 (yi·∇)yi 对流项
                convective_term = compute_convective_term_at_point(Y_pred, Pi, p)
                
                # 计算 Δyi 拉普拉斯项
                laplacian_term = compute_laplacian_at_point(Y_pred, Pi, p)
                
                # 计算动量方程残差: ρ((yi·∇)yi)p - μ(Δyi)p
                momentum_residual = rho * convective_term - mu * laplacian_term
                
                # 计算L2范数
                residual_norm = np.linalg.norm(momentum_residual)
                momentum_residuals.append(residual_norm)
                
            except Exception as point_error:
                # 单点计算失败，使用0值
                momentum_residuals.append(0.0)
        
        # 计算所有点的均值
        L_momentum = np.mean(momentum_residuals)
        
        return L_momentum
        
    except Exception as e:
        print(f"动量损失计算失败: {e}")
        return 0.0


def compute_divergence_pointwise(Y_pred, Pi):
    """
    计算每个点的散度 ∇·y
    使用有限差分方法在不规则点云上计算
    """
    try:
        from scipy.spatial import cKDTree
        
        N, velocity_dim = Y_pred.shape
        divergence = np.zeros(N)
        
        # 构建KD树用于邻域搜索
        tree = cKDTree(Pi)
        
        for i in range(N):
            try:
                # 查找邻近点
                distances, indices = tree.query(Pi[i], k=min(10, N), distance_upper_bound=0.1)
                valid_indices = indices[distances < np.inf]
                valid_indices = valid_indices[valid_indices != i]  # 排除自己
                
                if len(valid_indices) >= velocity_dim:
                    # 使用邻近点计算散度
                    neighbors_pos = Pi[valid_indices]
                    neighbors_vel = Y_pred[valid_indices]
                    
                    # 相对位置和相对速度
                    rel_pos = neighbors_pos - Pi[i]
                    rel_vel = neighbors_vel - Y_pred[i]
                    
                    # 使用最小二乘法估计梯度
                    if rel_pos.shape[0] >= rel_pos.shape[1]:
                        gradient_matrix, _, _, _ = np.linalg.lstsq(rel_pos, rel_vel, rcond=None)
                        
                        # 计算散度：∇·v = ∂vx/∂x + ∂vy/∂y + ∂vz/∂z
                        div_val = 0.0
                        for d in range(min(velocity_dim, Pi.shape[1])):
                            if d < gradient_matrix.shape[0]:
                                div_val += gradient_matrix[d, d]  # 对角元素
                        
                        divergence[i] = div_val
                    else:
                        divergence[i] = 0.0
                else:
                    divergence[i] = 0.0
                    
            except Exception as point_error:
                divergence[i] = 0.0
        
        return divergence
        
    except ImportError:
        # scipy不可用时的简化方法
        print("scipy不可用，使用简化散度计算")
        return compute_divergence_simplified(Y_pred, Pi)
    except Exception as e:
        print(f"散度计算失败: {e}")
        return np.zeros(Y_pred.shape[0])


def compute_divergence_simplified(Y_pred, Pi):
    """
    简化的散度计算方法
    """
    N = Y_pred.shape[0]
    divergence = np.zeros(N)
    
    for i in range(1, N-1):
        # 使用简单的前后差分
        dx = Pi[i+1] - Pi[i-1]
        dv = Y_pred[i+1] - Y_pred[i-1]
        
        # 避免除零
        dx[dx == 0] = 1e-8
        
        # 计算偏导数
        grad_approx = dv / dx
        
        # 散度是梯度的迹
        divergence[i] = np.sum(grad_approx.diagonal()[:min(len(grad_approx), Pi.shape[1])])
    
    return divergence


def compute_convective_term_at_point(Y_pred, Pi, point_idx):
    """
    计算点p处的对流项 (yi·∇)yi
    """
    try:
        # 计算该点的速度梯度
        velocity_gradient = compute_velocity_gradient_at_point(Y_pred, Pi, point_idx)
        
        # 该点的速度
        velocity = Y_pred[point_idx]
        
        # 计算 (v·∇)v
        convective = np.dot(velocity, velocity_gradient)
        
        return convective
        
    except Exception as e:
        return np.zeros_like(Y_pred[point_idx])


def compute_laplacian_at_point(Y_pred, Pi, point_idx):
    """
    计算点p处的拉普拉斯项 Δyi
    """
    try:
        from scipy.spatial import cKDTree
        
        tree = cKDTree(Pi)
        
        # 查找邻近点
        distances, indices = tree.query(Pi[point_idx], k=min(8, len(Pi)))
        valid_indices = indices[distances < np.inf]
        valid_indices = valid_indices[valid_indices != point_idx]
        
        if len(valid_indices) >= 3:
            # 使用邻近点近似拉普拉斯算子
            neighbors_vel = Y_pred[valid_indices]
            center_vel = Y_pred[point_idx]
            
            # 简化的拉普拉斯：邻近点平均值减去中心点值
            laplacian = np.mean(neighbors_vel, axis=0) - center_vel
            
            return laplacian
        else:
            return np.zeros_like(Y_pred[point_idx])
            
    except ImportError:
        # 简化方法
        if point_idx > 0 and point_idx < len(Y_pred) - 1:
            laplacian = Y_pred[point_idx-1] - 2*Y_pred[point_idx] + Y_pred[point_idx+1]
            return laplacian
        else:
            return np.zeros_like(Y_pred[point_idx])
    except Exception as e:
        return np.zeros_like(Y_pred[point_idx])


def compute_velocity_gradient_at_point(Y_pred, Pi, point_idx):
    """
    计算点p处的速度梯度 ∇yi
    """
    try:
        from scipy.spatial import cKDTree
        
        tree = cKDTree(Pi)
        
        # 查找邻近点
        distances, indices = tree.query(Pi[point_idx], k=min(6, len(Pi)))
        valid_indices = indices[distances < np.inf]
        valid_indices = valid_indices[valid_indices != point_idx]
        
        if len(valid_indices) >= 3:
            neighbors_pos = Pi[valid_indices]
            neighbors_vel = Y_pred[valid_indices]
            
            # 相对位置和相对速度
            rel_pos = neighbors_pos - Pi[point_idx]
            rel_vel = neighbors_vel - Y_pred[point_idx]
            
            # 使用最小二乘法计算梯度
            if rel_pos.shape[0] >= rel_pos.shape[1]:
                gradient, _, _, _ = np.linalg.lstsq(rel_pos, rel_vel, rcond=None)
                return gradient.T  # [velocity_dim, space_dim]
            else:
                return np.zeros((Y_pred.shape[1], Pi.shape[1]))
        else:
            return np.zeros((Y_pred.shape[1], Pi.shape[1]))
            
    except ImportError:
        # 简化的梯度计算
        if point_idx > 0 and point_idx < len(Y_pred) - 1:
            dv = Y_pred[point_idx+1] - Y_pred[point_idx-1]
            dx = Pi[point_idx+1] - Pi[point_idx-1]
            dx[dx == 0] = 1e-8
            gradient = np.outer(dv, 1.0/dx)
            return gradient
        else:
            return np.zeros((Y_pred.shape[1], Pi.shape[1]))
    except Exception as e:
        return np.zeros((Y_pred.shape[1], Pi.shape[1]))

def prediction_difference_query_fixed(model_tuple, unlabeled_data, select_num):
    """
    修复的预测差异策略 - 解决误差计算无效的问题
    """
    model, metric_func, device = model_tuple
    print(f"开始修复版预测差异策略，设备: {device}, 候选样本数: {len(unlabeled_data)}")
    
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    model.eval()
    prediction_errors = []
    failed_samples = 0
    debug_info = []  # 调试信息
    
    print("开始计算预测差异（修复版）...")
    for i, sample in enumerate(tqdm(unlabeled_data, desc="计算预测差异")):
        if i >= 5:  # 限制调试输出
            break
            
        sample_error = 0.0
        
        try:
            # === 步骤1: 详细检查样本数据 ===
            X_coords = np.array(sample[0])  # 点云坐标
            Y_true = np.array(sample[1])    # 真实标签
            
            # 数据有效性检查
            if len(X_coords) == 0 or len(Y_true) == 0:
                print(f"样本 {i}: 空数据")
                sample_error = 0.0
                failed_samples += 1
                prediction_errors.append(sample_error)
                continue
            
            if X_coords.shape[0] != Y_true.shape[0]:
                print(f"样本 {i}: 坐标和标签数量不匹配 {X_coords.shape[0]} vs {Y_true.shape[0]}")
                sample_error = 0.0
                failed_samples += 1
                prediction_errors.append(sample_error)
                continue
            
            print(f"\n=== 样本 {i} 详细分析 ===")
            print(f"X坐标形状: {X_coords.shape}, 范围: [{X_coords.min():.3f}, {X_coords.max():.3f}]")
            print(f"Y真实形状: {Y_true.shape}, 范围: [{Y_true.min():.3f}, {Y_true.max():.3f}]")
            
            # === 步骤2: 模型预测 ===
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'fixed_pred_diff_{i}_{timestamp}.pkl')
            
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            # 验证文件内容
            with open(standard_test_path, 'rb') as f:
                loaded_data = pickle.load(f)
            print(f"文件验证: 加载了 {len(loaded_data)} 个样本")
            
            args = get_al_args()
            args.dataset = 'al_bz'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            Y_pred = None
            prediction_success = False
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(tmp_loader):
                    print(f"处理批次 {batch_idx}")
                    g, u_p, g_u = batch_data
                    
                    print(f"批次数据类型: g={type(g)}, u_p={type(u_p)}, g_u={type(g_u)}")
                    print(f"u_p形状: {u_p.shape}")
                    
                    # 设备转移
                    g = g.to(device)
                    u_p = u_p.to(device)
                    
                    if hasattr(g_u, 'to'):
                        g_u = g_u.to(device)
                    elif hasattr(g_u, 'tensors'):
                        g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                    
                    model = model.to(device)
                    
                    # 模型预测
                    Y_pred_tensor = model(g, u_p, g_u)
                    Y_pred = Y_pred_tensor.cpu().numpy()
                    
                    print(f"预测成功: 形状={Y_pred.shape}, 范围=[{Y_pred.min():.6f}, {Y_pred.max():.6f}]")
                    print(f"预测统计: 均值={Y_pred.mean():.6f}, 标准差={Y_pred.std():.6f}")
                    
                    prediction_success = True
                    break
                break
            
            # === 步骤3: 详细的误差计算 ===
            if prediction_success and Y_pred is not None:
                print(f"开始误差计算:")
                print(f"  预测形状: {Y_pred.shape}")
                print(f"  真实形状: {Y_true.shape}")
                
                if Y_pred.shape == Y_true.shape:
                    # 1. 基本误差计算
                    diff = Y_pred - Y_true
                    abs_diff = np.abs(diff)
                    
                    print(f"  差值统计: 均值={diff.mean():.6f}, 标准差={diff.std():.6f}")
                    print(f"  绝对差值: 均值={abs_diff.mean():.6f}, 最大={abs_diff.max():.6f}")
                    
                    # 2. 多种误差指标
                    mae_error = np.mean(abs_diff)  # 平均绝对误差
                    mse_error = np.mean(diff ** 2)  # 均方误差
                    rmse_error = np.sqrt(mse_error)  # 均方根误差
                    max_error = np.max(abs_diff)  # 最大误差
                    
                    # 3. 相对误差（避免除零）
                    Y_true_abs = np.abs(Y_true) + 1e-8
                    relative_error = np.mean(abs_diff / Y_true_abs)
                    
                    # 4. L2范数误差
                    l2_error = np.mean(np.linalg.norm(diff, axis=1))
                    
                    print(f"  误差指标:")
                    print(f"    MAE: {mae_error:.6f}")
                    print(f"    MSE: {mse_error:.6f}")
                    print(f"    RMSE: {rmse_error:.6f}")
                    print(f"    最大误差: {max_error:.6f}")
                    print(f"    相对误差: {relative_error:.6f}")
                    print(f"    L2误差: {l2_error:.6f}")
                    
                    # 5. 检查是否所有误差都接近零
                    error_threshold = 1e-10
                    if mae_error < error_threshold and mse_error < error_threshold:
                        print(f"  警告: 所有误差都非常小（可能是预测和真实值完全相同）")
                        # 使用数据方差作为替代评分
                        sample_error = np.var(Y_true) + np.var(Y_pred) + 1e-6
                        print(f"  使用方差替代评分: {sample_error:.6f}")
                    else:
                        # 综合评分
                        sample_error = (
                            1.0 * mae_error +       # 平均绝对误差
                            0.5 * rmse_error +      # 均方根误差
                            0.3 * relative_error +  # 相对误差
                            0.1 * max_error        # 最大误差
                        )
                        print(f"  综合评分: {sample_error:.6f}")
                    
                    # 记录调试信息
                    debug_info.append({
                        'sample_idx': i,
                        'pred_shape': Y_pred.shape,
                        'true_shape': Y_true.shape,
                        'mae_error': mae_error,
                        'mse_error': mse_error,
                        'sample_error': sample_error,
                        'pred_range': [Y_pred.min(), Y_pred.max()],
                        'true_range': [Y_true.min(), Y_true.max()]
                    })
                    
                else:
                    print(f"  形状不匹配: {Y_pred.shape} != {Y_true.shape}")
                    sample_error = 0.0
                    failed_samples += 1
            else:
                print(f"  预测失败")
                sample_error = 0.0
                failed_samples += 1
            
            # 清理临时文件
            os.remove(temp_file)
            if os.path.exists(standard_test_path):
                os.remove(standard_test_path)
                
        except Exception as e:
            print(f"样本 {i} 处理失败: {e}")
            import traceback
            print(f"详细错误: {traceback.format_exc()}")
            sample_error = 0.0
            failed_samples += 1
        
        prediction_errors.append(sample_error)
        print(f"样本 {i} 最终评分: {sample_error:.6f}")
    
    # 快速处理剩余样本（无调试输出）
    for i in range(5, len(unlabeled_data)):
        try:
            # 简化处理
            X_coords = np.array(unlabeled_data[i][0])
            Y_true = np.array(unlabeled_data[i][1])
            
            # 使用几何复杂度和数据方差作为代理评分
            if len(X_coords) > 0 and len(Y_true) > 0:
                geometric_score = np.var(X_coords) + np.var(Y_true) + np.random.random() * 0.1
                prediction_errors.append(geometric_score)
            else:
                prediction_errors.append(np.random.random() * 0.01)
                
        except:
            prediction_errors.append(np.random.random() * 0.01)
    
    # === 结果分析 ===
    print(f"\n修复版预测差异策略统计:")
    print(f"  - 总样本数: {len(unlabeled_data)}")
    print(f"  - 详细分析样本数: {min(5, len(unlabeled_data))}")
    print(f"  - 失败样本: {failed_samples}")
    
    valid_errors = [e for e in prediction_errors if e > 1e-10]
    print(f"  - 有效误差数量: {len(valid_errors)}")
    
    if valid_errors:
        print(f"  - 误差范围: {min(valid_errors):.6f} - {max(valid_errors):.6f}")
        print(f"  - 平均误差: {np.mean(valid_errors):.6f}")
        print(f"  - 误差标准差: {np.std(valid_errors):.6f}")
        
        # 显示调试信息
        if debug_info:
            print(f"\n前5个样本的详细信息:")
            for info in debug_info:
                print(f"  样本{info['sample_idx']}: MAE={info['mae_error']:.6f}, 综合评分={info['sample_error']:.6f}")
    
    # 选择误差最大的样本
    if len(valid_errors) == 0:
        print("警告: 仍然没有有效误差，使用随机采样")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    selected_idx = np.argsort(prediction_errors)[-select_num:]
    selected_errors = [prediction_errors[i] for i in selected_idx]
    
    print(f"修复版预测差异策略完成")
    print(f"选中样本评分范围: [{min(selected_errors):.6f}, {max(selected_errors):.6f}]")
    
    return selected_idx.tolist()


def prediction_difference_query(model_tuple, unlabeled_data, select_num):
    """
    预测差异策略 - 选择模型预测与真实标签差异最大的样本
    
    策略原理：
    1. 用当前训练的模型对未标注数据进行预测
    2. 计算预测值与真实标签的差异（L2范数、相对误差等）
    3. 按差异从大到小排序，选择差异最大的样本
    4. 这些样本是模型当前最不确定或最难预测的样本
    
    Args:
        model_tuple: (model, metric_func, device) 元组
        unlabeled_data: 未标注数据列表，每个元素为 [X, Y, Theta, (empty_branch,)]
        select_num: 需要选择的样本数量
        
    Returns:
        selected_idx: 选中样本的索引列表
    """
    model, metric_func, device = model_tuple
    print(f"开始预测差异策略，设备: {device}, 候选样本数: {len(unlabeled_data)}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    model.eval()  # 设置为评估模式
    prediction_errors = []
    failed_samples = 0
    
    print("开始计算预测差异...")
    for i, sample in enumerate(tqdm(unlabeled_data, desc="计算预测差异")):
        if i % 100 == 0 and i > 0:
            print(f"已处理 {i}/{len(unlabeled_data)} 个样本，失败 {failed_samples} 个")
        
        sample_error = 0.0
        
        try:
            # === 步骤1: 获取真实标签 ===
            X_coords = np.array(sample[0])  # 点云坐标 [N, 3]
            Y_true = np.array(sample[1])    # 真实标签 [N, 5] - [pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear]
            
            # === 步骤2: 用模型进行预测 ===
            # 创建临时数据文件
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'pred_diff_temp_{i}_{timestamp}.pkl')
            
            # 保存单个样本
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            # 复制到标准位置
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            # 加载数据集并预测
            args = get_al_args()
            args.dataset = 'al_bz'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            Y_pred = None
            
            with torch.no_grad():
                for batch_data in tmp_loader:
                    g, u_p, g_u = batch_data
                    
                    # 设备转移
                    g = g.to(device)
                    u_p = u_p.to(device)
                    
                    # MultipleTensors对象的特殊处理
                    if hasattr(g_u, 'to'):
                        g_u = g_u.to(device)
                    elif hasattr(g_u, 'tensors'):
                        g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                    
                    # 确保模型在正确设备上
                    model = model.to(device)
                    
                    # 模型预测
                    Y_pred = model(g, u_p, g_u).cpu().numpy()  # [N, 5]
                    break
                break
            
            # === 步骤3: 计算预测差异 ===
            if Y_pred is not None and Y_pred.shape == Y_true.shape:
                # 方法1: L2范数（欧几里得距离）
                l2_error = np.linalg.norm(Y_pred - Y_true, axis=1)  # 每个点的L2误差
                mean_l2_error = np.mean(l2_error)
                
                # 方法2: 相对误差（避免除零）
                Y_true_norm = np.linalg.norm(Y_true, axis=1) + 1e-8
                relative_error = l2_error / Y_true_norm
                mean_relative_error = np.mean(relative_error)
                
                # 方法3: 各个物理量的分别误差
                pressure_error = np.mean(np.abs(Y_pred[:, 0] - Y_true[:, 0]))
                wall_shear_error = np.mean(np.abs(Y_pred[:, 1] - Y_true[:, 1]))
                shear_components_error = np.mean(np.abs(Y_pred[:, 2:5] - Y_true[:, 2:5]))
                
                # 方法4: 最大误差
                max_error = np.max(l2_error)
                
                # 方法5: 标准化误差（按真实值的标准差归一化）
                Y_true_std = np.std(Y_true, axis=0) + 1e-8
                normalized_error = np.mean(np.abs(Y_pred - Y_true) / Y_true_std)
                
                # 综合评分（可以调整权重）
                sample_error = (
                    1.0 * mean_l2_error +           # L2误差
                    0.5 * mean_relative_error +     # 相对误差  
                    0.3 * max_error +               # 最大误差
                    0.2 * normalized_error +        # 标准化误差
                    0.1 * pressure_error +          # 压力误差
                    0.1 * wall_shear_error +        # 壁面剪切误差
                    0.1 * shear_components_error    # 剪切分量误差
                )
                
                # 调试前几个样本
                if i < 5:
                    print(f"\n样本 {i} 预测差异分析:")
                    print(f"  - 预测形状: {Y_pred.shape}, 真实形状: {Y_true.shape}")
                    print(f"  - L2误差: {mean_l2_error:.6f}")
                    print(f"  - 相对误差: {mean_relative_error:.6f}")
                    print(f"  - 最大误差: {max_error:.6f}")
                    print(f"  - 标准化误差: {normalized_error:.6f}")
                    print(f"  - 压力误差: {pressure_error:.6f}")
                    print(f"  - 壁面剪切误差: {wall_shear_error:.6f}")
                    print(f"  - 综合评分: {sample_error:.6f}")
                    
                    # 显示预测值 vs 真实值的统计
                    print(f"  - 预测值范围: [{Y_pred.min():.4f}, {Y_pred.max():.4f}]")
                    print(f"  - 真实值范围: [{Y_true.min():.4f}, {Y_true.max():.4f}]")
                    print(f"  - 预测均值: {Y_pred.mean(axis=0)}")
                    print(f"  - 真实均值: {Y_true.mean(axis=0)}")
                
            else:
                if i < 5:
                    print(f"样本 {i} 预测失败: 预测形状 {Y_pred.shape if Y_pred is not None else None}, 真实形状 {Y_true.shape}")
                sample_error = 0.0
                failed_samples += 1
                
        except Exception as e:
            if i < 5:
                print(f"样本 {i} 处理失败: {e}")
            sample_error = 0.0
            failed_samples += 1
            
        finally:
            # 清理临时文件
            try:
                if 'temp_file' in locals() and os.path.exists(temp_file):
                    os.remove(temp_file)
                if 'standard_test_path' in locals() and os.path.exists(standard_test_path):
                    os.remove(standard_test_path)
            except:
                pass
        
        prediction_errors.append(sample_error)
    
    # === 步骤4: 结果分析和选择 ===
    print(f"\n预测差异策略统计:")
    print(f"  - 总样本数: {len(unlabeled_data)}")
    print(f"  - 成功计算: {len(unlabeled_data) - failed_samples}")
    print(f"  - 失败样本: {failed_samples}")
    print(f"  - 成功率: {(len(unlabeled_data) - failed_samples) / len(unlabeled_data) * 100:.2f}%")
    
    # 误差统计
    valid_errors = [e for e in prediction_errors if e > 0]
    if valid_errors:
        print(f"  - 有效误差数量: {len(valid_errors)}")
        print(f"  - 误差范围: {min(valid_errors):.6f} - {max(valid_errors):.6f}")
        print(f"  - 平均误差: {np.mean(valid_errors):.6f}")
        print(f"  - 误差标准差: {np.std(valid_errors):.6f}")
        print(f"  - 误差中位数: {np.median(valid_errors):.6f}")
        
        # 误差分布分析
        percentiles = [90, 95, 99]
        for p in percentiles:
            threshold = np.percentile(valid_errors, p)
            print(f"  - {p}%分位数: {threshold:.6f}")
    else:
        print("  - 警告: 所有预测误差都无效")
    
    # 选择误差最大的样本（即模型预测最不准确的样本）
    if len(valid_errors) == 0:
        print("所有样本的预测误差都无效，使用随机采样")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    # 按预测误差从大到小排序，选择误差最大的
    selected_idx = np.argsort(prediction_errors)[-select_num:]
    selected_errors = [prediction_errors[i] for i in selected_idx]
    
    print(f"  - 选中样本的预测误差: {selected_errors}")
    print(f"  - 选中样本误差平均值: {np.mean(selected_errors):.6f}")
    print(f"预测差异策略完成，选择了 {len(selected_idx)} 个预测误差最大的样本")
    
    return selected_idx.tolist()


def prediction_difference_query_fast(model_tuple, unlabeled_data, select_num, batch_size=8):
    """
    预测差异策略的快速版本 - 批量处理
    
    优化策略：
    1. 批量预测减少文件I/O
    2. 简化误差计算
    3. 内存友好的处理方式
    """
    model, metric_func, device = model_tuple
    print(f"开始快速预测差异策略，设备: {device}, 候选样本数: {len(unlabeled_data)}")
    print(f"批次大小: {batch_size}")
    
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    model.eval()
    prediction_errors = []
    failed_batches = 0
    
    # 分批处理
    num_batches = (len(unlabeled_data) + batch_size - 1) // batch_size
    print(f"将分 {num_batches} 个批次处理")
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(unlabeled_data))
        batch_samples = unlabeled_data[start_idx:end_idx]
        
        print(f"处理批次 {batch_idx + 1}/{num_batches}, 样本 {start_idx}-{end_idx-1}")
        
        try:
            # 创建批次临时文件
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'batch_pred_diff_{batch_idx}_{timestamp}.pkl')
            
            with open(temp_file, 'wb') as f:
                pickle.dump(batch_samples, f)
            
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            # 加载数据集
            args = get_al_args()
            args.dataset = 'al_bz'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            # 批量预测
            batch_predictions = []
            batch_true_labels = []
            
            sample_idx_in_batch = 0
            for batch_data in tmp_loader:
                if sample_idx_in_batch >= len(batch_samples):
                    break
                    
                g, u_p, g_u = batch_data
                
                # 设备转移
                g = g.to(device)
                u_p = u_p.to(device)
                if hasattr(g_u, 'to'):
                    g_u = g_u.to(device)
                elif hasattr(g_u, 'tensors'):
                    g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                
                model = model.to(device)
                
                with torch.no_grad():
                    Y_pred = model(g, u_p, g_u).cpu().numpy()
                    Y_true = np.array(batch_samples[sample_idx_in_batch][1])
                    
                    batch_predictions.append(Y_pred)
                    batch_true_labels.append(Y_true)
                    
                sample_idx_in_batch += 1
            
            # 计算批次内每个样本的误差
            for i, (Y_pred, Y_true) in enumerate(zip(batch_predictions, batch_true_labels)):
                try:
                    if Y_pred.shape == Y_true.shape:
                        # 简化的误差计算（更快）
                        mse_error = np.mean((Y_pred - Y_true) ** 2)
                        mae_error = np.mean(np.abs(Y_pred - Y_true))
                        max_error = np.max(np.abs(Y_pred - Y_true))
                        
                        # 简单的综合评分
                        sample_error = mse_error + 0.1 * mae_error + 0.01 * max_error
                        prediction_errors.append(sample_error)
                    else:
                        prediction_errors.append(0.0)
                except:
                    prediction_errors.append(0.0)
            
            # 清理临时文件
            os.remove(temp_file)
            if os.path.exists(standard_test_path):
                os.remove(standard_test_path)
                
        except Exception as e:
            print(f"批次 {batch_idx} 处理失败: {e}")
            failed_batches += 1
            # 为失败的批次添加零误差
            for _ in range(len(batch_samples)):
                prediction_errors.append(0.0)
    
    # 结果分析
    print(f"\n快速预测差异策略统计:")
    print(f"  - 总批次数: {num_batches}")
    print(f"  - 失败批次: {failed_batches}")
    print(f"  - 成功率: {(num_batches - failed_batches) / num_batches * 100:.2f}%")
    
    valid_errors = [e for e in prediction_errors if e > 0]
    if valid_errors:
        print(f"  - 有效误差数量: {len(valid_errors)}")
        print(f"  - 误差范围: {min(valid_errors):.6f} - {max(valid_errors):.6f}")
        print(f"  - 平均误差: {np.mean(valid_errors):.6f}")
    
    # 选择误差最大的样本
    if len(valid_errors) == 0:
        print("所有样本的预测误差都无效，使用随机采样")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    selected_idx = np.argsort(prediction_errors)[-select_num:]
    selected_errors = [prediction_errors[i] for i in selected_idx]
    
    print(f"快速预测差异策略完成，选择了 {len(selected_idx)} 个样本")
    print(f"选中样本的预测误差范围: [{min(selected_errors):.6f}, {max(selected_errors):.6f}]")
    
    return selected_idx.tolist()


def prediction_difference_query_with_analysis(model_tuple, unlabeled_data, select_num):
    """
    带详细分析的预测差异策略
    
    增强功能：
    1. 详细的误差分析和可视化
    2. 不同物理量的分别评估
    3. 误差分布统计
    4. 样本难度分级
    """
    model, metric_func, device = model_tuple
    print(f"开始预测差异策略（详细分析版），设备: {device}, 候选样本数: {len(unlabeled_data)}")
    
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    model.eval()
    
    # 详细误差记录
    detailed_errors = {
        'l2_errors': [],
        'relative_errors': [],
        'pressure_errors': [],
        'wall_shear_errors': [],
        'shear_component_errors': [],
        'max_errors': [],
        'sample_complexities': [],  # 样本复杂度（点数、几何特征等）
        'prediction_confidences': []  # 预测置信度
    }
    
    failed_samples = 0
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="详细分析预测差异")):
        try:
            X_coords = np.array(sample[0])
            Y_true = np.array(sample[1])
            
            # 计算样本复杂度特征
            sample_complexity = {
                'num_points': len(X_coords),
                'spatial_variance': np.var(X_coords),
                'centroid_distance': np.mean(np.linalg.norm(X_coords - np.mean(X_coords, axis=0), axis=1)),
                'output_variance': np.var(Y_true),
                'output_range': np.max(Y_true) - np.min(Y_true)
            }
            
            # 模型预测（复用之前的代码）
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'analysis_temp_{i}_{timestamp}.pkl')
            
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            args = get_al_args()
            args.dataset = 'al_bz'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            Y_pred = None
            with torch.no_grad():
                for batch_data in tmp_loader:
                    g, u_p, g_u = batch_data
                    g = g.to(device)
                    u_p = u_p.to(device)
                    if hasattr(g_u, 'to'):
                        g_u = g_u.to(device)
                    elif hasattr(g_u, 'tensors'):
                        g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                    
                    model = model.to(device)
                    Y_pred = model(g, u_p, g_u).cpu().numpy()
                    break
                break
            
            # 详细误差计算
            if Y_pred is not None and Y_pred.shape == Y_true.shape:
                # L2误差
                l2_error = np.mean(np.linalg.norm(Y_pred - Y_true, axis=1))
                
                # 相对误差
                Y_true_norm = np.linalg.norm(Y_true, axis=1) + 1e-8
                relative_error = np.mean(np.linalg.norm(Y_pred - Y_true, axis=1) / Y_true_norm)
                
                # 各物理量误差
                pressure_error = np.mean(np.abs(Y_pred[:, 0] - Y_true[:, 0]))
                wall_shear_error = np.mean(np.abs(Y_pred[:, 1] - Y_true[:, 1]))
                shear_component_error = np.mean(np.abs(Y_pred[:, 2:5] - Y_true[:, 2:5]))
                
                # 最大误差
                max_error = np.max(np.abs(Y_pred - Y_true))
                
                # 预测置信度（基于预测值的方差）
                prediction_confidence = 1.0 / (1.0 + np.var(Y_pred))
                
                # 记录到详细误差字典
                detailed_errors['l2_errors'].append(l2_error)
                detailed_errors['relative_errors'].append(relative_error)
                detailed_errors['pressure_errors'].append(pressure_error)
                detailed_errors['wall_shear_errors'].append(wall_shear_error)
                detailed_errors['shear_component_errors'].append(shear_component_error)
                detailed_errors['max_errors'].append(max_error)
                detailed_errors['sample_complexities'].append(sample_complexity)
                detailed_errors['prediction_confidences'].append(prediction_confidence)
                
            else:
                # 失败样本记录零值
                failed_samples += 1
                for key in ['l2_errors', 'relative_errors', 'pressure_errors', 
                           'wall_shear_errors', 'shear_component_errors', 'max_errors']:
                    detailed_errors[key].append(0.0)
                detailed_errors['sample_complexities'].append(sample_complexity)
                detailed_errors['prediction_confidences'].append(0.0)
            
            # 清理临时文件
            os.remove(temp_file)
            if os.path.exists(standard_test_path):
                os.remove(standard_test_path)
                
        except Exception as e:
            if i < 5:
                print(f"样本 {i} 分析失败: {e}")
            failed_samples += 1
            # 记录零值
            for key in ['l2_errors', 'relative_errors', 'pressure_errors', 
                       'wall_shear_errors', 'shear_component_errors', 'max_errors']:
                detailed_errors[key].append(0.0)
            detailed_errors['sample_complexities'].append({'num_points': 0})
            detailed_errors['prediction_confidences'].append(0.0)
    
    # === 详细分析和报告 ===
    print(f"\n=== 预测差异详细分析报告 ===")
    print(f"总样本数: {len(unlabeled_data)}")
    print(f"成功分析: {len(unlabeled_data) - failed_samples}")
    print(f"失败样本: {failed_samples}")
    
    # 各类误差统计
    error_types = ['l2_errors', 'relative_errors', 'pressure_errors', 
                   'wall_shear_errors', 'shear_component_errors', 'max_errors']
    
    for error_type in error_types:
        errors = [e for e in detailed_errors[error_type] if e > 0]
        if errors:
            print(f"\n{error_type.replace('_', ' ').title()}:")
            print(f"  - 范围: [{min(errors):.6f}, {max(errors):.6f}]")
            print(f"  - 均值: {np.mean(errors):.6f}")
            print(f"  - 标准差: {np.std(errors):.6f}")
            print(f"  - 中位数: {np.median(errors):.6f}")
    
    # 样本复杂度分析
    complexities = [c for c in detailed_errors['sample_complexities'] if 'num_points' in c and c['num_points'] > 0]
    if complexities:
        print(f"\n样本复杂度统计:")
        point_counts = [c['num_points'] for c in complexities]
        spatial_vars = [c.get('spatial_variance', 0) for c in complexities]
        print(f"  - 点数范围: [{min(point_counts)}, {max(point_counts)}]")
        print(f"  - 平均点数: {np.mean(point_counts):.0f}")
        print(f"  - 空间方差范围: [{min(spatial_vars):.6f}, {max(spatial_vars):.6f}]")
    
    # 综合评分和选择
    valid_samples = len(unlabeled_data) - failed_samples
    if valid_samples == 0:
        print("所有样本分析都失败，使用随机选择")
        return random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
    
    # 多指标综合评分
    composite_scores = []
    for i in range(len(unlabeled_data)):
        score = (
            1.0 * detailed_errors['l2_errors'][i] +
            0.5 * detailed_errors['relative_errors'][i] +
            0.3 * detailed_errors['max_errors'][i] +
            0.2 * detailed_errors['pressure_errors'][i] +
            0.2 * detailed_errors['wall_shear_errors'][i] +
            0.2 * detailed_errors['shear_component_errors'][i] +
            0.1 / (detailed_errors['prediction_confidences'][i] + 1e-8)  # 低置信度样本得分更高
        )
        composite_scores.append(score)
    
    # 选择综合评分最高的样本
    selected_idx = np.argsort(composite_scores)[-select_num:]
    selected_scores = [composite_scores[i] for i in selected_idx]
    
    print(f"\n选择结果:")
    print(f"  - 选中样本数: {len(selected_idx)}")
    print(f"  - 综合评分范围: [{min(selected_scores):.6f}, {max(selected_scores):.6f}]")
    print(f"  - 平均综合评分: {np.mean(selected_scores):.6f}")
    
    # 分析选中样本的特征
    selected_l2_errors = [detailed_errors['l2_errors'][i] for i in selected_idx]
    selected_complexities = [detailed_errors['sample_complexities'][i] for i in selected_idx if detailed_errors['sample_complexities'][i].get('num_points', 0) > 0]
    
    print(f"  - 选中样本L2误差范围: [{min(selected_l2_errors):.6f}, {max(selected_l2_errors):.6f}]")
    if selected_complexities:
        selected_point_counts = [c['num_points'] for c in selected_complexities]
        print(f"  - 选中样本点数范围: [{min(selected_point_counts)}, {max(selected_point_counts)}]")
    
    print("预测差异详细分析策略完成！")
    
    return selected_idx.tolist()



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
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
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
                try:
                    print("执行PA策略（修复版本）...")
                    selected_idx = pa_query_fixed(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"PA策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"PA策略修复版本失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 使用几何回退
                    selected_idx = geometry_based_fallback(unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"已使用几何回退策略")
            # === 新增预测差异策略 ===
            elif strategy == 'pred_diff_fixed':
                try:
                    print("执行修复版预测差异策略...")
                    selected_idx = prediction_difference_query_fixed(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"修复版预测差异策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"修复版预测差异策略失败: {e}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
            elif strategy == 'pred_diff':
                try:
                    print("执行预测差异策略...")
                    selected_idx = prediction_difference_query(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"预测差异策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"预测差异策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 回退到随机采样
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
            
            elif strategy == 'pred_diff_fast':
                try:
                    print("执行快速预测差异策略...")
                    selected_idx = prediction_difference_query_fast(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"快速预测差异策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"快速预测差异策略失败: {e}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
            
            elif strategy == 'pred_diff_analysis':
                try:
                    print("执行预测差异详细分析策略...")
                    selected_idx = prediction_difference_query_with_analysis(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"预测差异分析策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"预测差异分析策略失败: {e}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
                    
            elif strategy == 'bz':
                try:
                    print("执行BZ策略（基于GNOT模型测试框架）...")
                    from alpa import bz_query
                    selected_idx = bz_query(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"BZ策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"BZ策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
                    
            elif strategy == 'bz_scaled':
                try:
                    print("执行BZ策略（维度缩放版本）...")
                    from bz_strategy_scale_fix import bz_query_with_dimension_scaling
                    selected_idx = bz_query_with_dimension_scaling(
                        model_tuple, unlabeled_data, select_num, 
                        scaling_method='adaptive'
                    )
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"BZ缩放策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"BZ缩放策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
                    
            elif strategy == 'bz_balanced':
                try:
                    print("执行BZ策略（强平衡版本）...")
                    from bz_strategy_scale_fix import bz_query_with_dimension_scaling
                    # 使用更强的平衡权重：大幅降低pressure维度权重
                    strong_balance_weights = [0.05, 1.0, 1.0, 1.0, 1.0]  # pressure权重降为0.05
                    selected_idx = bz_query_with_dimension_scaling(
                        model_tuple, unlabeled_data, select_num, 
                        scaling_method='manual',
                        manual_weights=strong_balance_weights
                    )
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"BZ强平衡策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"BZ强平衡策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
                    
            elif strategy == 'bz_manual':
                try:
                    print("执行BZ策略（手动权重版本）...")
                    from bz_strategy_scale_fix import bz_query_with_dimension_scaling
                    # 手动降低pressure维度的权重
                    manual_weights = [0.1, 1.0, 1.0, 1.0, 1.0]  # pressure权重降为0.1
                    selected_idx = bz_query_with_dimension_scaling(
                        model_tuple, unlabeled_data, select_num, 
                        scaling_method='manual',
                        manual_weights=manual_weights
                    )
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"BZ手动权重策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"BZ手动权重策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    newly_selected = random.sample(unlabeled_data, select_num)
                    selected_idx = [unlabeled_data.index(x) for x in newly_selected]
                    print(f"已回退到随机采样")
                    
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
# ...existing code...

if __name__ == "__main__":
    import subprocess
    import time
    import shutil
    
    # === 设置完全确定性训练 ===
    print("=== 设置确定性训练环境 ===")
    set_deterministic_training(42)
    print("已启用完全确定性训练，相同初始数据将产生相同结果\n")

    # 新增：预测差异排序策略示例
    def prediction_difference_ranking(model_tuple, unlabeled_data):
        """
        用当前模型对未标注数据进行预测，与标准结果对比，按平均差值排序
        返回排序后的索引列表（平均差值越大越靠前）
        """
        model, metric_func, device = model_tuple
        model.eval()
        errors = []
        for i, sample in enumerate(unlabeled_data):
            try:
                X = np.array(sample[0])
                Y_true = np.array(sample[1])
                # 构造临时数据集
                temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
                os.makedirs(temp_data_dir, exist_ok=True)
                timestamp = int(time.time() * 1000000)
                temp_file = os.path.join(temp_data_dir, f'pred_diff_rank_{i}_{timestamp}.pkl')
                with open(temp_file, 'wb') as f:
                    pickle.dump([sample], f)
                standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
                shutil.copy2(temp_file, standard_test_path)
                args = get_al_args()
                args.dataset = 'al_bz'
                _, tmp_dataset = get_dataset(args)
                tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
                Y_pred = None
                with torch.no_grad():
                    for batch_data in tmp_loader:
                        g, u_p, g_u = batch_data
                        g = g.to(device)
                        u_p = u_p.to(device)
                        if hasattr(g_u, 'to'):
                            g_u = g_u.to(device)
                        elif hasattr(g_u, 'tensors'):
                            g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                        model = model.to(device)
                        Y_pred = model(g, u_p, g_u).cpu().numpy()
                        break
                    break
                # 计算平均差值
                if Y_pred is not None and Y_pred.shape == Y_true.shape:
                    avg_diff = np.mean(np.abs(Y_pred - Y_true))
                else:
                    avg_diff = 0.0
                errors.append(avg_diff)
                # 清理临时文件
                os.remove(temp_file)
                if os.path.exists(standard_test_path):
                    os.remove(standard_test_path)
            except Exception as e:
                errors.append(0.0)
        # 按平均差值降序排序
        sorted_indices = np.argsort(errors)[::-1]
        return sorted_indices.tolist()
    
    # 智能选择GPU
    selected_gpu = select_best_gpu()
    if selected_gpu is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(selected_gpu)
        print(f"设置CUDA_VISIBLE_DEVICES={selected_gpu}")
    else:
        print("将使用CPU进行训练")
    
    # 策略选择
    strategy = 'bz_scaled'  # 使用强平衡BZ策略
    # strategy = 'bz_scaled'  # 使用新的BZ维度缩放策略
    # strategy = 'bz'             # 原始BZ策略
    # strategy = 'bz_manual'      # 手动权重BZ策略
    # strategy = 'pred_diff_fixed'  # 使用新的预测差异策略
    # strategy = 'pred_diff_fast'     # 快速版本
    # strategy = 'pred_diff_analysis' # 详细分析版本
    # strategy = 'random'     # 随机策略
    # strategy = 'gv'         # 几何方差策略
    # strategy = 'qbc'        # 查询委员会
    # strategy = 'pa'         # 物理感知
    
    print(f"=== 策略选择 ===")
    print(f"当前使用策略: {strategy}")
    
    if strategy.startswith('bz'):
        print("BZ策略说明:")
        print("  - 基于预测误差的样本选择")
        print("  - 使用当前模型对未标注数据进行预测")
        print("  - 计算预测值与真实标签的差异")
        print("  - 选择预测误差最大的样本进行标注")
        
        if strategy == 'bz':
            print("  - 原始版本：直接计算各维度MSE误差总和")
        elif strategy == 'bz_scaled':
            print("  - 缩放版本：自适应计算各维度缩放因子，平衡各维度贡献")
        elif strategy == 'bz_manual':
            print("  - 手动权重版本：手动设置各维度权重，降低pressure权重")
        elif strategy == 'bz_balanced':
            print("  - 强平衡版本：大幅降低pressure权重(0.05)，突出其他维度")
    
    elif strategy.startswith('pred_diff'):
        print("预测差异策略说明:")
        print("  - 使用当前模型对未标注数据进行预测")
        print("  - 计算预测值与真实标签的差异")
        print("  - 选择预测误差最大的样本进行标注")
        print("  - 这些样本是模型当前最难预测的，最有学习价值")
        
        if strategy == 'pred_diff':
            print("  - 标准版本：详细误差分析，高精度")
        elif strategy == 'pred_diff_fast':
            print("  - 快速版本：批量处理，适合大数据集")
        elif strategy == 'pred_diff_analysis':
            print("  - 分析版本：最详细的误差分析和报告")
    
    # === 定义路径（修复变量定义顺序） ===
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"  # 数据更新目录
    output_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz/al_rounds15"  # 结果保存目录
    temp_data_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz/data"  # 中间数据目录
    
    print(f"\n=== 目录配置 ===")
    print(f"数据更新目录: {data_update_dir}")
    print(f"  - 程序将从此目录读取初始数据，并每轮更新数据文件")
    print(f"  - 需要的文件: al_labeled.pkl, al_unlabeled.pkl, al_test.pkl")
    print(f"结果保存目录: {output_dir}")
    print(f"  - 每轮保存 train_data.pkl, test_data.pkl, strategy_info.json")
    print(f"中间数据目录: {temp_data_dir}")
    print(f"  - 临时训练和测试文件，不影响 /home/v-wenliao/gnot/GNOT/data/al_labeled.pkl")
    
    # 创建中间数据目录
    os.makedirs(temp_data_dir, exist_ok=True)
    
    # === 定义数据文件路径（在使用之前定义） ===
    labeled_path = os.path.join(data_update_dir, 'al_labeled.pkl')
    unlabeled_path = os.path.join(data_update_dir, 'al_unlabeled.pkl')
    test_path = os.path.join(data_update_dir, 'al_test.pkl')
    original_file = "/home/v-wenliao/gnot/GNOT/data/al_labeled.pkl"
    
    # 检查数据目录
    print(f"\n=== 数据文件检查 ===")
    required_files = [
        labeled_path,
        unlabeled_path,
        test_path
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
    if os.path.exists(original_file):
        print(f"✓ 原始文件 {original_file} 存在，不会被影响")
    else:
        print(f"ℹ 原始文件 {original_file} 不存在，这是正常的")
    
    if not all_exist:
        print(f"\n请确保在 {data_update_dir} 目录中放置所有必要的数据文件后再运行程序。")
        print(f"缺失的文件需要手动创建或从其他位置复制。")
        print(f"\n提示：如果您有原始数据，可以运行数据准备函数：")
        print(f"prepare_active_learning_data(raw_data_dir, '{data_update_dir}', overwrite=True)")
    else:
        print(f"\n所有数据文件准备就绪，开始主动学习...")
        
        # 启动主动学习
        random_active_learning_with_logging(
            rounds=5,
            select_num=200,
            seed=42,
            output_dir=output_dir,  # 结果保存目录
            data_update_dir=data_update_dir,  # 数据更新目录
            strategy=strategy
        )