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
try:
    from train import train, validate_epoch
except ImportError as e:
    print(f"Warning: Could not import train functions: {e}")
    train = None
    validate_epoch = None
from args import get_args as get_original_args
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from utils import get_seed, MultipleTensors
from torch.nn.utils.rnn import pad_sequence


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
        epochs = 500
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



def pa_query_fixed(model_tuple, unlabeled_data, select_num, error_weights=None):
    """
    基于预测误差的PA查询策略
    
    Args:
        model_tuple: (model, metric_func, device)
        unlabeled_data: 未标注数据列表，每个元素为 [X_coords, Y_true, Theta, branch_data]
        select_num: 需要选择的样本数量
        error_weights: 误差权重向量，默认为[1, 1, 1, 1, 1]表示五个输出等权重
    
    Returns:
        selected_idx: 选择的样本索引列表，按预测误差从大到小排序
    """
    if error_weights is None:
        error_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])  # pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear
    
    model, metric_func, device = model_tuple
    print(f"PA查询开始（基于预测误差），设备: {device}, 样本数: {len(unlabeled_data)}")
    print(f"误差权重: {error_weights}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"中间数据目录: {temp_data_dir}")
    
    model.eval()
    prediction_errors = []
    failed_samples = 0
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="PA预测误差评估")):
        if i % 50 == 0:
            print(f"PA进度: {i}/{len(unlabeled_data)}")
        
        sample_error = 0.0
        
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
            
            Y_pred = None  # 预测值
            Y_true = None  # 真实值
            
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
                    Y_pred = model(g, u_p, g_u).cpu().numpy()  # [N, 5] - 预测的五个值
                    
                    # 获取真实标签值
                    Y_true = np.array(sample[1])  # [N, 5] - 真实的pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear
                    
                    break
                break
            
            # === 步骤2: 计算预测误差 ===
            if Y_pred is not None and Y_true is not None:
                try:
                    if i < 5:  # 调试前5个样本的详细信息
                        print(f"样本 {i} 调试信息:")
                        print(f"  Y_pred形状: {Y_pred.shape}, 类型: {type(Y_pred)}")
                        print(f"  Y_true形状: {Y_true.shape}, 类型: {type(Y_true)}")
                        print(f"  Y_pred前几个值: {Y_pred[:3] if len(Y_pred) > 0 else 'empty'}")
                        print(f"  Y_true前几个值: {Y_true[:3] if len(Y_true) > 0 else 'empty'}")
                    
                    # 检查维度匹配
                    if Y_pred.shape != Y_true.shape:
                        print(f"警告: 样本 {i} 预测和真实值维度不匹配 {Y_pred.shape} vs {Y_true.shape}")
                        sample_error = 0.0
                        failed_samples += 1
                    else:
                        # 计算每个输出的误差
                        prediction_errors_per_field = []
                        
                        for field_idx in range(min(5, Y_pred.shape[1])):  # 最多5个字段
                            # 计算该字段的平均绝对误差
                            field_pred = Y_pred[:, field_idx]
                            field_true = Y_true[:, field_idx]
                            
                            # 计算平均绝对误差 (MAE)
                            mae = np.mean(np.abs(field_pred - field_true))
                            
                            # 计算相对误差 (避免除零)
                            true_scale = np.mean(np.abs(field_true)) + 1e-8
                            relative_error = mae / true_scale
                            
                            prediction_errors_per_field.append(relative_error)
                            
                            if i < 3:  # 调试前3个样本的每个字段
                                print(f"    字段 {field_idx}: MAE={mae:.6f}, true_scale={true_scale:.6f}, relative_error={relative_error:.6f}")
                        
                        # 扩展到5个字段（如果不足）
                        while len(prediction_errors_per_field) < 5:
                            prediction_errors_per_field.append(0.0)
                        
                        # 使用权重计算总误差
                        prediction_errors_array = np.array(prediction_errors_per_field[:5])
                        sample_error = np.sum(prediction_errors_array * error_weights)
                        
                        if i < 5:  # 调试前5个样本
                            print(f"样本 {i} 各字段误差: {prediction_errors_per_field[:5]}")
                            print(f"样本 {i} 误差权重: {error_weights}")
                            print(f"样本 {i} 加权总误差: {sample_error:.6f}")
                        
                except Exception as error_calc_error:
                    if i < 5:
                        print(f"样本 {i} 误差计算失败: {error_calc_error}")
                        import traceback
                        print(f"详细错误: {traceback.format_exc()}")
                    sample_error = 0.0
                    failed_samples += 1
            else:
                if i < 5:
                    print(f"样本 {i} 预测失败")
                sample_error = 0.0
                failed_samples += 1
                
        except Exception as general_error:
            if i < 5:
                print(f"样本 {i} 处理失败: {general_error}")
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
    
    # === 步骤3: 结果分析和选择 ===
    print(f"\nPA预测误差查询统计:")
    print(f"  - 总样本数: {len(unlabeled_data)}")
    print(f"  - 成功计算: {len(unlabeled_data) - failed_samples}")
    print(f"  - 失败样本: {failed_samples}")
    print(f"  - 成功率: {(len(unlabeled_data) - failed_samples) / len(unlabeled_data) * 100:.2f}%")
    
    # 误差统计 - 修改有效误差的判断条件
    valid_errors = [e for e in prediction_errors if e >= 0 and not np.isnan(e) and not np.isinf(e)]
    non_zero_errors = [e for e in prediction_errors if e > 0]
    
    print(f"  - 所有误差统计: 最小值={min(prediction_errors):.8f}, 最大值={max(prediction_errors):.8f}")
    print(f"  - 有效误差数量 (>=0): {len(valid_errors)}")
    print(f"  - 非零误差数量 (>0): {len(non_zero_errors)}")
    
    if valid_errors:
        print(f"  - 有效误差范围: {min(valid_errors):.8f} - {max(valid_errors):.8f}")
        print(f"  - 平均误差: {np.mean(valid_errors):.8f}")
        print(f"  - 误差标准差: {np.std(valid_errors):.8f}")
        
        # 检查是否所有有效误差都是0
        if all(e == 0 for e in valid_errors):
            print("  - 警告: 所有有效误差都为0，可能是预测与真实值完全相同")
            print("  - 分析原因: 可能是模型过拟合或数据预处理问题")
            
            # 检查前几个样本的详细情况
            print("  - 检查前3个样本的详细预测情况...")
            has_actual_difference = False
            
            for check_i in range(min(3, len(unlabeled_data))):
                sample = unlabeled_data[check_i]
                Y_true_check = np.array(sample[1])
                print(f"    样本{check_i} 真实值统计: 均值={Y_true_check.mean():.6f}, 标准差={Y_true_check.std():.6f}")
                print(f"    样本{check_i} 真实值范围: [{Y_true_check.min():.6f}, {Y_true_check.max():.6f}]")
                
                # 简单检查是否有变化
                if Y_true_check.std() > 1e-6:
                    has_actual_difference = True
            
            if has_actual_difference:
                print("  - 数据有变化，使用基于数据变异性的选择策略")
                # 基于真实值的变异性选择样本
                variance_scores = []
                for i, sample in enumerate(unlabeled_data):
                    Y_true = np.array(sample[1])
                    # 计算数据的变异性
                    variance_score = np.var(Y_true) + np.mean(np.abs(Y_true))
                    variance_scores.append(variance_score)
                
                # 选择变异性最大的样本
                selected_idx = np.argsort(variance_scores)[-select_num:]
                selected_scores = [variance_scores[i] for i in selected_idx]
                print(f"  - 基于数据变异性选择的样本变异性分数: {selected_scores}")
                print(f"PA查询完成，基于数据变异性选择了 {len(selected_idx)} 个样本")
                return selected_idx.tolist()
            else:
                print("  - 数据无明显变化，使用随机选择策略")
                import random
                selected_idx = random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
                print(f"PA查询完成，随机选择了 {len(selected_idx)} 个样本（因为所有误差为0且数据无变化）")
                return selected_idx
    else:
        print("  - 警告: 所有预测误差都无效")
    
    # 选择预测误差最大的样本
    if len(valid_errors) == 0:
        print("所有样本的预测误差都无效，使用几何复杂度评分")
        return geometry_based_fallback(unlabeled_data, select_num)
    
    # 按预测误差排序，选择误差最大的
    selected_idx = np.argsort(prediction_errors)[-select_num:]
    selected_errors = [prediction_errors[i] for i in selected_idx]
    
    print(f"  - 选中样本的预测误差: {selected_errors}")
    print(f"PA查询完成，选择了 {len(selected_idx)} 个预测误差最大的样本")
    
    return selected_idx.tolist()


def bz_query_simple(model_tuple, unlabeled_data, select_num):
    """
    简化版BZ策略 - 直接基于数据特征计算complexity metrics
    
    不依赖复杂的数据集重建，直接分析数据特征
    """
    model, metric_func, device = model_tuple
    print(f"简化BZ查询开始，设备: {device}, 样本数: {len(unlabeled_data)}")
    
    sample_metrics = []
    
    for i, sample in enumerate(tqdm(unlabeled_data, desc="计算样本复杂度")):
        try:
            # 获取坐标和值数据
            coords = np.array(sample[0])  # [N, 3]
            values = np.array(sample[1])  # [N, 5] 
            
            # 计算多种复杂度指标
            
            # 1. 坐标复杂度 - 几何形状的复杂性
            coord_std = np.std(coords, axis=0).mean()
            coord_range = np.ptp(coords, axis=0).mean()  # peak-to-peak
            
            # 2. 值复杂度 - 物理量的变化复杂性
            value_std = np.std(values, axis=0).mean()
            value_range = np.ptp(values, axis=0).mean()
            
            # 3. 空间梯度复杂度 - 相邻点之间的变化
            if len(coords) > 1:
                coord_diff = np.diff(coords, axis=0)
                coord_gradient = np.linalg.norm(coord_diff, axis=1).mean()
                
                value_diff = np.diff(values, axis=0) 
                value_gradient = np.linalg.norm(value_diff, axis=1).mean()
            else:
                coord_gradient = 0
                value_gradient = 0
            
            # 4. 点云密度
            density_score = len(coords) / (coord_range + 1e-8)
            
            # 综合复杂度评分 (组合多个指标)
            complexity_score = (
                0.3 * coord_std +           # 坐标分布复杂度
                0.2 * coord_range +         # 几何尺度
                0.3 * value_std +           # 值变化复杂度  
                0.1 * value_range +         # 值范围
                0.05 * coord_gradient +     # 空间梯度
                0.05 * value_gradient +     # 值梯度
                0.01 * density_score        # 密度贡献
            )
            
            # 添加小的随机扰动避免完全相同
            complexity_score += np.random.random() * 1e-6
            
            sample_metrics.append(complexity_score)
            
            if i < 5:  # 调试前5个样本
                print(f"样本 {i}: 复杂度={complexity_score:.6f} (坐标std={coord_std:.3f}, 值std={value_std:.3f})")
            
        except Exception as e:
            # 失败时使用随机值
            random_metric = np.random.random() * 0.01 + 0.01
            sample_metrics.append(random_metric)
            if i < 5:
                print(f"样本 {i}: 计算失败，使用随机值 {random_metric:.6f}")
    
    # 分析metrics
    print(f"\n简化BZ查询统计:")
    print(f"  - 总样本数: {len(sample_metrics)}")
    print(f"  - metric范围: {min(sample_metrics):.6f} - {max(sample_metrics):.6f}")
    print(f"  - 平均metric: {np.mean(sample_metrics):.6f}")
    print(f"  - metric标准差: {np.std(sample_metrics):.6f}")
    
    # 选择复杂度最高的样本
    selected_idx = np.argsort(sample_metrics)[-select_num:]
    selected_metrics = [sample_metrics[i] for i in selected_idx]
    
    print(f"  - 选中样本的复杂度: {selected_metrics[:10]}{'...' if len(selected_metrics) > 10 else ''}")
    print(f"简化BZ查询完成，选择了 {len(selected_idx)} 个复杂度最高的样本")
    
    return selected_idx.tolist()


def bz_query(model_tuple, unlabeled_data, select_num):
    """
    BZ策略 - 基于模型预测误差的样本选择
    
    唯一标准：模型预测值与标准值的误差
    """
    model, metric_func, device = model_tuple
    print(f"BZ策略：基于预测误差的样本选择")
    print(f"处理 {len(unlabeled_data)} 个候选样本，选择 {select_num} 个")
    
    model.eval()
    sample_errors = []
    
    with torch.no_grad():
        for i, sample in enumerate(unlabeled_data):
            if i % 100 == 0:
                print(f"进度: {i}/{len(unlabeled_data)}")
            
            try:
                # 使用正确的样本数据结构
                coords = sample[0]  # np.ndarray (N, 3)
                true_values = sample[1]  # np.ndarray (N, 5)
                theta = sample[2]  # np.ndarray (2,)
                branch_data = sample[3]  # tuple
                
                # 转换为tensor
                coords_tensor = torch.tensor(coords, dtype=torch.float32).to(device)
                true_values_tensor = torch.tensor(true_values, dtype=torch.float32).to(device)
                theta_tensor = torch.tensor(theta, dtype=torch.float32).to(device)
                
                # 限制点数量避免内存问题
                max_points = 33000
                num_points = min(coords_tensor.shape[0], max_points)
                
                coords_limited = coords_tensor[:num_points]
                true_values_limited = true_values_tensor[:num_points]
                
                # 创建图结构
                import dgl
                
                # 创建简单的k-近邻图
                k = min(6, num_points - 1)  # 每个节点连接最近的k个邻居
                
                edges_src = []
                edges_dst = []
                
                # 简化：创建链式连接图
                for j in range(num_points - 1):
                    edges_src.append(j)
                    edges_dst.append(j + 1)
                
                # 添加一些额外连接
                for j in range(0, num_points, 10):
                    for k_offset in range(1, min(k, num_points - j)):
                        if j + k_offset < num_points:
                            edges_src.append(j)
                            edges_dst.append(j + k_offset)
                
                if len(edges_src) == 0:  # 确保至少有一条边
                    edges_src = [0]
                    edges_dst = [0 if num_points == 1 else 1]
                
                g = dgl.graph((edges_src, edges_dst), num_nodes=num_points)
                g = g.to(device)
                
                # 设置节点特征
                g.ndata['x'] = coords_limited  # 输入特征（坐标）
                g.ndata['y'] = true_values_limited  # 真实值（用于计算误差）
                
                # 准备模型输入
                u_p = theta_tensor.unsqueeze(0)  # [1, 2]
                
                # 处理分支数据 - 使用pad_sequence模拟数据加载器的格式
                if isinstance(branch_data, tuple) and len(branch_data) > 0:
                    # 原始分支数据
                    branch_array = branch_data[0][:num_points]
                    
                    # 转换为张量并确保正确的形状
                    if isinstance(branch_array, np.ndarray):
                        branch_tensor = torch.tensor(branch_array, dtype=torch.float32).to(device)
                    else:
                        branch_tensor = torch.tensor(branch_array, dtype=torch.float32).to(device)
                    
                    # 确保是2D张量 [N, features]
                    if len(branch_tensor.shape) == 1:
                        branch_tensor = branch_tensor.unsqueeze(-1)  # [N, 1]
                    
                    # 使用pad_sequence来模拟批处理，然后permute得到正确格式
                    # pad_sequence期望输入是 [tensor1, tensor2, ...] 的列表
                    # 对于单个样本，我们创建包含一个张量的列表
                    padded = pad_sequence([branch_tensor]).permute(1, 0, 2)  # [B=1, T, F]
                    
                    g_u = MultipleTensors([padded])
                else:
                    # 创建零张量，使用pad_sequence格式
                    zero_tensor = torch.zeros((num_points, 1), dtype=torch.float32).to(device)
                    padded = pad_sequence([zero_tensor]).permute(1, 0, 2)  # [B=1, T, F]
                    g_u = MultipleTensors([padded])
                
                # 模型预测
                pred = model(g, u_p, g_u)
                
                # 计算预测误差
                target = true_values_limited
                
                # 确保形状匹配
                if pred.shape != target.shape:
                    min_rows = min(pred.shape[0], target.shape[0])
                    min_cols = min(pred.shape[1] if len(pred.shape) > 1 else 1, 
                                  target.shape[1] if len(target.shape) > 1 else 1)
                    pred = pred[:min_rows, :min_cols]
                    target = target[:min_rows, :min_cols]
                
                # 计算MSE误差
                mse_error = torch.mean((pred - target) ** 2).item()
                sample_errors.append(mse_error)
                
                # 记录每个样本的详细误差信息到日志
                print(f"样本 {i}: 预测误差 = {mse_error:.6f}")
                print(f"  数据点数: {num_points}")
                print(f"  预测形状: {pred.shape}, 真实值形状: {target.shape}")
                print(f"  坐标范围: [{coords_limited.min().item():.4f}, {coords_limited.max().item():.4f}]")
                print(f"  真实值范围: [{target.min().item():.4f}, {target.max().item():.4f}]")
                print(f"  预测值范围: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                
                # 计算每个输出维度的单独误差
                dim_errors = []
                for dim in range(target.shape[1]):
                    dim_error = torch.mean((pred[:, dim] - target[:, dim]) ** 2).item()
                    dim_errors.append(dim_error)
                    print(f"    维度 {dim}: 误差 = {dim_error:.6f}")
                
                # 记录样本的物理参数
                print(f"  物理参数 theta: [{theta_tensor[0].item():.4f}, {theta_tensor[1].item():.4f}]")
                
                if i < 3:  # 前3个样本显示更详细信息
                    print(f"  详细预测前5点:")
                    for pt in range(min(5, pred.shape[0])):
                        print(f"    点{pt}: 预测=[{', '.join(f'{pred[pt,d].item():.4f}' for d in range(pred.shape[1]))}]")
                        print(f"         真实=[{', '.join(f'{target[pt,d].item():.4f}' for d in range(target.shape[1]))}]")
                
            except Exception as e:
                print(f"样本 {i} 预测失败: {e}")
                print(f"  样本数据结构: 坐标{coords.shape if 'coords' in locals() else 'N/A'}, "
                      f"数值{values.shape if 'values' in locals() else 'N/A'}, "
                      f"参数{theta.shape if 'theta' in locals() else 'N/A'}")
                if i < 3:  # 前3个失败样本显示详细错误
                    import traceback
                    print(f"  详细错误: {traceback.format_exc()}")
                # 使用较大的误差值（表示难预测）
                sample_errors.append(999.0)
                print(f"  设置失败样本误差为: 999.0")
    
    # 统计信息
    valid_errors = [e for e in sample_errors if e != 999.0]
    
    print(f"\n预测误差统计:")
    print(f"  成功预测样本数: {len(valid_errors)}")
    print(f"  失败样本数: {len(sample_errors) - len(valid_errors)}")
    
    if len(valid_errors) > 0:
        print(f"  有效误差最小值: {min(valid_errors):.6f}")
        print(f"  有效误差最大值: {max(valid_errors):.6f}")
        print(f"  有效误差平均值: {np.mean(valid_errors):.6f}")
        print(f"  有效误差标准差: {np.std(valid_errors):.6f}")
    else:
        print(f"  所有样本预测都失败了")
    
    # 选择预测误差最大的样本（模型最难预测的样本）
    selected_idx = np.argsort(sample_errors)[-select_num:]
    selected_errors = [sample_errors[i] for i in selected_idx]
    
    print(f"  选中样本的误差范围: {min(selected_errors):.6f} - {max(selected_errors):.6f}")
    
    # 详细记录每个选中样本的信息
    print(f"\n=== 选中样本详细信息 ===")
    sorted_selection = sorted(zip(selected_idx, selected_errors), key=lambda x: x[1], reverse=True)
    for rank, (idx, error) in enumerate(sorted_selection):
        print(f"排名 {rank+1}: 样本索引 {idx}, 预测误差 {error:.6f}")
    
    # 记录误差分布统计
    if len(valid_errors) > 0:
        error_percentiles = np.percentile(valid_errors, [10, 25, 50, 75, 90])
        print(f"\n=== 误差分布统计 ===")
        print(f"  10%分位数: {error_percentiles[0]:.6f}")
        print(f"  25%分位数: {error_percentiles[1]:.6f}")
        print(f"  50%分位数(中位数): {error_percentiles[2]:.6f}")
        print(f"  75%分位数: {error_percentiles[3]:.6f}")
        print(f"  90%分位数: {error_percentiles[4]:.6f}")
        
        # 选择阈值分析
        selection_threshold = min(selected_errors)
        above_threshold = sum(1 for e in valid_errors if e >= selection_threshold)
        print(f"  选择阈值: {selection_threshold:.6f}")
        print(f"  超过阈值的样本数: {above_threshold}")
    
    print(f"BZ策略完成，选择了 {len(selected_idx)} 个高误差样本")
    
    return selected_idx.tolist()


def bz_query_model_based(model_tuple, unlabeled_data, select_num):
    """
    BZ策略 - 基于GNOT模型的逐样本预测误差策略
    
    为每个未标注样本单独计算预测误差，选择误差最大的样本
    
    Args:
        model_tuple: (model, metric_func, device)
        unlabeled_data: 未标注数据列表
        select_num: 需要选择的样本数量
    
    Returns:
        selected_idx: 选择的样本索引列表，按误差从大到小排序
    """
    model, metric_func, device = model_tuple
    print(f"BZ查询开始，设备: {device}, 样本数: {len(unlabeled_data)}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"中间数据目录: {temp_data_dir}")
    
    model.eval()
    sample_metrics = []
    failed_samples = 0
    
    for i, sample in enumerate(unlabeled_data):
        if i % 50 == 0:
            print(f"BZ进度: {i}/{len(unlabeled_data)}")
        
        if i < 3:
            print(f"开始处理样本 {i}，当前sample_metrics长度: {len(sample_metrics)}")
        
        sample_metric = 0.0
        
        try:
            if i < 3:  # 调试前3个样本
                print(f"开始处理样本 {i}...")
                
            # === 步骤1: 构建临时数据集 ===
            import time
            timestamp = int(time.time() * 1000000)
            temp_file = os.path.join(temp_data_dir, f'bz_temp_{i}_{timestamp}.pkl')
            
            if i < 3:
                print(f"  创建临时文件: {temp_file}")
            
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            # 复制到标准位置
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            # 加载数据
            args = get_al_args()
            args.dataset = 'al_bz'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            if i < 3:
                print(f"  数据集加载成功，数据量: {len(tmp_dataset)}")
            
            # === 步骤2: 直接计算单样本的预测误差 ===
            with torch.no_grad():
                for batch_data in tmp_loader:
                    if i < 3:
                        print(f"  开始模型预测...")
                    g, u_p, g_u = batch_data
                    
                    # 设备转移
                    g = g.to(device)
                    u_p = u_p.to(device)
                    if hasattr(g_u, 'to'):
                        g_u = g_u.to(device)
                    elif hasattr(g_u, 'tensors'):
                        g_u.tensors = [t.to(device) if torch.is_tensor(t) else t for t in g_u.tensors]
                    
                    model = model.to(device)
                    
                    # 模型预测
                    pred = model(g, u_p, g_u)
                    
                    if i < 3:
                        print(f"  模型预测完成，pred shape: {pred.shape}")
                    
                    # 获取真实值
                    if hasattr(g, 'ndata') and 'y' in g.ndata:
                        target = g.ndata['y']
                        if i < 3:
                            print(f"  从图中获取target，shape: {target.shape}")
                    else:
                        # 如果图中没有y，从原始数据获取
                        target = torch.tensor(np.array(sample[1]), dtype=torch.float32).to(device)
                        if i < 3:
                            print(f"  从原始数据获取target，shape: {target.shape}")
                    
                    # 计算误差 - 使用metric_func
                    if pred.shape != target.shape:
                        if i < 3:
                            print(f"  形状不匹配: pred {pred.shape} vs target {target.shape}")
                        # 形状不匹配时的处理
                        min_size = min(pred.size(0), target.size(0))
                        min_dims = min(pred.size(1) if len(pred.shape) > 1 else 1, 
                                      target.size(1) if len(target.shape) > 1 else 1)
                        pred = pred[:min_size, :min_dims] if len(pred.shape) > 1 else pred[:min_size]
                        target = target[:min_size, :min_dims] if len(target.shape) > 1 else target[:min_size]
                    
                    # 使用metric_func计算误差
                    try:
                        if metric_func is not None:
                            # 直接使用MSE，避免metric_func的复杂性
                            sample_metric = torch.mean((pred - target) ** 2).item()
                            if i < 3:
                                print(f"  使用MSE计算，结果: {sample_metric}")
                        else:
                            # 如果没有metric_func，使用MSE
                            sample_metric = torch.mean((pred - target) ** 2).item()
                            if i < 3:
                                print(f"  使用MSE计算，结果: {sample_metric}")
                    except Exception as metric_error:
                        if i < 3:
                            print(f"  metric计算失败: {metric_error}")
                        # 如果metric_func失败，使用简单的L2范数
                        diff = pred - target
                        sample_metric = torch.norm(diff).item()
                        if i < 3:
                            print(f"  回退到L2范数，结果: {sample_metric}")
                    
                    break  # 只处理第一个（也是唯一的）batch
                break
            
            # 调试前5个样本
            if i < 5:
                print(f"样本 {i}: metric={sample_metric:.6f}")
                        
        except Exception as general_error:
            if i < 5:
                print(f"样本 {i} 处理失败: {general_error}")
                import traceback
                print(f"详细错误: {traceback.format_exc()}")
            # 设置默认metric值而不是0，避免被过滤掉
            sample_metric = np.random.random() * 0.001 + 0.001  # 小的随机值
            failed_samples += 1
        
        # 确保每个样本都有metric值被添加
        if i < 3:
            print(f"  准备添加sample_metric到列表: {sample_metric}")
        sample_metrics.append(sample_metric)
        if i < 3:
            print(f"  sample_metrics列表长度: {len(sample_metrics)}")
            print(f"  最新添加的值: {sample_metrics[-1]}")
        
        # 清理临时文件
        try:
            if 'temp_file' in locals() and os.path.exists(temp_file):
                os.remove(temp_file)
            if 'standard_test_path' in locals() and os.path.exists(standard_test_path):
                os.remove(standard_test_path)
        except:
            pass
        
        if i < 3:
            print(f"样本 {i} 处理完成，继续下一个样本...")
        
        # 添加一个提前退出机制来测试
        if i == 0:
            print(f"第一个样本处理完成，sample_metrics: {sample_metrics}")
            print(f"继续处理剩余样本...")
    
    # === 步骤3: 结果分析和选择 ===
    print(f"\nBZ查询统计:")
    print(f"  - 总样本数: {len(unlabeled_data)}")
    print(f"  - 成功计算: {len(unlabeled_data) - failed_samples}")
    print(f"  - 失败样本: {failed_samples}")
    print(f"  - 成功率: {(len(unlabeled_data) - failed_samples) / len(unlabeled_data) * 100:.2f}%")
    
    # 调试：显示前10个sample_metrics的值
    print(f"  - 前10个sample_metrics: {sample_metrics[:10]}")
    
    # metric统计 - 修复逻辑
    # 只过滤掉NaN和inf，保留0值
    all_metrics = [m for m in sample_metrics if not np.isnan(m) and not np.isinf(m)]
    # 有效metric应该是大于0的
    valid_metrics = [m for m in all_metrics if m > 0]
    
    print(f"  - 有效metric数量 (>0): {len(valid_metrics)}")
    print(f"  - 全部metric数量: {len(all_metrics)}")
    
    if len(all_metrics) > 0:
        print(f"  - 所有metric统计: 最小值={min(all_metrics):.6f}, 最大值={max(all_metrics):.6f}")
        print(f"  - metric范围: {min(all_metrics):.6f} - {max(all_metrics):.6f}")
        print(f"  - 平均metric: {np.mean(all_metrics):.6f}")
        print(f"  - metric标准差: {np.std(all_metrics):.6f}")
        
        # 检查是否所有metric都相同或都是0
        if np.std(all_metrics) < 1e-6:
            print("  - 警告: 所有metric都相同，可能存在问题")
            if max(all_metrics) == 0:
                print("  - 所有metric都是0，使用几何复杂度评分")
                return geometry_based_fallback(unlabeled_data, select_num)
            else:
                print("  - 使用随机选择策略")
                import random
                selected_idx = random.sample(range(len(unlabeled_data)), min(select_num, len(unlabeled_data)))
                print(f"BZ查询完成，随机选择了 {len(selected_idx)} 个样本（因为所有metric相同）")
                return selected_idx
    else:
        print("  - 警告: 所有metric都无效")
    
    # 选择metric最大的样本（表现最差的样本）
    if len(all_metrics) == 0:
        print("所有样本的metric都无效，使用几何复杂度评分")
        return geometry_based_fallback(unlabeled_data, select_num)
    
    # 按metric排序，选择最大的（最需要改进的）
    selected_idx = np.argsort(sample_metrics)[-select_num:]
    selected_metrics = [sample_metrics[i] for i in selected_idx]
    
    print(f"  - 选中样本的metric: {selected_metrics[:10]}{'...' if len(selected_metrics) > 10 else ''}")
    print(f"BZ查询完成，选择了 {len(selected_idx)} 个metric最大的样本")
    
    return selected_idx.tolist()


def pa_query_prediction_error(model_tuple, unlabeled_data, select_num, 
                            pressure_weight=1.0, wall_shear_weight=1.0, 
                            x_wall_shear_weight=1.0, y_wall_shear_weight=1.0, z_wall_shear_weight=1.0):
    """
    基于预测误差的PA查询策略 - 可配置各字段权重
    
    Args:
        model_tuple: (model, metric_func, device)
        unlabeled_data: 未标注数据列表
        select_num: 需要选择的样本数量
        pressure_weight: pressure字段误差权重
        wall_shear_weight: wall-shear字段误差权重
        x_wall_shear_weight: x-wall-shear字段误差权重
        y_wall_shear_weight: y-wall-shear字段误差权重
        z_wall_shear_weight: z-wall-shear字段误差权重
    
    Returns:
        selected_idx: 选择的样本索引列表，按预测误差从大到小排序
    """
    error_weights = np.array([
        pressure_weight, 
        wall_shear_weight, 
        x_wall_shear_weight, 
        y_wall_shear_weight, 
        z_wall_shear_weight
    ])
    
    print(f"使用自定义权重的PA预测误差查询:")
    print(f"  - pressure权重: {pressure_weight}")
    print(f"  - wall-shear权重: {wall_shear_weight}")
    print(f"  - x-wall-shear权重: {x_wall_shear_weight}")
    print(f"  - y-wall-shear权重: {y_wall_shear_weight}")
    print(f"  - z-wall-shear权重: {z_wall_shear_weight}")
    
    return pa_query_fixed(model_tuple, unlabeled_data, select_num, error_weights)


def pa_query_average_error(model_tuple, unlabeled_data, select_num):
    """
    基于五个值平均误差的PA查询策略（用户需求的默认版本）
    
    Args:
        model_tuple: (model, metric_func, device)
        unlabeled_data: 未标注数据列表
        select_num: 需要选择的样本数量
    
    Returns:
        selected_idx: 选择的样本索引列表，按平均预测误差从大到小排序
    """
    # 使用等权重（即计算五个值的平均误差）
    error_weights = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    
    print("使用PA预测误差查询策略（五个值等权重平均误差）")
    print("字段顺序: pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear")
    
    return pa_query_fixed(model_tuple, unlabeled_data, select_num, error_weights)


# 为了向后兼容，创建旧版本的别名
def pa_query_physics_based(model_tuple, unlabeled_data, select_num, rho=1.0, mu=1.0, lam=1e-4):
    """
    基于物理一致性的PA查询策略（已删除，使用几何特征替代）
    """
    print("物理一致性PA策略已删除，使用几何特征替代")
    return geometry_based_fallback(unlabeled_data, select_num)
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
            
            elif strategy == 'bz':
                try:
                    print("执行BZ策略（基于GNOT模型测试框架）...")
                    selected_idx = bz_query(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"BZ策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"BZ策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 使用几何回退
                    selected_idx = geometry_based_fallback(unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"已使用几何回退策略")
                
            elif strategy == 'pa':
                try:
                    print("执行PA策略（基于预测误差版本）...")
                    # 使用基于平均预测误差的PA策略（用户需求的默认版本）
                    selected_idx = pa_query_average_error(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"PA预测误差策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"PA预测误差策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 使用几何回退
                    selected_idx = geometry_based_fallback(unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"已使用几何回退策略")
            
            elif strategy == 'pa_physics':
                try:
                    print("执行PA策略（基于物理一致性版本）...")
                    selected_idx = pa_query_physics_based(model_tuple, unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"PA物理一致性策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"PA物理一致性策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 使用几何回退
                    selected_idx = geometry_based_fallback(unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"已使用几何回退策略")
            
            elif strategy == 'pa_custom':
                try:
                    print("执行PA策略（自定义权重版本）...")
                    # 这里可以自定义各字段的权重
                    selected_idx = pa_query_prediction_error(
                        model_tuple, unlabeled_data, select_num,
                        pressure_weight=1.0,        # pressure权重
                        wall_shear_weight=1.0,      # wall-shear权重
                        x_wall_shear_weight=1.0,    # x-wall-shear权重
                        y_wall_shear_weight=1.0,    # y-wall-shear权重
                        z_wall_shear_weight=1.0     # z-wall-shear权重
                    )
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"PA自定义权重策略成功选择 {len(newly_selected)} 个样本")
                except Exception as e:
                    print(f"PA自定义权重策略失败: {e}")
                    import traceback
                    print(f"详细错误: {traceback.format_exc()}")
                    # 使用几何回退
                    selected_idx = geometry_based_fallback(unlabeled_data, select_num)
                    newly_selected = [unlabeled_data[i] for i in selected_idx]
                    print(f"已使用几何回退策略")
                
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
    strategy = 'bz'  # 当前使用random策略
    # strategy = 'gv'        # 高精度几何方法
    # strategy = 'qbc'       # 查询委员会
    # strategy = 'pa'        # 物理感知
    
    print(f"=== 策略选择 ===")
    print(f"当前使用策略: {strategy}")
    
    # 定义路径
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"  # 数据更新目录
    output_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz/al_rounds16"  # 结果保存目录
    temp_data_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz/data"  # 中间数据目录
    
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

    # === 修改数据集名称为 al_bz ===
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"  # 数据更新目录
    output_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz/al_rounds16"  # 结果保存目录
    temp_data_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz/data"  # 中间数据目录
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
            select_num=500,  # 临时减少到5个样本以快速测试详细日志
            seed=42,
            output_dir=output_dir,  # 结果保存目录
            data_update_dir=data_update_dir,  # 数据更新目录
            strategy=strategy
        )