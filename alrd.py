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
        dataset = 'al_rd'
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
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_rd/data'
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
        args.dataset = 'al_rd'  # 确保使用正确的数据集名称
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
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_rd/data'
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
        args.dataset = 'al_rd'  # 确保使用正确的数据集名称
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


def qbc_query(model_tuple, unlabeled_data, select_num, mc_times=10):
    """
    Query by Committee - 使用Monte Carlo Dropout选择不确定性最高的样本
    修改为使用独立的中间数据目录
    """
    model, metric_func, device = model_tuple
    model.train()  # 启用dropout
    variances = []
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_rd/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    
    print(f"开始QBC查询，使用设备: {device}")
    print(f"中间数据目录: {temp_data_dir}")
    
    for i, sample in enumerate(unlabeled_data):
        if i % 100 == 0:
            print(f"QBC进度: {i}/{len(unlabeled_data)}")
        
        # 使用时间戳创建唯一的临时文件
        import time
        timestamp = int(time.time() * 1000000)
        temp_file = os.path.join(temp_data_dir, f'qbc_temp_{i}_{timestamp}.pkl')
        
        try:
            # 为每个样本构建临时数据集
            with open(temp_file, 'wb') as f:
                pickle.dump([sample], f)
            
            # 复制到标准位置
            standard_test_path = os.path.join(temp_data_dir, 'al_test.pkl')
            import shutil
            shutil.copy2(temp_file, standard_test_path)
            
            args = get_al_args()
            args.dataset = 'al_rd'
            _, tmp_dataset = get_dataset(args)
            tmp_loader = MIODataLoader(tmp_dataset, batch_size=1, shuffle=False, drop_last=False)
            
            preds = []
            for mc_iter in range(mc_times):
                with torch.no_grad():
                    for g, u_p, g_u in tmp_loader:
                        # 关键修复：确保所有输入都在正确的设备上
                        g = g.to(device)
                        u_p = u_p.to(device)
                        g_u = g_u.to(device)
                        
                        # 模型预测
                        pred = model(g, u_p, g_u).cpu().numpy()
                        preds.append(pred)
                        break  # 只需要第一个batch
            
            # 计算方差
            preds = np.stack(preds, axis=0)
            var = np.mean(np.var(preds, axis=0))
            variances.append(var)
            
        except Exception as e:
            print(f"QBC样本 {i} 处理失败: {e}")
            variances.append(0.0)
        finally:
            # 清理临时文件
            try:
                os.remove(temp_file)
                if os.path.exists(standard_test_path):
                    os.remove(standard_test_path)
            except:
                pass
    
    # 选择方差最大的样本
    selected_idx = np.argsort(variances)[-select_num:]
    print(f"QBC查询完成，选择了 {len(selected_idx)} 个高不确定性样本")
    return selected_idx.tolist()

def pa_query(model_tuple, unlabeled_data, select_num, rho=1.0, mu=1.0, lam=1e-4):
    """
    Physics-Aware查询 - 选择违反物理定律最严重的样本
    修改为使用独立的中间数据目录
    """
    model, metric_func, device = model_tuple
    print(f"PA查询开始，设备: {device}, 样本数: {len(unlabeled_data)}")
    
    # 创建独立的中间数据目录
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_rd/data'
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
            args.dataset = 'al_rd'
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


# === 4. 主动学习主循环 ===
def random_active_learning(
    rounds=5, select_num=5, seed=42, 
    output_dir='./al_rounds', data_update_dir=None, strategy='random'
):
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
        print(f"\n请确保在 {data_update_dir} 目录中放置以下文件:")
        print("  - al_labeled.pkl (初始训练数据)")
        print("  - al_unlabeled.pkl (未标注数据池)")
        print("  - al_test.pkl (测试数据)")
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
    temp_data_dir = '/home/v-wenliao/gnot/GNOT/data/al_rd/data'
    os.makedirs(temp_data_dir, exist_ok=True)
    print(f"中间数据工作目录: {temp_data_dir}")
    
    # 创建目录
    os.makedirs(output_dir, exist_ok=True)  # 结果保存目录
    
    random.seed(seed)

    # 准备CSV文件记录性能指标
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear', 'train_num', 'round'])

    for r in range(rounds):
        print(f"\n=== 主动学习第{r+1}轮 ===")
        
        # === 1. 更新数据文件到指定目录（供下轮使用） ===
        # 覆盖更新主动学习数据文件
        with open(labeled_path, 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(unlabeled_path, 'wb') as f:
            pickle.dump(unlabeled_data, f)
        with open(test_path, 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"第{r+1}轮数据已更新到: {data_update_dir}")
        print(f"  - al_labeled.pkl: {len(labeled_data)} 个训练样本")
        print(f"  - al_unlabeled.pkl: {len(unlabeled_data)} 个未标注样本")
        print(f"  - al_test.pkl: {len(test_data)} 个测试样本")
        
        # === 2. 保存轮次结果（仅保存训练集和测试集） ===
        round_data_dir = os.path.join(output_dir, f'round_{r+1}')
        os.makedirs(round_data_dir, exist_ok=True)
        
        # 只保存训练集和测试集（节省空间）
        with open(os.path.join(round_data_dir, 'train_data.pkl'), 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(os.path.join(round_data_dir, 'test_data.pkl'), 'wb') as f:
            pickle.dump(test_data, f)
        
        print(f"第{r+1}轮结果已保存到: {round_data_dir}")
        print(f"  - train_data.pkl: 该轮训练数据 ({len(labeled_data)} 样本)")
        print(f"  - test_data.pkl: 测试数据 ({len(test_data)} 样本)")
        
        # === 3. 训练和评估 ===
        model_tuple = train_model(labeled_data)
        metric = evaluate_model(model_tuple, test_data)
        print(f"第{r+1}轮模型性能: {metric}")

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
            print("未标注池已空，主动学习提前结束。")
            break
            
        if len(unlabeled_data) < select_num:
            select_num = len(unlabeled_data)
            
        # === 5. 策略选择新样本 ===
        print(f"开始{strategy}策略选择，从 {len(unlabeled_data)} 个候选中选择 {select_num} 个...")
        
        if strategy == 'random':
            newly_selected = random.sample(unlabeled_data, select_num)
        elif strategy == 'gv':
            selected_idx = geometry_variance_query(labeled_data, unlabeled_data, select_num)
            newly_selected = [unlabeled_data[i] for i in selected_idx]
        elif strategy == 'gv_fast':
            selected_idx = geometry_variance_query_fast(labeled_data, unlabeled_data, select_num)
            newly_selected = [unlabeled_data[i] for i in selected_idx]
        elif strategy == 'qbc':
            try:
                selected_idx = qbc_query(model_tuple, unlabeled_data, select_num)
                newly_selected = [unlabeled_data[i] for i in selected_idx]
            except Exception as e:
                print(f"QBC查询失败: {e}, 回退到随机采样")
                newly_selected = random.sample(unlabeled_data, select_num)
        elif strategy == 'pa':
            try:
                selected_idx = pa_query(model_tuple, unlabeled_data, select_num)
                newly_selected = [unlabeled_data[i] for i in selected_idx]
            except Exception as e:
                print(f"PA查询失败: {e}, 回退到随机采样")
                newly_selected = random.sample(unlabeled_data, select_num)
        else:
            print(f"未知策略: {strategy}, 使用随机采样")
            newly_selected = random.sample(unlabeled_data, select_num)

        # === 6. 保存策略信息 ===
        strategy_info = {
            'strategy': strategy,
            'select_num': len(newly_selected),
            'round': r + 1,
            'total_labeled': len(labeled_data),
            'total_unlabeled': len(unlabeled_data),
            'data_update_dir': data_update_dir,
            'temp_data_dir': temp_data_dir,
            'files_updated': {
                'al_labeled.pkl': f"{len(labeled_data)} samples",
                'al_unlabeled.pkl': f"{len(unlabeled_data)} samples", 
                'al_test.pkl': f"{len(test_data)} samples"
            }
        }
        with open(os.path.join(round_data_dir, 'strategy_info.json'), 'w') as f:
            json.dump(strategy_info, f, indent=2)

        # === 7. 更新数据集 ===
        labeled_data.extend(newly_selected)
        newly_selected_ids = set(id(x) for x in newly_selected)
        unlabeled_data = [x for x in unlabeled_data if id(x) not in newly_selected_ids]

        print(f"第{r+1}轮完成:")
        print(f"  - 本轮新增样本: {len(newly_selected)}")
        print(f"  - 累计已标注: {len(labeled_data)}")
        print(f"  - 剩余未标注: {len(unlabeled_data)}")

    # === 8. 最终更新和总结 ===
    # 最终更新数据文件
    with open(labeled_path, 'wb') as f:
        pickle.dump(labeled_data, f)
    with open(unlabeled_path, 'wb') as f:
        pickle.dump(unlabeled_data, f)
    
    print(f"\n=== 最终数据已更新到 {data_update_dir} ===")
    print(f"  - al_labeled.pkl: {len(labeled_data)} 个最终训练样本")
    print(f"  - al_unlabeled.pkl: {len(unlabeled_data)} 个剩余未标注样本")
    print(f"  - al_test.pkl: {len(test_data)} 个测试样本")

    print(f"\n=== 主动学习流程结束 ===")
    print(f"结果保存位置:")
    print(f"  - 性能记录: {csv_path}")
    print(f"  - 轮次结果: {output_dir}")
    print(f"  - 数据更新: {data_update_dir}")
    print(f"  - 中间数据: {temp_data_dir}")

    
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
    strategy = 'random'  # 当前使用random策略
    # strategy = 'gv'        # 高精度几何方法
    # strategy = 'qbc'       # 查询委员会
    # strategy = 'pa'        # 物理感知
    
    print(f"=== 策略选择 ===")
    print(f"当前使用策略: {strategy}")
    
    # 定义路径
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_rd"  # 数据更新目录
    output_dir = "/home/v-wenliao/gnot/GNOT/data/al_rd/al_rounds16"  # 结果保存目录
    temp_data_dir = "/home/v-wenliao/gnot/GNOT/data/al_rd/data"  # 中间数据目录
    
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
        random_active_learning(
            rounds=6,
            select_num=200,
            seed=42,
            output_dir=output_dir,  # 结果保存目录
            data_update_dir=data_update_dir,  # 数据更新目录
            strategy=strategy
        )