import os
import pickle
import random
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import csv
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import torch
import csv

# 设置matplotlib后端以避免PIL兼容性问题
import matplotlib
matplotlib.use('Agg')

# === 1. 导入你的GNOT训练和评估相关函数 ===
from train import train, validate_epoch
from args import get_args
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from utils import get_seed

import os
import json
import numpy as np
import pandas as pd


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
# 这里定义一个函数来获取参数，方便后续调用

def get_args():
    class Args:
        # 数据与运行相关
        dataset = 'al_tmp_train'
        component = 'all'
        seed = 2023
        space_dim = 3
        gpu = 0
        use_tb = 0
        comment = ""
        train_num = 'all'
        test_num = 'all'
        sort_data = 0
        normalize_x = 'unit'
        use_normalizer = 'unit'
        # 训练相关 - 针对多GPU优化
        epochs = 500
        optimizer = 'AdamW'
        lr = 1e-3
        weight_decay = 5e-6
        grad_clip = 1000.0
        batch_size = 32  # 增加基础batch size，多GPU时会进一步增加
        val_batch_size = 16  # 增加验证batch size
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

# === 2. 训练模型 (多GPU并行版本) ===
def train_model(labeled_data):
    # 保存临时训练集
    with open('./data/al_labeled.pkl', 'wb') as f:
        pickle.dump(labeled_data, f)
    args = get_args()
    args.dataset = 'al_tmp_train'
    args.no_cuda = False  # 使用GPU
    
    # 检查GPU数量
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"检测到 {num_gpus} 块GPU，将使用多GPU并行训练")
        # 为多GPU调整batch size
        args.batch_size = args.batch_size * num_gpus  # 增加batch size以充分利用多GPU
        args.val_batch_size = args.val_batch_size * num_gpus
    else:
        print(f"检测到 {num_gpus} 块GPU，使用单GPU训练")
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    get_seed(args.seed)
    train_dataset, _ = get_dataset(args)
    train_loader = MIODataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = get_model(args).to(device)
    
    # 多GPU并行
    if num_gpus > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(num_gpus)))
        print(f"模型已设置为使用GPU: {list(range(num_gpus))}")
    
    loss_func = get_loss_func(args.loss_name, args, regularizer=True, normalizer=train_dataset.y_normalizer)
    metric_func = get_loss_func('rel2', args, regularizer=False, normalizer=train_dataset.y_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    train(model, loss_func, metric_func, train_loader, train_loader, optimizer, scheduler, epochs=args.epochs, device=device)
    return model, metric_func, device

# === 3. 评估模型 (支持多GPU) ===
def evaluate_model(model_tuple, test_data):
    model, metric_func, device = model_tuple
    
    # 如果是DataParallel模型，获取原始模型用于评估
    if isinstance(model, torch.nn.DataParallel):
        eval_model = model.module
        print("使用多GPU模型进行评估")
    else:
        eval_model = model
    
    # 保存临时测试集
    with open('./data/al_test.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    args = get_args()
    args.dataset = 'al_tmp_train'
    _, test_dataset = get_dataset(args)
    test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)
    val_result = validate_epoch(eval_model, metric_func, test_loader, device)
    metric = val_result["metric"]
    return metric

# === 4. 主动学习主循环 ===
def random_active_learning(
    labeled_path, unlabeled_path, test_path, rounds=5, select_num=5, seed=42, output_dir='./al_rounds'
):
    # 显示GPU信息
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    with open(labeled_path, 'rb') as f:
        labeled_data = pickle.load(f)
    with open(unlabeled_path, 'rb') as f:
        unlabeled_data = pickle.load(f)
    with open(test_path, 'rb') as f:
        test_data = pickle.load(f)

    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    # 新增：准备CSV文件
    csv_path = os.path.join(output_dir, 'metrics.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['pressure', 'wall-shear', 'x-wall-shear', 'y-wall-shear', 'z-wall-shear', 'train_num', 'round'])

    for r in range(rounds):
        print(f"\n=== 主动学习第{r+1}轮 ===")
        # 训练模型
        model_tuple = train_model(labeled_data)
        # 评估模型
        metric = evaluate_model(model_tuple, test_data)
        print(f"第{r+1}轮模型性能: {metric}")

        # 新增：写入CSV
        train_num = len(labeled_data)
        # metric 需为长度5的list/array
        if isinstance(metric, (np.ndarray, list)) and len(metric) == 5:
            row = list(metric) + [train_num, r+1]
        else:
            # 若metric为单值，补齐为5列
            row = [metric]*5 + [train_num, r+1]
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(row)

        # 随机采样select_num个未标注样本
        if len(unlabeled_data) < select_num:
            select_num = len(unlabeled_data)
        newly_selected = random.sample(unlabeled_data, select_num)
        labeled_data.extend(newly_selected)
        newly_selected_ids = set(id(x) for x in newly_selected)
        unlabeled_data = [x for x in unlabeled_data if id(x) not in newly_selected_ids]

        # 保存每轮数据快照
        with open(os.path.join(output_dir, f'al_labeled_round{r+1}.pkl'), 'wb') as f:
            pickle.dump(labeled_data, f)
        with open(os.path.join(output_dir, f'al_unlabeled_round{r+1}.pkl'), 'wb') as f:
            pickle.dump(unlabeled_data, f)

        print(f"本轮新增样本: {len(newly_selected)}，累计已标注: {len(labeled_data)}，未标注: {len(unlabeled_data)}")

        if len(unlabeled_data) == 0:
            print("未标注池已空，主动学习提前结束。")
            break

    print("主动学习流程结束。")

# === 5. 启动 ===
if __name__ == "__main__":
    # 设置GPU环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'  # 确保所有4块GPU都可见
    
    # 先准备主动学习初始数据
    prepare_active_learning_data(
        data_dir='/home/v-wenliao/gnot/GNOT/data/result',
        output_dir='/home/v-wenliao/gnot/GNOT//data/',
        init_labeled_num=100,   # 你想要的初始训练集数量
        test_ratio=0.05,
        random_seed=42,
        overwrite=True  # 设置为True即可覆盖
    )
    # 然后启动主动学习主循环
    random_active_learning(
        labeled_path='/home/v-wenliao/gnot/GNOT//data/al_labeled.pkl',
        unlabeled_path='/home/v-wenliao/gnot/GNOT//data/al_unlabeled.pkl',
        test_path='/home/v-wenliao/gnot/GNOT//data/al_test.pkl',
        rounds=28,
        select_num=100,
        seed=42,
        output_dir='/home/v-wenliao/gnot/GNOT//data/al_rounds'
    )