import os
import pickle
import random
import csv
import json
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

# === 1. 导入你的GNOT训练和评估相关函数 ===
from train import train, validate_epoch
from args import get_args as get_original_args
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from utils import get_seed


# 多GPU训练的辅助函数
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_model_ddp(labeled_data, rank, world_size):
    """使用DistributedDataParallel的多GPU训练函数"""
    setup(rank, world_size)
    
    # 保存临时训练集
    with open('./data/al_labeled.pkl', 'wb') as f:
        pickle.dump(labeled_data, f)
    
    args = get_al_args()
    args.dataset = 'al_tmp_train'
    device = torch.device(f'cuda:{rank}')
    
    get_seed(args.seed)
    train_dataset, _ = get_dataset(args)
    
    # 创建分布式采样器
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank
    )
    
    train_loader = MIODataLoader(
        train_dataset, 
        batch_size=args.batch_size//world_size,  # 调整批处理大小
        shuffle=False,  # 使用分布式采样器时不要shuffle
        sampler=train_sampler,
        drop_last=False
    )
    
    model = get_model(args).to(device)
    model = DDP(model, device_ids=[rank])
    
    loss_func = get_loss_func(args.loss_name, args, regularizer=True, normalizer=train_dataset.y_normalizer)
    metric_func = get_loss_func('rel2', args, regularizer=False, normalizer=train_dataset.y_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    
    # 自定义训练循环以处理DGL图
    model.train()
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for batch in train_loader:
            optimizer.zero_grad()
            
            g, u_p, g_u = batch
            # 确保所有数据都在正确的设备上
            g = g.to(device)
            u_p = u_p.to(device)
            g_u = g_u.to(device)
            
            out = model(g, u_p, g_u)
            y_pred, y = out.squeeze(), g.ndata['y'].squeeze()
            
            loss, reg, _ = loss_func(g, y_pred, y)
            loss = loss + reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
    
    cleanup()
    
    # 只在rank 0上返回模型
    if rank == 0:
        return model.module, metric_func, device
    else:
        return None, None, None

def get_al_args():
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
        # 训练相关 - 多GPU优化
        epochs = 100  # 减少epochs以加快测试
        optimizer = 'AdamW'
        lr = 1e-3
        weight_decay = 5e-6
        grad_clip = 1000.0
        batch_size = 32  # 总batch size，会被world_size分割
        val_batch_size = 16
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

def train_model_multi_gpu(labeled_data):
    """多GPU训练的包装函数"""
    world_size = torch.cuda.device_count()
    if world_size <= 1:
        # 回退到单GPU训练
        return train_model_single_gpu(labeled_data)
    
    # 使用multiprocessing启动多GPU训练
    mp.spawn(train_model_ddp, args=(labeled_data, world_size), nprocs=world_size, join=True)
    
    # 加载训练好的模型（这里需要一些额外的逻辑来保存和加载模型）
    # 为简化，这里暂时返回单GPU训练的结果
    return train_model_single_gpu(labeled_data)

def train_model_single_gpu(labeled_data):
    """单GPU训练函数（原来的逻辑）"""
    # 保存临时训练集
    with open('./data/al_labeled.pkl', 'wb') as f:
        pickle.dump(labeled_data, f)
    args = get_al_args()
    args.dataset = 'al_tmp_train'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    get_seed(args.seed)
    train_dataset, _ = get_dataset(args)
    train_loader = MIODataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    model = get_model(args).to(device)
    
    loss_func = get_loss_func(args.loss_name, args, regularizer=True, normalizer=train_dataset.y_normalizer)
    metric_func = get_loss_func('rel2', args, regularizer=False, normalizer=train_dataset.y_normalizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = None
    
    train(model, loss_func, metric_func, train_loader, train_loader, optimizer, scheduler, epochs=args.epochs, device=device)
    return model, metric_func, device

# 使用示例：
# if __name__ == "__main__":
#     # 设置多GPU训练
#     labeled_data = [...]  # 你的标注数据
#     model_tuple = train_model_multi_gpu(labeled_data)
