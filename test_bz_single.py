#!/usr/bin/env python3
"""
测试BZ策略是否能正确计算单个样本的预测误差
"""

import os
import sys
import pickle
import torch
import numpy as np
from tqdm import tqdm

# 导入必要的模块
sys.path.append('/home/v-wenliao/gnot/GNOT')
from alpa import bz_query
from args import get_args as get_original_args  
from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
from utils import get_seed, MultipleTensors

def test_bz_single_sample():
    """测试BZ策略是否能处理单个样本"""
    print("开始测试BZ策略单样本处理...")
    
    # 1. 载入测试数据
    data_dir = "/home/v-wenliao/gnot/GNOT/data"
    
    with open(f"{data_dir}/al_labeled.pkl", 'rb') as f:
        labeled_data = pickle.load(f)
    with open(f"{data_dir}/al_unlabeled.pkl", 'rb') as f:
        unlabeled_data = pickle.load(f)
    
    print(f"已标注数据: {len(labeled_data)} 个样本")
    print(f"未标注数据: {len(unlabeled_data)} 个样本")
    
    # 2. 准备模型参数
    args = get_original_args()
    args.data_name = 'result'
    args.data_dir = '/home/v-wenliao/gnot/GNOT/data'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 3. 创建数据集和模型
    train_data = get_dataset(args)
    model = get_model(args)
    device = torch.device(args.device)
    model = model.to(device)
    
    # 4. 模型设为评估模式
    model.eval()
    
    # 5. 测试前3个未标注样本
    test_samples = unlabeled_data[:3]
    
    print(f"\n测试前{len(test_samples)}个未标注样本:")
    
    try:
        # 直接调用bz_query函数
        selected_indices = bz_query(
            model=model,
            unlabeled_data=test_samples,
            device=device,
            select_num=2,  # 选择2个样本
            args=args
        )
        
        print(f"✓ BZ策略成功完成!")
        print(f"选中的样本索引: {selected_indices}")
        print(f"选中样本数量: {len(selected_indices)}")
        return True
        
    except Exception as e:
        print(f"✗ BZ策略失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_bz_single_sample()
    if success:
        print("\n🎉 BZ策略测试成功! 模型可以正确处理单个样本的预测误差计算。")
    else:
        print("\n❌ BZ策略测试失败，需要进一步调试。")
