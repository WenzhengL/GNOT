#!/usr/bin/env python3
"""
测试修改后的BZ策略详细日志功能
"""

import sys
import os
import pickle
import numpy as np
sys.path.append('/home/v-wenliao/gnot/GNOT')

def test_bz_detailed_logging():
    """测试BZ策略的详细日志记录功能"""
    print("测试BZ策略详细日志记录...")
    
    # 导入必要模块
    from alpa import bz_query
    from args import get_args as get_original_args
    from data_utils import get_model
    
    # 1. 准备少量测试数据
    print("1. 准备测试数据...")
    data_dir = "/home/v-wenliao/gnot/GNOT/data"
    
    with open(f"{data_dir}/al_unlabeled.pkl", 'rb') as f:
        all_unlabeled = pickle.load(f)
    
    # 只取前5个样本进行测试
    test_samples = all_unlabeled[:5]
    print(f"   使用 {len(test_samples)} 个样本进行测试")
    
    # 2. 创建模型
    print("2. 创建模型...")
    args = get_original_args()
    args.data_name = 'result'
    args.data_dir = '/home/v-wenliao/gnot/GNOT/data'
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = get_model(args)
    device = torch.device(args.device)
    model = model.to(device)
    model.eval()
    
    # 3. 测试BZ策略的详细日志
    print("3. 运行BZ策略（详细日志模式）...")
    print("="*60)
    
    try:
        selected_indices = bz_query(
            model=model,
            unlabeled_data=test_samples,
            device=device,
            select_num=3,  # 选择3个样本
            args=args
        )
        
        print("="*60)
        print(f"✓ BZ策略测试成功!")
        print(f"选中的样本索引: {selected_indices}")
        print(f"选中样本数量: {len(selected_indices)}")
        return True
        
    except Exception as e:
        print(f"✗ BZ策略测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # 导入torch
    import torch
    
    success = test_bz_detailed_logging()
    if success:
        print("\n🎉 BZ策略详细日志测试成功!")
        print("现在每个样本的预测误差都会详细记录到日志中")
    else:
        print("\n❌ BZ策略详细日志测试失败")
