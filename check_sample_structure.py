#!/usr/bin/env python3
"""
检查样本数据结构
"""

import os
import sys
import pickle
import numpy as np

sys.path.append('/home/v-wenliao/gnot/GNOT')

def check_sample_structure():
    """检查样本数据的实际结构"""
    
    print("="*60)
    print("检查样本数据结构")
    print("="*60)
    
    # 加载未标注数据
    data_file = '/home/v-wenliao/gnot/GNOT/data/al_bz/al_unlabeled.pkl'
    if not os.path.exists(data_file):
        data_file = '/home/v-wenliao/gnot/GNOT/data/al_unlabeled.pkl'
    
    with open(data_file, 'rb') as f:
        unlabeled_data = pickle.load(f)
    
    print(f"加载了 {len(unlabeled_data)} 个未标注样本")
    
    # 检查前3个样本的结构
    for i in range(min(3, len(unlabeled_data))):
        sample = unlabeled_data[i]
        print(f"\n样本 {i}:")
        print(f"  类型: {type(sample)}")
        print(f"  长度: {len(sample)}")
        
        for j, item in enumerate(sample):
            print(f"  元素 {j}:")
            print(f"    类型: {type(item)}")
            if isinstance(item, (list, tuple, np.ndarray)):
                print(f"    形状: {np.array(item).shape if hasattr(item, 'shape') else len(item)}")
                print(f"    内容预览: {str(item)[:100]}...")
            elif isinstance(item, dict):
                print(f"    字典键: {list(item.keys())}")
            else:
                print(f"    内容: {item}")

if __name__ == "__main__":
    check_sample_structure()
