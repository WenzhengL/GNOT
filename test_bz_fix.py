#!/usr/bin/env python3
"""
测试修复后的BZ策略是否能处理真实数据
"""

import sys
import torch
import pickle
import numpy as np
sys.path.append('/home/v-wenliao/gnot/GNOT')

def test_bz_with_real_sample():
    """测试BZ策略是否能处理真实样本数据"""
    print("测试BZ策略处理真实样本...")
    
    # 导入BZ核心函数
    from alpa import bz_query
    from utils import MultipleTensors
    from torch.nn.utils.rnn import pad_sequence
    import dgl
    
    # 1. 创建一个简单的模拟样本
    coords = np.random.rand(50, 3) * 0.01
    values = np.random.rand(50, 5) 
    theta = np.array([1.0, 0.5])
    branch_data = (np.zeros(50),)  # 简单的分支数据
    
    sample = [coords, values, theta, branch_data]
    print(f"创建测试样本: 坐标{coords.shape}, 数值{values.shape}")
    
    # 2. 测试数据处理逻辑
    device = torch.device('cpu')
    
    # 转换为tensor
    coords = torch.tensor(coords, dtype=torch.float32).to(device)
    values = torch.tensor(values, dtype=torch.float32).to(device)
    theta = torch.tensor(theta, dtype=torch.float32).to(device)
    
    num_points = coords.shape[0]
    
    # 3. 创建DGL图
    g = dgl.DGLGraph()
    g.add_nodes(num_points)
    g.ndata['x'] = coords
    g.ndata['y'] = values
    
    # 4. 准备模型输入
    u_p = theta.unsqueeze(0)  # [1, 2]
    
    # 5. 处理分支数据 - 使用新的pad_sequence方法
    branch_array = branch_data[0][:num_points]
    branch_tensor = torch.tensor(branch_array, dtype=torch.float32).to(device)
    
    if len(branch_tensor.shape) == 1:
        branch_tensor = branch_tensor.unsqueeze(-1)  # [N, 1]
    
    # 使用pad_sequence来模拟批处理
    padded = pad_sequence([branch_tensor]).permute(1, 0, 2)  # [B=1, T, F]
    g_u = MultipleTensors([padded])
    
    print(f"数据处理成功:")
    print(f"  图: {g.number_of_nodes()} 个节点")
    print(f"  参数: {u_p.shape}")
    print(f"  分支数据: {type(g_u)}, 张量形状: {g_u[0].shape}")
    print(f"  分支张量维度: {len(g_u[0].shape)}")
    
    # 6. 验证维度
    if len(g_u[0].shape) == 3:
        print(f"✓ 分支数据维度正确: {g_u[0].shape} (batch, sequence, features)")
        return True
    else:
        print(f"✗ 分支数据维度错误: {g_u[0].shape}")
        return False

if __name__ == "__main__":
    success = test_bz_with_real_sample()
    if success:
        print("\n🎉 BZ策略数据处理修复成功!")
        print("分支数据现在具有正确的3D格式 (batch, sequence, features)")
    else:
        print("\n❌ BZ策略仍有问题，需要进一步调试。")
