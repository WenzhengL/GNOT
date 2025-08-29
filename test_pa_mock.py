#!/usr/bin/env python3
"""
测试修改后的PA策略

这个脚本用于测试修改后的PA策略是否能正确处理误差为0的情况
"""

import os
import sys
import numpy as np
import pickle
import json

# 添加当前目录到路径
sys.path.append('/home/v-wenliao/gnot/GNOT')

def create_test_data():
    """创建测试数据"""
    
    print("创建测试数据...")
    
    # 创建测试目录
    test_dir = '/home/v-wenliao/gnot/GNOT/data/al_pa_test_simple'
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建简单的测试数据
    test_data = []
    
    for i in range(10):  # 创建10个样本
        # 坐标数据 [N, 3]
        n_points = 50 + i * 5  # 不同样本有不同的点数
        coords = np.random.rand(n_points, 3) * 0.01  # 小范围坐标
        
        # 真实值数据 [N, 5] - pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear
        pressure = np.random.rand(n_points) * 50 + 10
        wall_shear = np.random.rand(n_points) * 5 + 1
        x_wall_shear = np.random.rand(n_points) * 2 - 1
        y_wall_shear = np.random.rand(n_points) * 2 - 1  
        z_wall_shear = np.random.rand(n_points) * 2 - 1
        
        values = np.column_stack([pressure, wall_shear, x_wall_shear, y_wall_shear, z_wall_shear])
        
        # 参数数据
        theta = np.array([1.0 + i * 0.1, 0.5 + i * 0.05])  # 两个参数
        
        # 分支数据
        branch_data = (np.zeros((n_points, 1)),)  # 空分支
        
        sample = [coords, values, theta, branch_data]
        test_data.append(sample)
    
    # 保存测试数据
    labeled_data = test_data[:3]  # 前3个作为已标注
    unlabeled_data = test_data[3:8]  # 中间5个作为未标注
    test_data_final = test_data[8:]  # 最后2个作为测试
    
    with open(os.path.join(test_dir, 'al_labeled.pkl'), 'wb') as f:
        pickle.dump(labeled_data, f)
    
    with open(os.path.join(test_dir, 'al_unlabeled.pkl'), 'wb') as f:
        pickle.dump(unlabeled_data, f)
    
    with open(os.path.join(test_dir, 'al_test.pkl'), 'wb') as f:
        pickle.dump(test_data_final, f)
    
    # 创建数据集信息文件
    dataset_info = {
        "name": "al_pa_test_simple",
        "total_samples": len(test_data),
        "labeled_samples": len(labeled_data),
        "unlabeled_samples": len(unlabeled_data),
        "test_samples": len(test_data_final)
    }
    
    with open(os.path.join(test_dir, 'dataset_info.json'), 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"测试数据创建完成:")
    print(f"  总样本: {len(test_data)}")
    print(f"  已标注: {len(labeled_data)}")
    print(f"  未标注: {len(unlabeled_data)}")
    print(f"  测试集: {len(test_data_final)}")
    print(f"  保存位置: {test_dir}")
    
    return test_dir, labeled_data, unlabeled_data


def test_pa_with_mock_model():
    """使用模拟模型测试PA策略"""
    
    print("\n测试PA策略（使用模拟模型）...")
    
    test_dir, labeled_data, unlabeled_data = create_test_data()
    
    # 模拟模型预测
    class MockModel:
        def __init__(self, perfect_prediction=True):
            self.perfect_prediction = perfect_prediction
            
        def eval(self):
            pass
            
        def to(self, device):
            return self
            
        def __call__(self, g, u_p, g_u):
            # 模拟预测结果
            # 这里我们需要返回一个torch张量形式的结果
            import torch
            
            # 从真实数据中获取形状信息
            sample = unlabeled_data[0]  # 获取第一个样本
            true_values = np.array(sample[1])
            
            if self.perfect_prediction:
                # 完美预测（误差为0的情况）
                pred = torch.tensor(true_values, dtype=torch.float32)
            else:
                # 有误差的预测
                noise = np.random.normal(0, 0.1, true_values.shape)
                pred = torch.tensor(true_values + noise, dtype=torch.float32)
            
            return pred
    
    # 测试完美预测的情况
    print("\n1. 测试完美预测（误差为0）的情况...")
    
    try:
        from alpa import pa_query_average_error
        
        perfect_model = MockModel(perfect_prediction=True)
        mock_metric_func = lambda x: 0.0
        device = 'cpu'
        
        model_tuple = (perfect_model, mock_metric_func, device)
        
        # 调用PA策略
        selected_idx = pa_query_average_error(model_tuple, unlabeled_data[:3], 2)  # 从前3个中选2个
        
        print(f"完美预测情况下选择的样本索引: {selected_idx}")
        
    except Exception as e:
        print(f"完美预测测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")
    
    # 测试有误差预测的情况
    print("\n2. 测试有误差预测的情况...")
    
    try:
        imperfect_model = MockModel(perfect_prediction=False)
        model_tuple = (imperfect_model, mock_metric_func, device)
        
        selected_idx = pa_query_average_error(model_tuple, unlabeled_data[:3], 2)  # 从前3个中选2个
        
        print(f"有误差预测情况下选择的样本索引: {selected_idx}")
        
    except Exception as e:
        print(f"有误差预测测试失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":
    print("开始测试修改后的PA策略...")
    
    test_pa_with_mock_model()
    
    print("\n测试完成!")
    
    print("""
测试总结:
1. 如果测试成功，说明修改后的PA策略能正确处理误差为0的情况
2. 在误差为0时，策略会使用基于数据变异性的选择方法
3. 如果数据变异性也很小，则使用随机选择
4. 这样避免了回退到几何复杂度评分

下一步:
- 如果测试成功，可以重新运行主动学习流程
- 观察是否还会出现"所有预测误差都无效"的警告
- 策略应该能够正常选择样本
""")
