#!/usr/bin/env python3
"""
直接测试单个样本误差计算
"""

import os
import sys
import pickle
import numpy as np
import torch

# 激活conda环境
print("正在激活conda环境gnot_cuda11...")
os.system("source ~/.bashrc && conda activate gnot_cuda11")

# 设置路径
sys.path.append('/home/v-wenliao/gnot/GNOT')

def test_sample_error_calculation():
    """测试单个样本的误差计算"""
    
    print("="*60)
    print("测试单个样本误差计算")
    print("="*60)
    
    try:
        print("1. 导入必要模块...")
        from alpa import get_al_args  # 从alpa.py导入
        from data_utils import get_dataset, get_model, get_loss_func, MIODataLoader
        
        print("2. 加载未标注数据...")
        data_file = '/home/v-wenliao/gnot/GNOT/data/al_bz/al_unlabeled.pkl'
        if not os.path.exists(data_file):
            data_file = '/home/v-wenliao/gnot/GNOT/data/al_unlabeled.pkl'
        
        if not os.path.exists(data_file):
            print("数据文件不存在！")
            return
            
        with open(data_file, 'rb') as f:
            unlabeled_data = pickle.load(f)
        
        print(f"加载了 {len(unlabeled_data)} 个未标注样本")
        
        print("3. 设置参数...")
        args = get_al_args()
        args.dataset = 'al_bz'
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        print("4. 加载数据集和模型...")
        # 先加载数据集来设置dataset_config
        train_dataset, test_dataset = get_dataset(args)
        
        model = get_model(args)
        model = model.to(device)
        model.eval()
        
        print(f"模型加载成功")
        
        print("5. 测试前5个样本的误差计算...")
        sample_errors = []
        
        for i in range(min(5, len(unlabeled_data))):
            print(f"\n--- 样本 {i} ---")
            sample = unlabeled_data[i]
            
            try:
                # 创建临时数据文件
                temp_dir = '/home/v-wenliao/gnot/GNOT/data/al_bz/data'
                os.makedirs(temp_dir, exist_ok=True)
                temp_file = os.path.join(temp_dir, f'temp_sample_{i}.pkl')
                
                with open(temp_file, 'wb') as f:
                    pickle.dump([sample], f)
                
                # 复制为测试文件
                test_file = os.path.join(temp_dir, 'al_test.pkl')
                import shutil
                shutil.copy2(temp_file, test_file)
                
                print(f"  临时文件创建成功")
                
                # 加载数据集
                _, test_dataset = get_dataset(args)
                test_loader = MIODataLoader(test_dataset, batch_size=1, shuffle=False)
                
                print(f"  数据集加载成功，数据点数: {len(test_dataset)}")
                
                # 模型预测
                with torch.no_grad():
                    for batch_data in test_loader:
                        g, u_p, g_u = batch_data
                        
                        # 移动到设备
                        g = g.to(device)
                        u_p = u_p.to(device)
                        if hasattr(g_u, 'to'):
                            g_u = g_u.to(device)
                        
                        # 预测
                        pred = model(g, u_p, g_u)
                        
                        # 获取真实值
                        if hasattr(g, 'ndata') and 'y' in g.ndata:
                            target = g.ndata['y']
                        else:
                            target = torch.tensor(np.array(sample[1]), dtype=torch.float32).to(device)
                        
                        print(f"  预测形状: {pred.shape}")
                        print(f"  真实值形状: {target.shape}")
                        
                        # 计算MSE误差
                        if pred.shape != target.shape:
                            min_size = min(pred.size(0), target.size(0))
                            min_dims = min(pred.size(1) if len(pred.shape) > 1 else 1, 
                                          target.size(1) if len(target.shape) > 1 else 1)
                            pred = pred[:min_size, :min_dims] if len(pred.shape) > 1 else pred[:min_size]
                            target = target[:min_size, :min_dims] if len(target.shape) > 1 else target[:min_size]
                            print(f"  调整后形状: pred {pred.shape}, target {target.shape}")
                        
                        # 计算误差
                        mse_error = torch.mean((pred - target) ** 2).item()
                        l2_error = torch.norm(pred - target).item()
                        mae_error = torch.mean(torch.abs(pred - target)).item()
                        
                        print(f"  MSE误差: {mse_error:.6f}")
                        print(f"  L2误差: {l2_error:.6f}")
                        print(f"  MAE误差: {mae_error:.6f}")
                        
                        sample_errors.append({
                            'sample_id': i,
                            'mse': mse_error,
                            'l2': l2_error,
                            'mae': mae_error,
                            'pred_shape': pred.shape,
                            'target_shape': target.shape
                        })
                        
                        break  # 只处理第一个batch
                
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                if os.path.exists(test_file):
                    os.remove(test_file)
                    
            except Exception as e:
                print(f"  样本 {i} 处理失败: {e}")
                import traceback
                print(f"  详细错误: {traceback.format_exc()}")
        
        print(f"\n" + "="*60)
        print("误差计算结果总结:")
        print(f"总共处理了 {len(sample_errors)} 个样本")
        
        if sample_errors:
            print("\n各样本误差:")
            for error_info in sample_errors:
                print(f"样本 {error_info['sample_id']}: MSE={error_info['mse']:.6f}, "
                      f"L2={error_info['l2']:.6f}, MAE={error_info['mae']:.6f}")
            
            mse_values = [e['mse'] for e in sample_errors]
            print(f"\nMSE统计:")
            print(f"  最小值: {min(mse_values):.6f}")
            print(f"  最大值: {max(mse_values):.6f}")
            print(f"  平均值: {np.mean(mse_values):.6f}")
            print(f"  标准差: {np.std(mse_values):.6f}")
            
            if np.std(mse_values) > 1e-6:
                print("✅ 不同样本有不同的误差值 - 可以用于主动学习!")
            else:
                print("❌ 所有样本的误差值相同 - 需要进一步调试")
        else:
            print("❌ 没有成功计算任何样本的误差")
            
    except Exception as e:
        print(f"脚本执行失败: {e}")
        import traceback
        print(f"详细错误: {traceback.format_exc()}")


if __name__ == "__main__":
    test_sample_error_calculation()
