#!/usr/bin/env python3
"""
BZ策略维度缩放使用示例

演示如何在主动学习中使用改进的BZ策略
"""

import os
import sys

# 添加路径
sys.path.append('/home/v-wenliao/gnot/GNOT')

def run_bz_scaled_active_learning():
    """运行使用缩放BZ策略的主动学习"""
    
    print("=== 运行缩放BZ策略主动学习 ===")
    
    # 设置策略
    strategies_to_test = [
        'bz_scaled',    # 自适应维度缩放BZ策略
        'bz_manual',    # 手动权重BZ策略  
        'bz'            # 原始BZ策略（对比）
    ]
    
    for strategy in strategies_to_test:
        print(f"\n--- 测试策略: {strategy} ---")
        
        # 修改albz.py中的策略设置
        albz_file = '/home/v-wenliao/gnot/GNOT/albz.py'
        
        # 读取文件
        with open(albz_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 查找并替换策略设置行
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'strategy = ' in line and 'bz' in line and not line.strip().startswith('#'):
                # 找到当前策略设置行，替换它
                lines[i] = f"    strategy = '{strategy}'  # 当前测试策略"
                break
        
        # 写回文件
        with open(albz_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"已设置策略为: {strategy}")
        
        # 运行主动学习（这里只做演示，实际运行需要数据）
        print(f"运行命令: python albz.py")
        print(f"策略说明:")
        
        if strategy == 'bz_scaled':
            print("  - 自适应计算各维度的缩放因子")
            print("  - 根据数据分布自动平衡各维度贡献")
            print("  - 推荐用于生产环境")
            
        elif strategy == 'bz_manual':
            print("  - 使用手动设置的维度权重")
            print("  - pressure权重设为0.1，其他维度权重为1.0")
            print("  - 适合已知特定维度主导的情况")
            
        elif strategy == 'bz':
            print("  - 原始BZ策略，各维度误差直接求和")
            print("  - 通常由数值最大的维度主导")
            print("  - 用于对比效果")
        
        print(f"结果将保存在: /home/v-wenliao/gnot/GNOT/data/al_bz/al_rounds*")
        print(f"日志文件: /home/v-wenliao/gnot/GNOT/data/al_bz/al_rounds*/round_*/round_log.txt")


def create_quick_test():
    """创建快速测试脚本"""
    
    test_script = """#!/usr/bin/env python3
'''
快速测试BZ策略缩放效果
'''

import sys
sys.path.append('/home/v-wenliao/gnot/GNOT')

from albz import random_active_learning_with_logging

def test_strategy(strategy_name, rounds=2, select_num=10):
    '''测试指定策略'''
    print(f"测试策略: {strategy_name}")
    
    # 设置参数
    data_update_dir = "/home/v-wenliao/gnot/GNOT/data/al_bz"
    output_dir = f"/home/v-wenliao/gnot/GNOT/data/al_bz/test_{strategy_name}"
    
    try:
        # 运行主动学习
        random_active_learning_with_logging(
            rounds=rounds,
            select_num=select_num,
            seed=42,
            output_dir=output_dir,
            data_update_dir=data_update_dir,
            strategy=strategy_name
        )
        print(f"策略 {strategy_name} 测试完成")
        return True
        
    except Exception as e:
        print(f"策略 {strategy_name} 测试失败: {e}")
        return False

if __name__ == "__main__":
    # 测试三种策略
    strategies = ['bz_scaled', 'bz_manual', 'bz']
    
    for strategy in strategies:
        print(f"\\n{'='*50}")
        success = test_strategy(strategy, rounds=1, select_num=5)
        if success:
            print(f"✓ {strategy} 测试成功")
        else:
            print(f"✗ {strategy} 测试失败")
"""
    
    with open('/home/v-wenliao/gnot/GNOT/quick_test_bz.py', 'w', encoding='utf-8') as f:
        f.write(test_script)
    
    print("已创建快速测试脚本: quick_test_bz.py")


def main():
    """主函数"""
    print("BZ策略维度缩放使用指南")
    print("=" * 50)
    
    print("\n=== 问题分析 ===")
    print("从您提供的日志可以看出:")
    print("- 维度 0 (pressure): 误差 = 281.529999")
    print("- 维度 1 (wall-shear): 误差 = 1.273253")
    print("- 维度 2-4: 误差在 0.2-0.5 范围")
    print("- pressure误差比其他维度大200-1000倍")
    print("- 导致BZ策略完全由pressure维度主导")
    
    print("\n=== 解决方案 ===")
    print("1. 自适应缩放 (推荐): 'bz_scaled'")
    print("   - 根据数据分布自动计算缩放因子")
    print("   - 平衡各维度的贡献")
    
    print("2. 手动权重: 'bz_manual'")
    print("   - 手动设置各维度权重")
    print("   - 当前设置: [0.1, 1.0, 1.0, 1.0, 1.0]")
    
    print("3. 等权重: 在代码中可使用 scaling_method='equal'")
    
    print("\n=== 使用方法 ===")
    print("1. 直接运行: python albz.py")
    print("   (已将默认策略改为 'bz_scaled')")
    
    print("2. 修改策略: 编辑 albz.py 中的 strategy 变量")
    print("   strategy = 'bz_scaled'   # 自适应缩放")
    print("   strategy = 'bz_manual'   # 手动权重")
    print("   strategy = 'bz'          # 原始策略")
    
    print("3. 自定义权重: 修改 bz_strategy_scale_fix.py 中的 manual_weights")
    
    print("\n=== 预期效果 ===")
    print("- bz_scaled: pressure占比从99.1%降至约20%")
    print("- bz_manual: pressure占比降至约91.8%")
    print("- 各维度误差更平衡地影响样本选择")
    print("- 可能选择出在其他物理量上有更大误差的样本")
    
    # 创建使用示例
    create_quick_test()
    
    print(f"\n=== 下一步 ===")
    print("1. 运行: python albz.py (使用缩放策略)")
    print("2. 对比: 查看选择的样本是否更多样化")
    print("3. 分析: 检查round_log.txt中的维度误差分布")
    print("4. 快速测试: python quick_test_bz.py")


if __name__ == "__main__":
    main()
