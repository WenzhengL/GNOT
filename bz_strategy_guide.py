#!/usr/bin/env python3
"""
BZ策略使用示例 - 完整的主动学习流程

这是基于GNOT框架的最佳主动学习策略，直接使用train.py中的validate_epoch函数
"""

import os
import sys

# 添加项目路径
sys.path.append('/home/v-wenliao/gnot/GNOT')

def run_bz_strategy_example():
    """
    运行BZ策略的完整示例
    """
    
    print("="*80)
    print("BZ策略 - 基于GNOT框架的主动学习策略")
    print("="*80)
    
    print("""
BZ策略特点:
-----------
✓ 直接使用GNOT的validate_epoch函数评估样本
✓ 选择模型表现最差（metric最大）的样本
✓ 完美集成到GNOT框架，无需额外配置
✓ 自动适应不同的损失函数和数据集

策略原理:
---------
1. 对每个未标注样本，使用训练好的模型进行评估
2. 通过validate_epoch函数计算每个样本的metric值
3. 选择metric最大的样本（即模型表现最差的样本）
4. 这些样本最需要标注来改进模型性能

适用场景:
---------
• 任何GNOT支持的数据集和问题
• 需要高效主动学习的场景
• 希望直接基于模型表现选择样本的情况
""")
    
    print("\n" + "="*50)
    print("使用方法")
    print("="*50)
    
    # 显示使用代码
    example_code = '''
# 导入主动学习模块
from alpa import active_learning_loop

# 方法1: 最简单的使用方式
results = active_learning_loop(
    dataset_name='your_dataset',
    strategy='bz',              # 使用BZ策略
    rounds=10,                  # 10轮主动学习
    select_num=5               # 每轮选择5个样本
)

# 方法2: 完整配置
results = active_learning_loop(
    dataset_name='your_dataset',
    strategy='bz',
    rounds=10,
    select_num=5,
    initial_labeled_ratio=0.1,  # 初始标注比例
    use_normalization=True,     # 使用数据标准化
    save_path='./data/al_bz'   # 结果保存路径
)

# 方法3: 批量比较不同策略
strategies = ['bz', 'pa', 'qbc', 'gv', 'rd']
for strategy in strategies:
    print(f"\\n测试策略: {strategy}")
    results = active_learning_loop(
        dataset_name='your_dataset',
        strategy=strategy,
        rounds=5,
        select_num=3
    )
    print(f"{strategy}策略完成")
'''
    
    print(example_code)
    
    print("\n" + "="*50)
    print("实际运行示例")
    print("="*50)
    
    # 实际可运行的示例
    print("""
要运行BZ策略，执行以下命令:

# 进入GNOT目录
cd /home/v-wenliao/gnot/GNOT

# 运行主动学习（假设你有数据集）
python -c "
from alpa import active_learning_loop

# 运行BZ策略
results = active_learning_loop(
    dataset_name='your_dataset',  # 替换为你的数据集名称
    strategy='bz',
    rounds=3,
    select_num=2
)
print('BZ策略运行完成!')
"

或者使用已有的脚本:
python albz.py  # 如果存在专门的BZ策略脚本
""")
    
    print("\n" + "="*50)
    print("策略对比")
    print("="*50)
    
    comparison = """
策略名称    |  复杂度  |  GNOT集成  |  效果评估
-----------|----------|------------|----------
BZ         |  简单    |    完美    |   优秀
PA (旧)    |  复杂    |    一般    |   中等
PA (新)    |  中等    |    良好    |   良好
QBC        |  中等    |    一般    |   中等
GV         |  简单    |    一般    |   中等
RD         |  简单    |    完美    |   基准

推荐使用: BZ策略
理由:
1. 与GNOT框架集成度最高
2. 直接基于模型表现选择样本
3. 代码简单，维护容易
4. 适应性强，效果稳定
"""
    
    print(comparison)
    
    print("\n" + "="*50)
    print("常见问题")
    print("="*50)
    
    faq = """
Q: BZ策略和其他策略的主要区别是什么？
A: BZ策略直接使用GNOT的validate_epoch函数，其他策略需要自己计算误差

Q: BZ策略选择的是什么样的样本？
A: 选择模型表现最差（metric最大）的样本，这些样本最需要标注

Q: BZ策略是否需要调参？
A: 不需要，BZ策略自动适应模型和数据集

Q: 如何知道BZ策略的效果好不好？
A: 可以通过主动学习的收敛曲线和最终模型性能来评估

Q: BZ策略失败了怎么办？
A: 系统会自动回退到几何复杂度策略，确保程序继续运行
"""
    
    print(faq)
    
    print("\n" + "="*80)
    print("总结")
    print("="*80)
    
    print("""
BZ策略已经完全集成到GNOT主动学习系统中！

🎯 目标明确: 选择模型表现最差的样本进行改进
🔧 集成完美: 直接使用GNOT现有的评估框架  
📈 效果优秀: 基于实际模型表现的科学选择
⚡ 使用简单: 只需要设置 strategy='bz'

立即开始使用:
active_learning_loop(dataset_name='your_dataset', strategy='bz')

这是最适合GNOT框架的主动学习策略！
""")

if __name__ == "__main__":
    run_bz_strategy_example()
