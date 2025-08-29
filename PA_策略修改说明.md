# PA策略修改说明

## 概述

根据用户需求，我已经修改了PA（Precision Acquisition）策略，从原来基于物理一致性（Navier-Stokes方程违约程度）的选择策略，改为基于预测误差的选择策略。

## 主要修改

### 1. 核心策略函数

**新增的主要函数：**

- `pa_query_average_error()`: 基于五个值平均误差的PA查询（默认版本）
- `pa_query_prediction_error()`: 可自定义权重的PA查询
- `pa_query_fixed()`: 底层实现函数

### 2. 误差计算逻辑

对于每个未标注样本：

1. **输入数据**: x-coordinate, y-coordinate, z-coordinate（血管形状坐标）
2. **模型预测**: pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear
3. **真实标签**: 内部的真实pressure, wall-shear, x-wall-shear, y-wall-shear, z-wall-shear值

**误差计算方法：**
```python
# 对每个字段计算平均绝对误差
mae = np.mean(np.abs(predicted - true))

# 相对误差归一化
relative_error = mae / (np.mean(np.abs(true)) + 1e-8)

# 加权总误差
total_error = sum(relative_error_i * weight_i for i in range(5))
```

### 3. 策略选择

策略按总预测误差从大到小排序，优先选择预测误差最大的样本。这些样本是：
- 模型预测最不准确的样本
- 最需要被加入训练集的样本
- 对提高模型性能最有帮助的样本

## 使用方法

### 方法1：默认平均误差策略（推荐）

```python
# 在主动学习循环中使用
active_learning_loop(
    dataset_name='al_pa',
    output_dir='./output',
    strategy='pa',          # 使用基于预测误差的PA策略
    rounds=10,
    select_num=5
)
```

这种方法对五个输出值使用等权重（1.0, 1.0, 1.0, 1.0, 1.0），即计算平均误差。

### 方法2：自定义权重策略

```python
# 可以在代码中修改权重配置
active_learning_loop(
    dataset_name='al_pa',
    output_dir='./output', 
    strategy='pa_custom',   # 使用自定义权重策略
    rounds=10,
    select_num=5
)
```

在`alpa.py`中的`pa_query_prediction_error`函数里可以修改权重：
```python
selected_idx = pa_query_prediction_error(
    model_tuple, unlabeled_data, select_num,
    pressure_weight=2.0,        # 加重pressure的权重
    wall_shear_weight=1.0,      
    x_wall_shear_weight=1.0,    
    y_wall_shear_weight=1.0,    
    z_wall_shear_weight=1.0     
)
```

### 方法3：保留的物理一致性策略

```python
# 用于对比研究
active_learning_loop(
    dataset_name='al_pa',
    output_dir='./output',
    strategy='pa_physics',  # 使用原始物理一致性策略
    rounds=10,
    select_num=5
)
```

## 技术细节

### 数据流程

1. **数据读取**: 从`test2.csv`文件读取坐标和真实值
2. **模型推理**: 使用当前训练的模型对未标注样本进行预测
3. **误差计算**: 比较预测值与真实值，计算各字段误差
4. **样本排序**: 按总误差从大到小排序
5. **样本选择**: 选择误差最大的前N个样本

### 主要优势

1. **直接性**: 直接基于模型预测性能进行选择
2. **效率**: 比物理方程计算更高效
3. **针对性**: 专门针对提高模型预测准确性
4. **灵活性**: 支持自定义权重配置
5. **实用性**: 利用了未标注数据中的真实标签信息

### 容错机制

- 如果预测误差计算失败，自动回退到几何复杂度评分
- 支持设备转移的错误处理
- 临时文件的自动清理

## 文件修改

主要修改文件：`/home/v-wenliao/gnot/GNOT/alpa.py`

**新增函数：**
- `pa_query_average_error()`
- `pa_query_prediction_error()`
- `pa_query_physics_based()`（向后兼容）

**修改内容：**
- `pa_query_fixed()`: 完全重写为基于预测误差的实现
- 主动学习循环中的策略选择逻辑
- 添加了详细的错误处理和日志输出

## 测试验证

创建了测试脚本：`test_pa_prediction_error.py`

可以通过以下命令测试新策略：
```bash
cd /home/v-wenliao/gnot/GNOT
python test_pa_prediction_error.py
```

## 总结

修改后的PA策略现在根据预测误差进行样本选择，默认使用五个输出值的平均误差作为选择依据。这种方法更直接地针对模型性能进行优化，符合用户的需求。同时保留了灵活性，可以根据具体需求调整各字段的权重。
