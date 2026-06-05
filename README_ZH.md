# AI-IMU 航位推算系统 - 完整使用指南

[English README](README.md) | **中文文档**

## 📋 目录

- [项目简介](#项目简介)
- [环境准备](#环境准备)
- [数据集下载与预处理](#数据集下载与预处理)
- [模型配置选择](#模型配置选择)
- [训练流程](#训练流程)
- [测试流程](#测试流程)
- [结果可视化](#结果可视化)
- [完整示例](#完整示例)
- [常见问题](#常见问题)
- [性能指标](#性能指标)

---

## 项目简介

本项目实现了一种**仅基于惯性测量单元（IMU）**的高精度航位推算方法，适用于智能车辆在无GNSS、LiDAR或视觉传感器失效情况下的安全导航。

### 核心特点

✅ **纯IMU输入** - 无需其他传感器  
✅ **高精度估计** - KITTI数据集平均平移误差仅1.10%  
✅ **实时处理** - 支持100Hz IMU输入  
✅ **自适应噪声** - 深度学习动态调整卡尔曼滤波参数  
✅ **多模型支持** - Baseline CNN、固定窗口注意力、动态窗口注意力  

### 技术架构

系统由两个核心模块组成：

1. **滤波器模块** - 扩展卡尔曼滤波（IEKF），融合IMU积分与零速约束
2. **噪声参数适配器** - 深度神经网络，将原始IMU信号映射为实时协方差矩阵

![系统结构](temp/structure.jpg)

---

## 环境准备

### 系统要求

- **Python**: 3.5+（推荐 3.7-3.9）
- **PyTorch**: 开发版或稳定版
- **操作系统**: Linux / Windows / macOS

### 安装依赖

```bash
pip install matplotlib numpy termcolor scipy navpy torch
```

> 💡 **提示**: 如果使用GPU训练，请确保安装了正确版本的CUDA Toolkit和cuDNN。

---

## 数据集下载与预处理

### 方法一：直接下载预处理数据（⭐ 推荐）

这是最简单快速的方式：

```bash
# 1. 下载处理好的KITTI IMU数据
wget "https://github.com/user-attachments/files/17930695/data.zip"

# 2. 创建必要的目录
mkdir -p ai-imu-dr/results
mkdir -p ai-imu-dr/data
mkdir -p ai-imu-dr/temp

# 3. 解压数据到项目目录
unzip data.zip -d ai-imu-dr
rm data.zip
```

### 方法二：从原始KITTI数据处理

如果您想从头处理原始数据：

#### 步骤1: 下载KITTI Raw Dataset

访问 [KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php)，下载所需序列。

建议至少下载以下序列：
- **训练集**: 00, 01, 04-11
- **验证集**: 部分序列用于交叉验证
- **测试集**: 02, 03等

#### 步骤2: 配置数据路径

编辑 `src/main_kitti.py`，修改数据路径：

```python
class KITTIArgs():
    path_data_base = "/path/to/KITTI/raw"  # 改为您的KITTI原始数据路径
    path_data_save = "../data"              # 处理后数据保存路径
    path_results = "../results"             # 结果保存路径
    path_temp = "../temp"                   # 临时文件路径
```

#### 步骤3: 运行数据预处理

```bash
cd src
python main_kitti.py --read-data
```

或使用命令行参数指定路径：

```bash
python main_kitti.py --read-data --path-data-base /your/path/to/KITTI/raw
```

> ⏱️ **注意**: 数据预处理可能需要较长时间，取决于数据量大小。

---

## 模型配置选择

项目支持三种模型架构，您可以根据需求选择：

### 📊 配置对比表

| 配置名称 | 模型类型 | 窗口策略 | 性能 | 计算成本 | 适用场景 |
|---------|---------|---------|------|---------|---------|
| `baseline` | CNN基线 | 无 | ⭐⭐ | 低 | 快速实验、基线对比 |
| `fixed_small` | 固定窗口 | 小窗口 | ⭐⭐⭐ | 中 | 资源受限场景 |
| `fixed_standard` | 固定窗口 | 标准窗口 | ⭐⭐⭐⭐ | 中 | 平衡性能和成本 |
| `fixed_large` | 固定窗口 | 大窗口 | ⭐⭐⭐⭐ | 中高 | 需要更长历史依赖 |
| `dynamic_conservative` | 动态窗口 | 保守调整 | ⭐⭐⭐⭐⭐ | 高 | 稳定性优先 |
| `dynamic_standard` | 动态窗口 | 标准调整 | ⭐⭐⭐⭐⭐ | 高 | **推荐默认配置** |
| `dynamic_aggressive` | 动态窗口 | 激进调整 | ⭐⭐⭐⭐⭐ | 很高 | 追求极致性能 |

### 🔍 查看可用配置

```bash
cd src
python train_torch_filter.py --list-configs
```

### 💡 选择建议

- **初学者**: 从 `dynamic_standard` 开始
- **资源有限**: 使用 `fixed_standard` 或 `baseline`
- **追求精度**: 使用 `dynamic_aggressive`
- **快速验证**: 使用 `baseline`

---

## 训练流程

### 基本训练命令

```bash
cd src

# 使用默认配置（dynamic_standard）训练
python main_kitti.py --train

# 指定配置训练
python main_kitti.py --config dynamic_standard --train

# 自定义训练轮数和序列长度
python main_kitti.py --config fixed_standard --train --epochs 200 --seq-dim 6000
```

### 训练参数详解

| 参数 | 说明 | 默认值 | 示例 |
|-----|------|--------|------|
| `--config` | 模型配置名称 | dynamic_standard | baseline, fixed_standard, dynamic_aggressive |
| `--epochs` | 训练轮数 | 400 | 200, 600, 1000 |
| `--seq-dim` | 序列长度（帧数） | 6000 | 3000, 6000, 10000 |
| `--no-continue` | 从头开始训练 | False | 不继续之前的检查点 |
| `--path-temp` | 临时文件路径 | ../temp | 自定义检查点保存位置 |

### 继续训练 vs 从头训练

**继续训练**（默认行为）：
```bash
python main_kitti.py --config dynamic_standard --train
```
自动从上次保存的检查点继续训练。

**从头训练**：
```bash
python main_kitti.py --config dynamic_standard --train --no-continue
```
忽略之前的检查点，重新开始训练。

### 监控训练过程

训练过程中会实时输出：

```
Configuration: dynamic_standard
Model Type: dynamic_window
Description: Dynamic window attention with standard parameters
Parameters: {...}
✓ Dynamic Window Attention enabled
  - Window range: [min, max]
  - Number of heads: X
  - Number of candidates: Y

0 loss: 0.12345
1 loss: 0.11234
...
gradient norm: 0.56789
Train Epoch: 1   Loss: 12.34567
Amount of time spent for 1 epoch: 120s
```

关键指标：
- **Loss**: 损失值，应逐渐下降
- **Gradient norm**: 梯度范数，应在合理范围内（< 1.0）
- **Epoch time**: 每轮训练时间

### 训练检查点

训练检查点自动保存在：
```
../temp/iekfnets_{config_suffix}.p
```

例如：
- `iekfnets_dws.p` - dynamic_standard配置
- `iekfnets_fwm.p` - fixed_standard配置
- `iekfnets_base.p` - baseline配置

---

## 测试流程

### 基本测试命令

```bash
cd src

# 使用训练好的模型测试
python main_kitti.py --test

# 指定配置测试
python main_kitti.py --config dynamic_standard --test
```

### 测试输出

测试结果保存在：
```
../results/{sequence_name}_filter.p
```

每个测试序列的结果文件包含：
- `t`: 时间戳
- `Rot`: 估计的姿态矩阵
- `v`: 估计的速度
- `p`: 估计的位置
- `b_omega`: 陀螺仪偏差
- `b_acc`: 加速度计偏差
- `Rot_c_i`: IMU到车体的旋转矩阵
- `t_c_i`: IMU到车体的平移向量
- `measurements_covs`: 测量协方差矩阵

### 测试序列

默认测试所有KITTI里程计基准序列：
- 2011_10_03_drive_0027_extract
- 2011_10_03_drive_0042_extract
- 2011_10_03_drive_0034_extract
- 2011_09_26_drive_0067_extract
- 2011_09_30_drive_0016_extract
- 2011_09_30_drive_0018_extract
- 2011_09_30_drive_0020_extract
- 2011_09_30_drive_0027_extract
- 2011_09_30_drive_0028_extract (主要测试序列)
- 2011_09_30_drive_0033_extract
- 2011_09_30_drive_0034_extract

---

## 结果可视化

### 生成可视化图表

```bash
cd src

# 生成所有序列的可视化结果
python main_kitti.py --results

# 指定配置
python main_kitti.py --config dynamic_standard --results
```

### 可视化内容

生成的图表包括：

1. **位置和速度轨迹** (`position_velocity.png`)
   - 3D位置对比（估计 vs Ground Truth）
   - 3D速度对比
   - 机体坐标系下的速度

2. **姿态和偏差** (`orientation_bias.png`)
   - 欧拉角变化（roll, pitch, yaw）
   - 陀螺仪偏差估计
   - 加速度计偏差估计

3. **轨迹对比** (`position_xy.png`, `position_xy_aligned.png`)
   - XY平面轨迹
   - 对齐后的轨迹对比

4. **协方差演化** (`measurements_covs.png`)
   - 零横向速度协方差
   - 零垂直速度协方差
   - 对数尺度显示

5. **IMU输入** (`imu.png`)
   - 陀螺仪原始数据
   - 加速度计原始数据

6. **误差分析** (`errors.png`)
   - MATE (Mean Absolute Trajectory Error)
   - CATE (Cumulative Absolute Trajectory Error)
   - RMSE (Root Mean Square Error)

### 保存位置

可视化图片保存在：
```
../results/{sequence_name}/
```

例如：
```
../results/2011_09_30_drive_0028_extract/position_velocity.png
../results/2011_09_30_drive_0028_extract/errors.png
...
```

---

## 完整示例

### 从零开始的完整流程

```bash
# ==================== 1. 环境准备 ====================
pip install matplotlib numpy termcolor scipy navpy torch

# ==================== 2. 克隆项目 ====================
git clone https://github.com/mbrossar/ai-imu-dr.git
cd ai-imu-dr

# ==================== 3. 创建目录 ====================
mkdir -p results data temp

# ==================== 4. 下载数据 ====================
# 方法A: 下载预处理数据（推荐）
wget "https://github.com/user-attachments/files/17930695/data.zip"
unzip data.zip -d .
rm data.zip

# 方法B: 从原始数据处理（可选）
# python src/main_kitti.py --read-data --path-data-base /path/to/KITTI/raw

# ==================== 5. 下载预训练权重（可选） ====================
wget "https://www.dropbox.com/s/77kq4s7ziyvsrmi/temp.zip"
unzip temp.zip -d temp
rm temp.zip

# ==================== 6. 训练模型 ====================
cd src

# 使用推荐的dynamic_standard配置训练400轮
python main_kitti.py --config dynamic_standard --train --epochs 400

# 或者从头训练
python main_kitti.py --config dynamic_standard --train --epochs 400 --no-continue

# ==================== 7. 测试模型 ====================
python main_kitti.py --config dynamic_standard --test

# ==================== 8. 可视化结果 ====================
python main_kitti.py --config dynamic_standard --results
```

### 快速测试（使用预训练权重）

如果您已下载预训练权重，可以快速测试：

```bash
cd src
python main_kitti.py --test --results
```

### 不同配置的对比实验

```bash
cd src

# 实验1: Baseline CNN
python main_kitti.py --config baseline --train --epochs 200
python main_kitti.py --config baseline --test --results

# 实验2: Fixed Window Attention
python main_kitti.py --config fixed_standard --train --epochs 200
python main_kitti.py --config fixed_standard --test --results

# 实验3: Dynamic Window Attention（推荐）
python main_kitti.py --config dynamic_standard --train --epochs 400
python main_kitti.py --config dynamic_standard --test --results
```

---

## 常见问题

### ❓ Q1: 训练不收敛怎么办？

**解决方案：**

1. **降低学习率**
   ```python
   # 在 train_torch_filter.py 中修改
   lr_initprocesscov_net = 1e-5  # 原来是 1e-4
   lr_mesnet = {
       'cov_net': 1e-5,
       'cov_lin': 1e-5,
   }
   ```

2. **减小序列长度**
   ```bash
   python main_kitti.py --config dynamic_standard --train --seq-dim 3000
   ```

3. **检查数据预处理**
   - 确认数据格式正确
   - 检查是否有异常值
   - 验证归一化是否正确

4. **从简单配置开始**
   ```bash
   # 先用baseline训练
   python main_kitti.py --config baseline --train --epochs 100
   # 再切换到dynamic
   python main_kitti.py --config dynamic_standard --train --epochs 400
   ```

### ❓ Q2: 内存不足怎么办？

**解决方案：**

1. **减小序列长度**
   ```bash
   python main_kitti.py --config dynamic_standard --train --seq-dim 3000
   ```

2. **使用CPU训练**
   ```python
   # 在代码中设置
   device = torch.device('cpu')
   ```

3. **选择轻量级配置**
   ```bash
   python main_kitti.py --config fixed_small --train
   ```

4. **关闭其他程序**释放内存

### ❓ Q3: 如何提高精度？

**优化建议：**

1. **使用Dynamic配置**
   ```bash
   python main_kitti.py --config dynamic_aggressive --train --epochs 600
   ```

2. **增加训练轮数**
   ```bash
   python main_kitti.py --config dynamic_standard --train --epochs 800
   ```

3. **确保训练数据质量**
   - 排除有问题的序列
   - 增加训练数据量
   - 数据增强

4. **调整超参数**
   - 学习率调度
   - 权重衰减
   - Batch size

### ❓ Q4: 如何评估性能？

**评估方法：**

1. **查看可视化结果**
   ```bash
   python main_kitti.py --config dynamic_standard --results
   ```

2. **分析误差指标**
   - **MATE**: 平均绝对轨迹误差
   - **CATE**: 累积绝对轨迹误差
   - **RMSE**: 均方根误差

3. **计算平移误差百分比**
   ```python
   # 平移误差 = (估计轨迹长度 - GT轨迹长度) / GT轨迹长度 × 100%
   ```

4. **对比不同配置**
   - 在同一序列上测试不同配置
   - 比较误差曲线
   - 分析轨迹对齐程度

### ❓ Q5: 训练时间太长怎么办？

**加速方法：**

1. **使用GPU**
   ```python
   # 确保PyTorch使用GPU
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   ```

2. **减少训练轮数**
   ```bash
   python main_kitti.py --config dynamic_standard --train --epochs 200
   ```

3. **减小序列长度**
   ```bash
   python main_kitti.py --config dynamic_standard --train --seq-dim 4000
   ```

4. **使用较小的配置**
   ```bash
   python main_kitti.py --config fixed_small --train
   ```

### ❓ Q6: 如何使用自己的IMU数据？

**自定义数据集步骤：**

1. **继承BaseDataset类**
   ```python
   from dataset import BaseDataset
   
   class MyDataset(BaseDataset):
       def __init__(self, args):
           super(MyDataset, self).__init__(args)
           # 初始化您的数据集
       
       @staticmethod
       def read_data(args):
           # 实现数据读取逻辑
           pass
       
       def get_data(self, dataset_name):
           # 返回 t, ang_gt, p_gt, v_gt, u
           pass
   ```

2. **注册数据集**
   ```python
   # 在 main_kitti.py 中
   class MyArgs():
       dataset_class = MyDataset
       parameter_class = MyParameters
   ```

3. **定义训练/测试划分**
   ```python
   self.datasets_train_filter["sequence_1"] = [start_idx, end_idx]
   self.datasets_validatation_filter["sequence_2"] = [start_idx, end_idx]
   ```

参考 [`KITTIDataset`](file://d:\user\desktop\生成作业2\ai-imu-dr\src\main_kitti.py#L73-L415) 类的实现。

---

## 性能指标

### KITTI数据集基准性能

| 指标 | 数值 | 说明 |
|-----|------|------|
| **平均平移误差** | ~1.10% | 仅使用IMU |
| **IMU频率** | 100 Hz | 实时处理 |
| **训练序列** | 00, 01, 04-11 | 8个序列 |
| **测试序列** | 02, 03等 | 未见过的序列 |

### 不同配置的预期性能

| 配置 | 相对性能 | 训练时间 | 推理速度 | 推荐度 |
|-----|---------|---------|---------|--------|
| Baseline | ⭐⭐ | 快 | 快 | ⭐⭐ |
| Fixed Small | ⭐⭐⭐ | 中 | 中 | ⭐⭐⭐ |
| Fixed Standard | ⭐⭐⭐⭐ | 中 | 中 | ⭐⭐⭐⭐ |
| Fixed Large | ⭐⭐⭐⭐ | 较慢 | 较慢 | ⭐⭐⭐ |
| Dynamic Conservative | ⭐⭐⭐⭐⭐ | 慢 | 慢 | ⭐⭐⭐⭐ |
| Dynamic Standard | ⭐⭐⭐⭐⭐ | 慢 | 慢 | ⭐⭐⭐⭐⭐ |
| Dynamic Aggressive | ⭐⭐⭐⭐⭐ | 很慢 | 很慢 | ⭐⭐⭐⭐ |

> 💡 **注意**: Dynamic系列配置虽然计算开销增加20-30%，但通常能提供最佳精度。

---

## 数据集配置详情

### 训练集配置

当前项目使用的训练序列划分：

```python
# 训练集 (datasets_train_filter)
{
    "2011_10_03_drive_0042_extract": [0, None],        # 完整序列
    "2011_09_30_drive_0018_extract": [0, 15000],       # 前15000帧
    "2011_09_30_drive_0020_extract": [0, None],        # 完整序列
    "2011_09_30_drive_0027_extract": [0, None],        # 完整序列
    "2011_09_30_drive_0033_extract": [0, None],        # 完整序列
    "2011_10_03_drive_0027_extract": [0, 18000],       # 前18000帧
    "2011_10_03_drive_0034_extract": [0, 31000],       # 前31000帧
    "2011_09_30_drive_0034_extract": [0, None],        # 完整序列
}

# 验证集 (datasets_validatation_filter)
{
    "2011_09_30_drive_0028_extract": [11231, 53650],   # 交叉验证
}

# 测试集 (odometry_benchmark)
# 包含11个KITTI里程计基准序列
```

### 排除的数据集

以下序列因质量问题被自动排除：
- `2011_09_26_drive_0093_extract`
- `2011_09_28_drive_0039_extract`
- `2011_09_28_drive_0002_extract`

### 已知问题序列

- `2011_09_30_drive_0028_extract`: N=[6000, 14000] 区间有问题 → 用作测试数据
- `2011_10_03_drive_0027_extract`: N=29481 处有问题
- `2011_10_03_drive_0034_extract`: N=[33500, 34000] 区间有问题

---

## 引用

如果您在研究中使用了本项目的代码，请引用：

```bibtex
@article{brossard2019aiimu,
  author = {Martin Brossard and Axel Barrau and Silv\`ere Bonnabel},
  journal={IEEE Transactions on Intelligent Vehicles}, 
  title = {{AI-IMU Dead-Reckoning}},
  year = {2020}
}
```

**论文链接**: 
- [IEEE Xplore](https://ieeexplore.ieee.org/document/9035481)
- [ArXiv](https://arxiv.org/pdf/1904.06064.pdf)

---

## 作者信息

- **Martin Brossard*** - MINES ParisTech, PSL Research University
- **Axel Barrau°** - Safran Tech, Groupe Safran
- **Silvère Bonnabel*** - MINES ParisTech, PSL Research University

*Centre for Robotics, 60 Boulevard Saint-Michel, 75006 Paris, France  
°Rue des Jeunes Bois-Châteaufort, 78772, Magny Les Hameaux Cedex, France

---

## 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

---

## 相关链接

- [英文README](README.md)
- [模型详细说明](README_MODELS.md)
- [CLI配置总结](CLI_CONFIG_SUMMARY.md)
- [快速入门指南](GETTING_STARTED.md)
- [实现指南](IMPLEMENTATION_GUIDE.md)
- [CLI使用说明](USAGE_CLI.md)

---

**祝您使用愉快！如有问题，欢迎提交Issue。** 🚀
