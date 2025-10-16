# SAM对抗攻击研究项目

## 项目概述

本项目旨在系统性研究Segment Anything Model (SAM)在对抗攻击下的鲁棒性，探索不同攻击策略对SAM分割性能的影响，为提升视觉基础模型的安全性提供实证研究。

---

## 研究目标

### 主要目标
使SAM模型在对抗样本输入下产生**"Segment Nothing"**效果，即：
- 输入干净图像 + 提示 → 正确分割目标对象
- 输入对抗样本 + 相同提示 → 输出空mask或极小区域

### 科学问题
1. SAM对图像层面的对抗扰动有多脆弱？
2. 不同损失函数设计对攻击效果的影响？
3. FGSM vs PGD：哪种攻击方法对SAM更有效？
4. 扰动预算（ε）与攻击成功率的关系？

---

## 攻击方案设计

### 1. 攻击配置
- **攻击对象**: 输入图像
- **攻击类型**: Untargeted（无目标攻击）
- **攻击目标**: Segment Nothing
- **提示类型**: 单点提示（Point Prompt）

### 2. 攻击方法

#### FGSM (Fast Gradient Sign Method)
```
x_adv = x + ε · sign(∇_x L(x, y))
```
- 单步攻击
- 快速生成对抗样本
- 参数: ε ∈ {4/255, 8/255, 16/255}

#### PGD (Projected Gradient Descent)
```
x_adv^(t+1) = Proj_{x+S} (x_adv^(t) + α · sign(∇_x L(x_adv^(t), y)))
```
- 迭代攻击，更强大
- 多步优化对抗扰动
- 参数: 
  - ε = 8/255 (总预算)
  - α = 2/255 (步长)
  - steps ∈ {7, 10, 20}

### 3. 损失函数设计

本项目将系统评估4种损失函数：

#### Loss 1: 最小化Mask面积
```python
L_area = Σ sigmoid(logits_i,j)
```
- **直觉**: 让mask中的像素激活值尽可能小
- **优势**: 直接优化"segment nothing"目标
- **预期**: 最直接有效

#### Loss 2: 最小化质量分数
```python
L_quality = predicted_iou_score
```
- **直觉**: 让SAM自己判断分割质量很差
- **优势**: 利用SAM的内部质量评估机制
- **预期**: 可能导致低质量但非空的mask

#### Loss 3: 反转分割目标（BCE）
```python
L_BCE = BCE(sigmoid(logits), zeros_like(target))
```
- **直觉**: 强制让所有像素预测为背景
- **优势**: 明确的监督信号
- **预期**: 稳定但可能需要更大ε

#### Loss 4: 组合损失
```python
L_combined = α · L_area + β · (-L_quality)
α = 0.7, β = 0.3
```
- **直觉**: 同时优化mask面积和质量分数
- **优势**: 多目标优化，更鲁棒
- **预期**: 平衡效果，可能最优

---

## 评估指标体系

### 主要指标

#### 1. Mask面积比 (Mask Area Ratio)
```
MAR = Area(adv_mask) / Area(clean_mask)
```
- 范围: [0, +∞)
- **攻击成功**: MAR < 0.1 (对抗mask面积 < 10%原始面积)

#### 2. IoU (Intersection over Union)
```
IoU = |adv_mask ∩ clean_mask| / |adv_mask ∪ clean_mask|
```
- 范围: [0, 1]
- **攻击成功**: IoU < 0.3

#### 3. 质量分数下降 (Quality Score Drop)
```
QSD = clean_score - adv_score
```
- 范围: [-1, 1]
- **攻击成功**: QSD > 0.5

### 辅助指标

#### 4. 扰动大小
```
L_∞ = max|x_adv - x|
L_2 = ||x_adv - x||_2
```

#### 5. PSNR (Peak Signal-to-Noise Ratio)
```
PSNR = 20 · log10(MAX_I / sqrt(MSE))
```
- 衡量图像视觉质量

#### 6. 攻击成功率 (ASR)
```
ASR = #{攻击成功样本} / #{总样本}
```
- 综合判定标准：MAR < 0.1 OR IoU < 0.3

---

## 实验设计

### 实验矩阵

| 攻击方法 | 损失函数 | ε值 | 迭代步数 | 总实验数 |
|---------|---------|-----|---------|---------|
| FGSM    | Loss 1-4 | 4/255, 8/255, 16/255 | - | 12 |
| PGD     | Loss 1-4 | 8/255 | 7, 10, 20 | 12 |
| **总计** | - | - | - | **24** |

### 实验流程

```
1. Baseline建立
   ├─ 加载SAM模型 (ViT-L)
   ├─ 读取测试图像 (cat.png)
   ├─ 定义点提示坐标
   └─ 生成clean mask和score

2. FGSM攻击实验 (12组)
   ├─ Loss 1 × ε ∈ {4/255, 8/255, 16/255}
   ├─ Loss 2 × ε ∈ {4/255, 8/255, 16/255}
   ├─ Loss 3 × ε ∈ {4/255, 8/255, 16/255}
   └─ Loss 4 × ε ∈ {4/255, 8/255, 16/255}

3. PGD攻击实验 (12组)
   ├─ Loss 1 × steps ∈ {7, 10, 20}
   ├─ Loss 2 × steps ∈ {7, 10, 20}
   ├─ Loss 3 × steps ∈ {7, 10, 20}
   └─ Loss 4 × steps ∈ {7, 10, 20}

4. 结果分析
   ├─ 定量对比（24组实验 × 6个指标）
   ├─ 可视化对比（对抗样本、mask对比、扰动可视化）
   └─ 统计分析（最佳攻击策略）
```

---

## 预期成果

### 1. 实验报告
- 24组实验的详细数据表格
- 不同损失函数的效果对比
- FGSM vs PGD的性能分析
- 扰动预算的影响分析

### 2. 可视化结果
- 对抗样本vs干净样本
- Mask对比图（24组）
- 扰动可视化（放大查看）
- 指标曲线图（ε vs ASR, steps vs IoU等）

### 3. 关键发现
- **最优攻击策略**: 最有效的损失函数 + 攻击方法组合
- **鲁棒性分析**: SAM对哪种攻击最脆弱
- **扰动效率**: 达到攻击目标所需的最小ε

---

## 项目结构

```
SAM-Tutorial/
├── README.md
├── PROJECT_GOALS.md                 # 本文档
├── requirements.txt
├── data/
│   └── cat.png                      # 测试图像
├── weights/
│   └── sam_vit_l_0b3195.pth        # SAM权重
├── src/
│   ├── __init__.py
│   ├── attack.py                    # 攻击算法实现
│   ├── loss_functions.py            # 4种损失函数
│   ├── evaluate.py                  # 评估指标计算
│   └── visualize.py                 # 可视化工具
├── experiments/
│   ├── run_fgsm_attacks.py         # FGSM实验脚本
│   ├── run_pgd_attacks.py          # PGD实验脚本
│   └── run_all_experiments.py      # 完整实验流程
├── results/
│   ├── fgsm/                        # FGSM结果
│   ├── pgd/                         # PGD结果
│   ├── metrics.csv                  # 数据汇总
│   └── visualizations/              # 可视化结果
└── analysis/
    ├── compare_losses.py            # 损失函数对比分析
    ├── compare_methods.py           # 方法对比分析
    └── generate_report.py           # 自动生成实验报告
```

---

## 后续扩展方向

### Phase 2: 提示攻击
- 攻击点坐标（微小扰动）
- 攻击边界框坐标
- 攻击多点提示

### Phase 3: 更多攻击方法
- C&W (Carlini & Wagner)
- AutoAttack
- DeepFool
- Universal Adversarial Perturbations

### Phase 4: 防御机制
- 对抗训练
- 输入预处理（去噪、压缩）
- 集成防御

### Phase 5: 迁移性研究
- 对抗样本在不同SAM模型间的迁移 (ViT-B, ViT-L, ViT-H)
- 对其他分割模型的迁移 (Mask R-CNN, DeepLab等)

---

## 时间规划

- **Week 1**: 完成FGSM和PGD的4种损失函数实现
- **Week 2**: 运行全部24组实验，收集数据
- **Week 3**: 数据分析和可视化
- **Week 4**: 撰写实验报告，确定最优策略

---

## 参考文献

1. Kirillov et al. "Segment Anything." ICCV 2023.
2. Goodfellow et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015.
3. Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.

---

## 联系与贡献

本项目为AI安全研究项目，欢迎交流和贡献。

**核心研究问题**: 如何让SAM "Segment Nothing"？哪种攻击策略最有效？

