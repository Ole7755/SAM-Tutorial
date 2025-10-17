# SAM Adversarial Attack Research

## 项目概述

本项目旨在系统性研究Segment Anything Model (SAM)在对抗攻击下的鲁棒性，探索不同攻击方法对SAM分割能力的影响。

**核心研究问题**：能否通过对抗扰动让SAM "Segment Nothing"？

---

## 研究目标

### 主要目标
让SAM在接收对抗样本后产生错误的分割结果，具体表现为：
- 生成空mask（无分割区域）
- 生成极小分割区域
- 质量分数显著下降
- 分割结果与真实目标严重偏离

### 研究意义
1. **安全性评估**：评估SAM在对抗环境下的可靠性
2. **鲁棒性分析**：识别SAM的脆弱点
3. **防御启发**：为未来的防御机制提供insights
4. **理论贡献**：探索视觉基础模型的对抗特性

---

## 实验设计

### 攻击配置

#### 攻击对象
- **目标模型**：SAM (ViT-L)
- **攻击目标**：图像输入（固定prompt不变）
- **攻击类型**：Untargeted Attack
- **期望结果**：Segment Nothing

#### 提示配置
- **提示类型**：单点提示（Point Prompt）
- **提示位置**：目标对象中心（如猫的鼻子）
- **标签**：前景点（label=1）

#### 测试图像
- **主要测试图像**：`cat.png` - 蓝色背景猫咪照片
- **图像特点**：背景简洁、主体清晰、适合基准测试

---

## 攻击方法

### 1. FGSM (Fast Gradient Sign Method)
**原理**：单步梯度符号攻击
```
perturbation = epsilon * sign(∇_x Loss)
adv_image = clean_image + perturbation
```

**参数设置**：
- `epsilon`: [4/255, 8/255, 16/255]

### 2. PGD (Projected Gradient Descent)
**原理**：多步迭代优化攻击
```
for t in 1..T:
    perturbation_t = alpha * sign(∇_x Loss)
    adv_image = clip(adv_image + perturbation_t, [x-epsilon, x+epsilon])
```

**参数设置**：
- `epsilon`: 8/255
- `alpha`: 2/255
- `steps`: [7, 10, 20]

### 3. 未来扩展方法
- C&W Attack
- DeepFool
- AutoAttack
- 其他白盒/黑盒攻击

---

## 损失函数设计

针对"Segment Nothing"目标，我们设计并对比以下四种损失函数：

### Loss 1: 最小化Mask面积
```python
loss = torch.sigmoid(logits).sum()
```
**直觉**：直接最小化mask中激活像素的数量

### Loss 2: 最小化质量分数
```python
loss = predicted_iou_score
```
**直觉**：利用SAM自身的质量评估，让模型认为分割质量很差

### Loss 3: 反转分割目标（BCE）
```python
target = torch.zeros_like(clean_mask)
loss = F.binary_cross_entropy_with_logits(logits, target)
```
**直觉**：显式地将目标设为空mask

### Loss 4: 组合损失
```python
loss = alpha * torch.sigmoid(logits).sum() + beta * (-predicted_iou_score)
```
**直觉**：同时优化mask面积和质量分数
- `alpha`: 1.0
- `beta`: 10.0

---

## 评估指标

### 主要指标

#### 1. Mask Area Ratio (MAR)
```
MAR = Area(adv_mask) / Area(clean_mask)
```
- **成功标准**：MAR < 0.1（对抗mask面积不到干净mask的10%）
- **理想值**：接近0

#### 2. Intersection over Union (IoU)
```
IoU = |clean_mask ∩ adv_mask| / |clean_mask ∪ adv_mask|
```
- **成功标准**：IoU < 0.3
- **理想值**：接近0

#### 3. Quality Score Drop (QSD)
```
QSD = clean_score - adv_score
```
- **成功标准**：QSD > 0.5
- **理想值**：接近1.0

### 扰动指标

#### 4. L∞ Norm
```
L∞ = max|adv_image - clean_image|
```
- **约束**：≤ epsilon

#### 5. L2 Norm
```
L2 = ||adv_image - clean_image||_2
```
- **指标**：越小越好（在满足攻击成功的前提下）

### 成功率指标

#### 6. Attack Success Rate (ASR)
```
ASR = (攻击成功样本数 / 总样本数) × 100%
```
**攻击成功定义**：MAR < 0.1 OR IoU < 0.3

---

## 实验流程

```
1. 环境准备
   ├── 加载SAM模型（ViT-L）
   ├── 加载测试图像
   └── 设置实验参数

2. Baseline建立
   ├── 干净图像 + 点提示 → 分割结果
   ├── 记录clean_mask, clean_score
   └── 可视化baseline

3. FGSM攻击实验
   ├── 遍历4种损失函数
   ├── 遍历3种epsilon值
   ├── 生成对抗样本
   ├── 评估攻击效果
   └── 记录所有指标

4. PGD攻击实验
   ├── 遍历4种损失函数
   ├── 遍历3种steps值
   ├── 迭代生成对抗样本
   ├── 评估攻击效果
   └── 记录所有指标

5. 结果分析
   ├── 对比不同损失函数效果
   ├── 对比FGSM vs PGD
   ├── 分析攻击成功率
   ├── 可视化对比图
   └── 生成实验报告

6. 最佳方法识别
   └── 综合评估确定最优攻击配置
```

---

## 预期输出

### 1. 量化结果表格
```
| Method | Loss Type | Epsilon/Steps | MAR↓ | IoU↓ | QSD↑ | L∞ | ASR↑ |
|--------|-----------|---------------|------|------|------|-----|------|
| FGSM   | Loss1     | 8/255         | ...  | ...  | ...  | ... | ...  |
| ...    | ...       | ...           | ...  | ...  | ...  | ... | ...  |
```

### 2. 可视化结果
- 干净图像 vs 对抗图像
- 干净mask vs 对抗mask
- 扰动可视化（perturbation heatmap）
- 不同方法效果对比图

### 3. 实验报告
- 最佳攻击方法及其参数
- 不同损失函数的优劣分析
- FGSM vs PGD的对比结论
- 未来研究方向建议

---

## 文件结构

SAM-Tutorial/
├── README.md                          # 本文档
├── requirements.txt                   # 依赖包
├── sam_vit_l_0b3195.pth              # 模型权重
├── cat.png                            # 测试图像
│
├── tutorials/                         # 基础教程
│   ├── sam_tutorial.py               # SAM使用教程
│   └── sam_interactive.py            # 交互式工具
│
├── experiments/                       # 实验脚本
│   └── run_adversarial_attacks.py    # 完整的对抗攻击实验
│
└── results/                           # 实验结果（自动生成）
    ├── figures/                      # 可视化图片
    │   ├── baseline.png              # baseline结果
    │   ├── fgsm_*.png                # FGSM攻击结果
    │   ├── pgd_*.png                 # PGD攻击结果
    │   └── comparison_analysis.png   # 对比分析
    └── logs/                         # 实验日志
        ├── results.csv               # 结果表格
        └── results.json              # 详细结果

---

## 下一步工作

### Phase 1: 实现核心攻击框架 ✅ (当前)
- [x] 项目规划和文档
- [x] 实现4种损失函数
- [x] 实现FGSM攻击
- [x] 实现PGD攻击
- [x] 实现评估指标

### Phase 2: 完整实验和分析
- [x] 运行所有实验配置
- [x] 收集和整理数据
- [x] 生成可视化结果
- [x] 撰写实验报告
- [x] 识别最佳攻击方法

### Phase 3: 扩展研究（可选）
- [ ] 测试更多图像
- [ ] 实现其他攻击方法（C&W, DeepFool等）
- [ ] 测试其他提示类型（box, multi-point）
- [ ] 研究防御方法
- [ ] 撰写学术论文

---

## 参考文献

### SAM相关
- Kirillov et al. "Segment Anything." ICCV 2023.

### 对抗攻击相关
- Goodfellow et al. "Explaining and Harnessing Adversarial Examples." ICLR 2015. (FGSM)
- Madry et al. "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018. (PGD)

---
SAM权重:ViT-L 
下载命令:wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth

## 联系方式

**研究者**：Ole
**研究方向**：Adversarial Attacks on Vision Models
**项目仓库**：https://github.com/Ole7755/SAM-Tutorial.git

---

**最后更新**：2025-10-16
