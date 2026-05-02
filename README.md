# U-Net 语义分割 —— Stanford Background Dataset

从零搭建 U-Net 图像分割网络，在 Stanford Background Dataset 上对比不同损失函数（Cross-Entropy Loss、Dice Loss、Combined Loss）对分割性能的影响。

## 项目结构

```
unet_project/
├── main.py         # 入口，命令行参数解析
├── model.py        # U-Net 模型定义（从零实现，无预训练）
├── dataset.py      # Stanford Background Dataset 数据加载
├── train.py        # 训练 & 验证流程
├── losses.py       # Cross-Entropy Loss、Dice Loss、Combined Loss
├── utils.py        # mIoU 和 pixel accuracy 计算
├── visualize.py    # 推理 & 三栏可视化（原图 | 标签 | 预测）
├── run_all.py      # 依次运行三种损失函数的实验
├── requirements.txt
├── checkpoints/    # 保存的模型权重（训练后生成）
└── vis_results/    # 可视化结果（训练后生成）
```

## 环境配置

依赖项：

- Python 3.8+
- PyTorch >= 2.0.0
- torchvision >= 0.15.0
- numpy
- Pillow
- wandb（实验跟踪）

安装：

```bash
pip install -r requirements.txt
```

## 数据集

**Stanford Background Dataset** —— 轻量级语义分割数据集，包含 8 个语义类别：

| ID | 类别      |
|----|-----------|
| 0  | sky       |
| 1  | tree      |
| 2  | road      |
| 3  | grass     |
| 4  | water     |
| 5  | building  |
| 6  | mountain  |
| 7  | object    |

数据集按以下结构放在 `data/` 目录下：

```
data/
├── images/          # *.jpg
└── labels/          # *.regions.txt
```

数据集按 80/20 比例划分为训练集和验证集（固定随机种子，可复现）。

## 模型

经典 U-Net 架构，从零实现，不加载任何预训练权重：

- **编码器**：4 个下采样块（MaxPool → DoubleConv）
- **解码器**：4 个上采样块（转置卷积 → 跳跃连接 → DoubleConv）
- **跳跃连接**：编码器特征与解码器对应层拼接
- **参数初始化**：所有权重随机初始化

网络通道数：`3 → 64 → 128 → 256 → 512 → 1024 → 512 → 256 → 128 → 64 → 8`

总参数量：约 31M。

## 训练

### 单个实验

```bash
cd unet_project
python main.py --loss ce       --epochs 60 --batch_size 16 --lr 1e-3 --img_size 256
python main.py --loss dice     --epochs 60 --batch_size 16 --lr 1e-3 --img_size 256
python main.py --loss combined --epochs 60 --batch_size 16 --lr 1e-3 --img_size 256
```

### 三种损失依次运行

```bash
cd unet_project
python run_all.py
```

### 参数说明

| 参数          | 默认值  | 可选值                 | 说明                  |
|---------------|---------|------------------------|----------------------|
| `--loss`      | ce      | ce, dice, combined     | 损失函数              |
| `--epochs`    | 60      | —                      | 训练轮数              |
| `--batch_size`| 16      | —                      | 批大小                |
| `--lr`        | 1e-3    | —                      | 学习率                |
| `--img_size`  | 256     | —                      | 输入图像尺寸（正方形） |

### 损失函数

1. **Cross-Entropy Loss**（`ce`）—— 逐像素标准交叉熵，忽略 `ignore_index=255`
2. **Dice Loss**（`dice`）—— 手动实现的多类别 Dice Loss
3. **Combined Loss**（`combined`）—— CE + Dice 等权组合

### 训练细节

- 优化器：Adam（lr=1e-3）
- 学习率调度器：ReduceLROnPlateau（factor=0.5, patience=5）
- 评估指标：mIoU（mean Intersection-over-Union），对 8 个类别取平均
- 未知像素（label=-1）映射为 `ignore_index=255`，不参与损失计算和指标评估
- 实验跟踪：[wandb](https://wandb.ai/)（project 名：`hw2-unet`）

## 可视化

在验证集样本上运行推理，生成三栏对比图：

```bash
cd unet_project
python visualize.py --checkpoint checkpoints/best_ce.pth --loss ce --num_samples 8
```

输出：`vis_results/sample_*.png`（原图 | 真实标签 | 预测结果 三栏并排）。

## 评估指标

- **mIoU**（mean Intersection-over-Union）：逐类计算 IoU 后取平均（某类在样本中未出现则跳过）
- **Pixel Accuracy**：正确分类的像素占比（排除 `ignore_index`）

## 预训练模型权重

三种损失函数训练好的模型权重下载：[Google Drive](https://drive.google.com/drive/folders/1nV6mA84pEKzXWALl91da2m1_NyzMlBw0?usp=sharing)

权重文件放置方式：下载后放在 `unet_project/checkpoints/` 目录下。

| 文件                     | 对应损失函数 |
|--------------------------|--------------|
| `best_ce.pth`            | CE           |
| `best_dice.pth`          | Dice         |
| `best_combined.pth`      | CE + Dice    |

## 实验结果

| Loss      | 最佳验证集 mIoU |
|-----------|-----------------|
| CE        | 0.6264          |
| Dice      | 0.5992          |
| Combined  | 0.6322          |

*（训练完成后填入结果。）*

## 参考

- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)（Ronneberger et al., 2015）
- [Stanford Background Dataset](https://dags.stanford.edu/projects/sceneunderstanding.html)
