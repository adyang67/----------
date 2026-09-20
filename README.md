# 基于纹理特征的工业缺陷检测系统（无监督异常检测）

仅使用**合格品图像**训练，推理时同时完成**缺陷判定（图像级）**与**缺陷定位（像素级）**。
对比两种路线：

| 方法 | 路线 | 特征 | 建模 | 异常分数 |
|------|------|------|------|----------|
| **A（传统）** | 手工纹理 | LBP + HOG + Gabor | One-Class SVM（RBF） | `-decision_function` |
| **B（深度）** | 预训练 + 统计 | ImageNet 预训练 ResNet 多层特征 | 逐位置多元高斯 | 马氏距离（PaDiM） |

数据集：**MVTec AD**，默认类别 `screw`（含划痕/螺纹缺陷等），可切换到任意类别。

---

## 1. 目录结构

```
人工智能综合实践课程/
├── config.py            # 全局配置（路径、类别、超参数）
├── download_data.py     # 下载并解压 MVTec AD
├── run_traditional.py   # 运行方法 A
├── run_padim.py         # 运行方法 B
├── run_compare.py       # 对比两种方法（表格 + CSV + 图）
├── requirements.txt
├── src/
│   ├── data.py          # MVTec AD 数据加载
│   ├── features.py      # LBP / HOG / Gabor 特征
│   ├── traditional.py   # 方法 A：One-Class SVM 检测器
│   ├── padim.py         # 方法 B：PaDiM 检测器
│   ├── evaluate.py      # AUROC 评估 + 速度计时
│   └── visualize.py     # 热力图可视化
├── data/                # 数据集（下载后生成）
└── results/             # 结果 JSON / CSV / 图 / 可视化样本
```

---

## 2. 环境安装（Windows + CPU）

建议 Python 3.9~3.11。在项目目录打开终端（PowerShell 或 Anaconda Prompt）：

```bash
# 1) 安装 CPU 版 PyTorch（更快，体积小）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2) 安装其余依赖
pip install -r requirements.txt
```

> 若想用 GPU，把第 1 步换成 `pip install torch torchvision`（会自动检测 CUDA 版本）。

---

## 3. 数据准备

```bash
python download_data.py
```

脚本会下载约 4.9 GB 的 `mvtec_anomaly_detection.tar.xz` 并自动解压到 `data/`。

> 若镜像直链失效，去官方页面注册下载：
> https://www.mvtec.com/company/research/datasets/mvtec-ad
> 解压后把 `mvtec_anomaly_detection` 文件夹放到 `data/` 目录下即可。

---

## 4. 运行步骤

### 4.1 方法 A：传统路线

```bash
python run_traditional.py                    # 默认类别 screw
python run_traditional.py --category leather # 切换类别
python run_traditional.py --max-test 50      # 调试：只测 50 张
```

输出：图像级/像素级 AUROC、平均推理耗时、训练耗时，并保存 4 张缺陷热力图到
`results/traditional_<category>/`。

### 4.2 方法 B：PaDiM

```bash
python run_padim.py
python run_padim.py --category leather
```

首次运行会自动下载 ImageNet 预训练权重（ResNet18 约 45 MB）。

### 4.3 对比两种方法

```bash
python run_compare.py
```

输出对比表格，并保存 `results/compare_<category>.csv` 与 `compare_<category>.png`。
（若某方法已运行过，会直接读取缓存的 JSON，不会重复训练。）

---

## 5. 方法原理简述

### 方法 A（传统）
1. 对每张图用滑窗（patch 32×32、步长 8）切块；
2. 每个 patch 提取三类纹理特征并拼接成 ~350 维向量：
   - **LBP**：均匀模式直方图（描述局部纹理模式，对光照鲁棒）；
   - **HOG**：梯度方向直方图（描述边缘/形状）；
   - **Gabor**：4 方向 × 2 频率的滤波响应均值与方差（描述方向性纹理）；
3. 用所有合格 patch 的特征训练 **One-Class SVM**（`nu=0.01`，RBF 核）；
4. 推理时 `-decision_function` 为 patch 异常分数，反投影成像素图，再高斯平滑。

### 方法 B（PaDiM）
1. ImageNet 预训练 ResNet18 前向，取 `layer1/2/3` 特征；
2. 上采样到同一分辨率，逐位置拼接成 448 维嵌入，随机投影降到 128 维（省内存）；
3. 对每个空间位置用训练集嵌入拟合**多元高斯**（均值 + 协方差，特征分解求逆）；
4. 推理时算每个位置的**马氏距离**作为异常分数，上采样 + 平滑得到像素级异常图，
   图像级分数取热力图最大值。

---

## 6. 调参建议

- **换类别**：改 `config.py` 的 `CATEGORY` 或加 `--category`。金属类：`screw`、`metal_nut`、
  `capsule`；纹理类：`leather`、`tile`、`wood`、`carpet`、`grid`。
- **方法 A 定位更精细**：减小 `STRIDE`（如 4，但更慢）；`MAX_TRAIN_PATCHES` 调大可更准。
- **方法 B 更准**：`PADIM_DIMS` 调到 256（内存翻倍但 AUROC 更高），或
  `BACKBONE = "wide_resnet50_2"`（更重、更准，训练内存显著增加）。
- **图像尺寸**：`IMAGE_SIZE` 可调 224/256（影响精度与速度）。

---

## 7. 常见问题

- **内存不足**：调小 `PADIM_DIMS`（如 64）或 `IMAGE_SIZE`；方法 A 调小 `MAX_TRAIN_PATCHES`。
- **`torch` 安装很慢/失败**：换国内镜像或使用 `--index-url https://download.pytorch.org/whl/cpu`。
- **首次运行 PaDiM 报网络错误**：预训练权重下载失败，检查网络后重跑。
- **`skimage` 的 `hog` 报 `channel_axis` 参数错误**：已做版本兼容处理；若仍报错，升级
  `scikit-image>=0.21`。

---

## 8. 预期结果（参考）

以 `screw` 类别、CPU 运行 ResNet18/PaDiM 为例（机器不同数值会有差异）：

| 指标 | 方法 A 传统 | 方法 B PaDiM |
|------|------------|-------------|
| 图像级 AUROC | ~0.85 | ~0.97+ |
| 像素级 AUROC | ~0.80 | ~0.94+ |
| 推理耗时 | 数百 ms/张 | 数十 ms/张 |

**结论**：深度方法在精度与速度上通常都优于手工纹理特征，且定位更精确，符合
“仅用合格品训练、可定位缺陷区域”的要求；传统方法无需 GPU 训练、可解释性强，作为基线对比。
