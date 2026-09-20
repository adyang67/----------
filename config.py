# -*- coding: utf-8 -*-
"""全局配置：路径、数据集类别、两种方法的超参数。

所有脚本都可 `import config` 读取这里的默认值，
也可以用命令行参数覆盖类别等关键项。
"""
import os

# ---------- 路径 ----------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MVTEC_DIR = os.path.join(DATA_DIR, "mvtec_anomaly_detection")
RESULTS_DIR = os.path.join(ROOT_DIR, "results")

# ---------- 数据集 ----------
CATEGORY = "screw"      # MVTec AD 类别，可选：screw / metal_nut / capsule / leather / tile / wood ...
IMAGE_SIZE = 224        # 统一 resize 到 (IMAGE_SIZE, IMAGE_SIZE)

# ---------- 方法 A：传统路线（LBP + HOG + Gabor + One-Class SVM） ----------
PATCH_SIZE = 32         # 纹理块大小
STRIDE = 8              # 滑窗步长（越小定位越精细，但越慢）
LBP_P = 8               # LBP 采样点数量
LBP_R = 1               # LBP 邻域半径
GABOR_FREQS = (0.1, 0.3)           # Gabor 中心频率（低频/高频）
GABOR_THETAS = (0, 45, 90, 135)    # Gabor 方向（度）
MAX_TRAIN_PATCHES = 20000          # One-Class SVM 训练样本上限（随机下采样）
OCSVM_NU = 0.01                    # One-Class SVM 的 nu（异常比例上界）

# ---------- 方法 B：深度路线（PaDiM） ----------
BACKBONE = "resnet18"    # resnet18（CPU 友好）或 wide_resnet50_2（更准但更重）
PADIM_LAYERS = (1, 2, 3)  # 使用 ResNet 的 layer1 / layer2 / layer3
PADIM_DIMS = 128          # 随机投影降维后的嵌入维度（越大越准、越占内存）
PADIM_EPS = 0.01          # 协方差正则化（避免奇异）
SMOOTH_SIGMA = 4          # 异常图高斯平滑强度

SEED = 42


def ensure_dirs():
    for d in (DATA_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)
